import shutil

import mediapy
from PIL import Image, ImageDraw
import os.path
from enum import Enum
from pathlib import Path
import wandb
import time

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import trimesh
from pytorch3d.io import load_obj
from pytorch3d.ops import knn_points, knn_gather
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision.transforms.functional import gaussian_blur
from time import time


import pyvista as pv
import dreifus
from dreifus.matrix import Pose, Intrinsics, CameraCoordinateConvention, PoseType
from dreifus.pyvista import add_camera_frustum, render_from_camera

from pixel3dmm import env_paths
from pixel3dmm.tracking import util
from pixel3dmm.tracking.losses import UVLoss
from pixel3dmm.tracking import nvdiffrast_util
from pixel3dmm.tracking.renderer_nvdiffrast import NVDRenderer
from pixel3dmm import env_paths
from pixel3dmm.tracking.flame.FLAME import FLAME
from pixel3dmm.utils.misc import tensor2im
from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix, matrix_to_rotation_6d, euler_angles_to_matrix
from pixel3dmm.utils.drawing import plot_points


def timeit(t0, tag):
    t1 = time()
    #print(f'[PROFILER]: {tag} took {t1-t0} seconds')
    return t1


os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
rank = 42
torch.manual_seed(rank)
torch.cuda.manual_seed(rank)
cudnn.benchmark = True
np.random.seed(rank)
I = torch.eye(3)[None].cuda().detach()
I6D = matrix_to_rotation_6d(I)

left_iris_flame = [4597, 4542, 4510, 4603, 4570]
right_iris_flame = [4051, 3996, 3964, 3932, 4028]
left_iris_mp = [468, 469, 470, 471, 472]
right_iris_mp = [473, 474, 475, 476, 477]


torch.set_float32_matmul_precision('high')

class View(Enum):
    GROUND_TRUTH = 1
    COLOR_OVERLAY = 2
    SHAPE_OVERLAY = 4
    SHAPE = 8
    LANDMARKS = 16
    HEATMAP = 32
    DEPTH = 64


def get_intrinsics(focal_length, principal_point, use_hack : bool = True, size : int = 512):
    intrinsics = torch.eye(3)[None, ...].float().cuda().repeat(focal_length.shape[0], 1,1 )
    intrinsics[:, 0, 0] = focal_length.squeeze() * size
    intrinsics[:, 1, 1] = focal_length.squeeze() * size
    intrinsics[:, :2, 2] = size/2+0.5 + principal_point * (size/2+0.5)

    if use_hack:
        intrinsics[:, 0:1, 2:3] = size - intrinsics[:, 0:1, 2:3]  # TODO fix this hack

    return intrinsics



def get_extrinsics(R_base, t_base):
    timestep = 0
    w2c_openGL = torch.eye(4)[None, ...].float().cuda()
    w2c_openGL[:, :3, :3] = R_base[timestep]
    w2c_openGL[:, :3, 3] = t_base[timestep]
    return w2c_openGL


def project_points_screen_space(points3d, focal_length, principal_point, R_base, t_base, size : int = 512):
    # construct camera matrices
    intrinsics = get_intrinsics(focal_length, principal_point, size=size)
    w2c_openGL = get_extrinsics(R_base, t_base).repeat(focal_length.shape[0], 1, 1)

    B = points3d.shape[0]
    reps_extr = B if w2c_openGL.shape[0] == 1 else 1
    reps_intr = B if intrinsics.shape[0] == 1 else 1
    # apply w2c transformation
    lmk68_cam_space = torch.bmm(
        torch.cat([points3d, torch.ones_like(points3d[..., :1])], dim=-1),
        w2c_openGL.permute(0, 2, 1).repeat(reps_extr, 1, 1))

    # project from cam_space to screen_space
    lmk68_cam_space_prime = lmk68_cam_space[..., :3] / -lmk68_cam_space[..., [2]]
    lmk68_screen_space = (-1) * torch.bmm(lmk68_cam_space_prime, intrinsics.permute(0, 2, 1).repeat(reps_intr, 1, 1))[..., :2]
    lmk68_screen_space = torch.stack([size - 1 - lmk68_screen_space[..., 0], lmk68_screen_space[..., 1], lmk68_cam_space[..., 2]], dim=-1)
    return lmk68_screen_space


WFLW_2_iBUG68 = np.array(
                [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 33, 34, 35, 36, 37, 42, 43, 44, 45, 46, 51,
                 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63, 64, 65, 67, 68, 69, 71, 72, 73, 75, 76, 77, 78, 79, 80, 81, 82,
                 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95])

WFLW_2_iBUG68 = torch.from_numpy(WFLW_2_iBUG68).cuda()

COMPILE = True


if COMPILE:
    project_points_screen_space = torch.compile(project_points_screen_space)


class Tracker(object):
    def __init__(self, config,
                 device='cuda:0',
                 ):
        self.config = config
        self.device = device
        self.actor_name = self.config.video_name
        DATA_FOLDER = f'{env_paths.PREPROCESSED_DATA}/{self.actor_name}'
        self.MAX_STEPS = min(len([f for f in os.listdir(f'{DATA_FOLDER}/cropped/') if f.endswith('.jpg') or f.endswith('.png')]) - self.config.start_frame, 1000)
        self.FRAME_SKIP = 1
        self.BATCH_SIZE = min(self.config.batch_size, self.MAX_STEPS)

        print(f'''
                <<<<<<<< INITIALIZING TRACKER INSTANCE FOR {self.actor_name} >>>>>>>>
                ''')



        self.mirror_order = torch.from_numpy(np.load(f'{env_paths.MIRROR_INDEX}')).long().cuda()

        self.uv_loss_fn = UVLoss(stricter_mask=self.config.uv_loss.stricter_uv_mask,
                                 delta_uv= self.config.uv_loss.delta_uv,
                                 dist_uv=self.config.uv_loss.dist_uv)

        if COMPILE:
            self.uv_loss_fn.compute_loss = torch.compile(self.uv_loss_fn.compute_loss)



        self.actor_name = self.actor_name + f'_nV{config.num_views}'


        if config.no_lm:
            self.actor_name = self.actor_name + '_noLM'
        if config.no_pho:
            self.actor_name = self.actor_name + '_noPho'


        if self.config.ignore_mica:
            self.actor_name = self.actor_name + '_noMICA'

        if self.config.flame2023:
            self.actor_name = self.actor_name + '_FLAME23'



        if self.config.uv_map_super > 0:
            self.actor_name = self.actor_name + f'_uv{self.config.uv_map_super}'
        if self.config.normal_super > 0:
            self.actor_name = self.actor_name + f'_n{self.config.normal_super}'
        if self.config.normal_super_can > 0:
            self.actor_name = self.actor_name + f'_nc{self.config.normal_super_can}'


        self.global_step = 0

        self.no_sh = config.no_sh
        self.no_lm = config.no_lm
        self.no_pho = config.no_pho

        # Latter will be set up
        self.frame = 0
        self.is_initializing = False
        self.image_size = torch.tensor([[config.image_size[0], config.image_size[1]]]).cuda()
        if hasattr(self.config, 'output_folder'):
            self.save_folder = self.config.output_folder
        else:
            self.save_folder = env_paths.TRACKING_OUTPUT
        self.output_folder = os.path.join(self.save_folder, self.actor_name)
        self.checkpoint_folder = os.path.join(self.save_folder, self.actor_name, "checkpoint")
        self.mesh_folder = os.path.join(self.save_folder, self.actor_name, "mesh")
        self.create_output_folders()
        self.writer = SummaryWriter(log_dir=self.save_folder + self.actor_name + '/logs')

        self.cam_pose_nvd = {}
        self.R_base = {}
        self.t_base = {}

        flame_mesh_mask = np.load(f'{env_paths.FLAME_ASSETS}/FLAME2020/FLAME_masks/FLAME_masks.pkl', allow_pickle=True, encoding='latin1')
        self.vertex_face_mask = torch.from_numpy(flame_mesh_mask['face']).cuda().long()


        self.setup_renderer()


        self.intermediate_exprs = []
        self.intermediate_Rs = []
        self.intermediate_ts = []
        self.intermediate_eyes = []
        self.intermediate_eyelids = []
        self.intermediate_jaws = []
        self.intermediate_necks = []
        self.intermediate_fls = []
        self.intermediate_pps = []

        self.cached_data = {}




    def get_image_size(self):
        return self.image_size[0][0].item(), self.image_size[0][1].item()


    def create_output_folders(self):
        Path(self.save_folder).mkdir(parents=True, exist_ok=True)
        Path(self.checkpoint_folder).mkdir(parents=True, exist_ok=True)
        Path(self.mesh_folder).mkdir(parents=True, exist_ok=True)


    def setup_renderer(self):
        mesh_file = f'{env_paths.head_template}'
        self.config.image_size = self.get_image_size()
        self.flame = FLAME(self.config).to(self.device)
        self.flame.vertex_face_mask = self.vertex_face_mask


        if COMPILE:
            self.flame = torch.compile(self.flame)
            self.opt_pre = torch.compile(self.opt_pre)
            self.opt_post = torch.compile(self.opt_post)
            self.actual_smooth = torch.compile(self.actual_smooth)


        self.diff_renderer = NVDRenderer(self.config.size,
                                         obj_filename=mesh_file,
                                         no_sh=self.no_sh,
                                         white_bg= True,
                                         ).to(self.device)


        self.faces = load_obj(mesh_file)[1]


    def save_checkpoint(self, frame_id, selected_frames = None):

        if selected_frames is None:
            exp = self.exp
            eyes = self.eyes
            eyelids = self.eyelids
            R = self.R
            t = self.t
            jaw = self.jaw
            neck = self.neck
            focal_length = self.focal_length
            principal_point = self.principal_point
        else:
            exp = self.exp(selected_frames)
            eyes = self.eyes(selected_frames)
            eyelids = self.eyelids(selected_frames)
            R = self.R(selected_frames)
            t = self.t(selected_frames)
            jaw = self.jaw(selected_frames)
            neck = self.neck(selected_frames)
            if self.config.global_camera:
                focal_length = self.focal_length
                principal_point = self.principal_point
            else:
                focal_length = self.focal_length(selected_frames)
                principal_point = self.principal_point(selected_frames)

        frame = {
            'flame': {
                'exp': exp.clone().detach().cpu().numpy(),
                'shape': self.shape.clone().detach().cpu().numpy(),
                'eyes': eyes.clone().detach().cpu().numpy(),
                'eyelids': eyelids.clone().detach().cpu().numpy(),
                'jaw': jaw.clone().detach().cpu().numpy(),
                'neck': neck.clone().detach().cpu().numpy(),
                'R': R.clone().detach().cpu().numpy(),
                'R_rotation_matrix': rotation_6d_to_matrix(R).detach().cpu().numpy(),
                't': t.clone().detach().cpu().numpy(),
            },
            'img_size': self.image_size.clone().detach().cpu().numpy()[0],
            'frame_id': frame_id,
            'global_step': self.global_step
        }

        cam_params = {
            f'R_base_{serial}': self.R_base[serial].clone().detach().cpu().numpy() for serial in self.R_base.keys()
        }
        cam_pos = {
                    f't_base_{serial}': self.t_base[serial].clone().detach().cpu().numpy() for serial in self.R_base.keys()
                }
        intr = {
                    'fl': focal_length.clone().detach().cpu().numpy(),
                    'pp': principal_point.clone().detach().cpu().numpy(),
                    }
        cam_params.update(cam_pos)
        cam_params.update(intr)
        frame.update(
            {
                f'camera': cam_params
            }
        )
        bs = exp.shape[0]
        vertices, lmks_posed, joint_transforms, vertices_can_wexpr, vertices_noneck = self.flame(cameras=torch.inverse(self.R_base[0])[:1, ...].repeat(bs, 1, 1),
                   shape_params=self.shape[:1, ...].repeat(bs, 1),
                   expression_params=exp,
                   eye_pose_params=eyes,
                   jaw_pose_params=jaw,
                   neck_pose_params=neck,
                   rot_params_lmk_shift=R,
                   eyelid_params=eyelids,
        )
        _, lmks, _, vertices_can_can, _ = self.flame(cameras=torch.inverse(self.R_base[0])[:1, ...].repeat(bs, 1, 1),
            shape_params=self.shape[:1, ...].repeat(bs, 1),
            )
        frame.update(
            {
                f'joint_transforms': joint_transforms.detach().cpu().numpy(),
            }
        )

        f = self.diff_renderer.faces[0].cpu().numpy()
        for b_i in range(bs):

            v = vertices[b_i].cpu().numpy()

            if self.config.save_meshes:
                trimesh.Trimesh(faces=f, vertices=v, process=False).export(f'{self.mesh_folder}/{frame_id:05d}.ply')
            torch.save(frame, f'{self.checkpoint_folder}/{frame_id:05d}.frame')

            selction_indx = np.array([36, 39, 42, 45, 33, 48, 54])
            _lmks = lmks[b_i].detach().squeeze().cpu().numpy()

            if self.config.save_landmarks:
                np.save(f'{self.mesh_folder}/landmarks_{frame_id}_{b_i}.npy', _lmks[selction_indx])



        if frame_id == self.config.start_frame and self.config.save_meshes:
            faces = self.diff_renderer.faces[0].cpu().numpy()
            #trimesh.Trimesh(faces=faces, vertices=vertices_can[0].detach().cpu().numpy(), process=False).export(f'{self.mesh_folder}/canonical.ply')
            trimesh.Trimesh(faces=faces, vertices=vertices_can_can[0].detach().cpu().numpy(), process=False).export(f'{self.mesh_folder}/canonical.ply')
        if self.config.save_landmarks:
            lmks = lmks.detach().squeeze().cpu().numpy()
            np.save(f'{self.mesh_folder}/ibug68_{frame_id}.ply', lmks)
            selction_indx = np.array([36, 39, 42, 45, 33, 48, 54])
            np.save(f'{self.mesh_folder}/now_{frame_id}.ply', lmks[selction_indx])




    def get_heatmap(self, values):
        l2 = tensor2im(values)
        l2 = cv2.cvtColor(l2, cv2.COLOR_RGB2BGR)
        #l2[l2 > 125] = 125
        #l2 = cv2.normalize(l2, None, 0, 255, cv2.NORM_MINMAX)
        #l2[l2 > 35] = 35
        #l2 = cv2.normalize(l2, None, 0, 255, cv2.NORM_MINMAX)
        l2 = l2 - 127
        max_err = 25
        l2[l2>max_err] = max_err
        l2 = ((l2 / max_err)*255).astype(np.uint8)
        heatmap = cv2.applyColorMap(l2, cv2.COLORMAP_JET) #/ 255.
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.
        #heatmap = heatmap ** (1/3)
        #Image.fromarray((heatmap*255).astype(np.uint8)).show()
        #exit()
        #heatmap = cv2.cvtColor(cv2.addWeighted(heatmap, 0.75, l2, 0.25, 0).astype(np.uint8), cv2.COLOR_BGR2RGB) / 255.
        heatmap = torch.from_numpy(heatmap).permute(2, 0, 1)

        return heatmap


    def to_cuda(self, batch, unsqueeze=False):
        for key in batch.keys():
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(self.device)
                if unsqueeze:
                    batch[key] = batch[key][None]

        return batch


    def create_parameters(self, timestep, mica_shape):
        bz = 1
        pose_mat = np.eye(4)
        pose_mat[2, 3] = -1

        opencv_w2c_pose = Pose(pose_mat, camera_coordinate_convention=dreifus.matrix.CameraCoordinateConvention.OPEN_CV)
        opencv_w2c_pose = opencv_w2c_pose.change_pose_type(dreifus.matrix.PoseType.CAM_2_WORLD)

        opencv_w2c_pose.look_at(np.zeros(3), np.array([0, 1, 0]))

        opencv_w2c_pose = opencv_w2c_pose.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM)
        self.debug_pose_init = opencv_w2c_pose.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM).copy()


        self.shape = mica_shape.detach().clone()
        self.mica_shape = mica_shape.detach().clone()
        if self.config.ignore_mica:
            self.shape = torch.zeros_like(self.shape)
            self.mica_shape = torch.zeros_like(self.mica_shape)


        cam_pose = opencv_w2c_pose
        cam_pose = cam_pose.change_pose_type(dreifus.matrix.PoseType.CAM_2_WORLD)
        cam_pose_nvd = cam_pose.copy()
        cam_pose_nvd = cam_pose_nvd.change_camera_coordinate_convention(new_camera_coordinate_convention=dreifus.matrix.CameraCoordinateConvention.OPEN_GL)
        cam_pose_nvd = cam_pose_nvd.change_pose_type(dreifus.matrix.PoseType.WORLD_2_CAM)
        self.cam_pose_nvd[timestep] = torch.from_numpy(cam_pose_nvd.copy()).float().cuda()

        R = torch.from_numpy(cam_pose_nvd.get_rotation_matrix()).unsqueeze(0).cuda()
        T = torch.from_numpy(cam_pose_nvd.get_translation()).unsqueeze(0).cuda()
        R.requires_grad = True
        T.requires_grad = True

        self.R_base[timestep] = R
        self.t_base[timestep] = T


        init_f = 2000 * self.config.size/512
        self.focal_length = torch.tensor([[init_f/self.config.size]]).float().to(self.device)
        self.principal_point = torch.tensor([[0, 0]]).float().to(self.device)
        self.focal_length.requires_grad = True
        self.principal_point.requires_grad = True
        intrinsics = torch.tensor([[init_f, 0, self.config.size//2],
                               [0, init_f, self.config.size//2],
                               [0, 0, 1]]).float().cuda()
        proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics,
                                          znear=0.1, zfar=10,
                                          width=self.config.size,
                                          height=self.config.size)

        self.r_mvps = {}
        for serial in self.cam_pose_nvd.keys():
            self.r_mvps[serial] = ( proj_512 @ self.cam_pose_nvd[serial] )[None, ...]




        n_timesteps = 1
        expression_params = np.zeros([n_timesteps, 100])
        jaw_params = np.zeros([n_timesteps, 3])
        neck_params = np.zeros([n_timesteps, 3])
        flame_R = torch.from_numpy(np.stack([np.eye(3) for _ in range(n_timesteps)], axis=0))
        flame_t = torch.from_numpy(np.stack([np.zeros([3]) for _ in range(n_timesteps)], axis=0))
        self.R = nn.Parameter(matrix_to_rotation_6d(flame_R.float().to(self.device)))
        self.t = nn.Parameter(flame_t.float().to(self.device))

        self.expression_params = expression_params
        self.jaw_params = jaw_params.astype(np.float32)
        self.neck_params = neck_params.astype(np.float32)

        self.shape = nn.Parameter(self.mica_shape.detach().clone())

        self.texture_observation_mask = None

        self.exp = nn.Parameter(torch.from_numpy(self.expression_params[[0] + self.config.keyframes,..., :]).float().to(self.device))
        self.jaw = nn.Parameter(matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[[0]+ self.config.keyframes,..., :]).cuda(), 'XYZ')))
        self.neck = nn.Parameter(matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.neck_params[[0]+ self.config.keyframes,..., :]).cuda(), 'XYZ')))




        self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1).repeat(1+len(self.config.keyframes), 1) )
        self.eyelids = nn.Parameter(torch.zeros(1+len(self.config.keyframes), 2).float().to(self.device))



    def parse_mask(self, ops, batch, visualization=False):
        result = ops['mask_images_rendering']

        if visualization:
            result = ops['mask_images']

        return result.detach()



    def clone_params_keyframes_all(self, freeze_id : bool = False, is_joint : bool = False, freeze_cam : bool = False,
                                   include_neck : bool = False):

        lr_scale = 1.0
        lr_scale_id_related = 1.0
        if freeze_id:
            lr_scale_id_related = 0.1


        params = [
            {'params': [self.exp], 'lr': self.config.lr_exp * lr_scale, 'name': ['exp']},  # 0.025
            {'params': [self.eyes], 'lr': 0.005 * lr_scale, 'name': ['eyes']},
            # {'params': [self.eyelids.clone())], 'lr': 0.001, 'name': ['eyelids']},
            {'params': [self.eyelids], 'lr': 0.002 * lr_scale, 'name': ['eyelids']},
            # {'params': [self.sh.clone())], 'lr': 0.01, 'name': ['sh']},
            {'params': [self.t], 'lr': self.config.lr_t * lr_scale, 'name': ['t']},
            #{'params': [self.t.clone())], 'lr': 0.005 * lr_scale, 'name': ['t']},
            {'params': [self.R], 'lr': self.config.lr_R * lr_scale, 'name': ['R']},
            #{'params': [self.R.clone())], 'lr': 0.003 * lr_scale, 'name': ['R']},
            # {'params': [self.tex.clone())], 'lr': 0.001, 'name': ['tex']},
            # {'params': [self.principal_point.clone())], 'lr': 0.001, 'name': ['principal_point']},
            # {'params': [self.focal_length.clone())], 'lr': 0.001, 'name': ['focal_length']}
        ]
        #params.append({'params': [self.shape.clone())], 'lr': self.config.lr_id * lr_scale, 'name': ['shape']})
        if not freeze_id:
            if is_joint:
                params.append({'params': [self.shape], 'lr': self.config.lr_id * lr_scale * 1, 'name': ['shape']})
            else:
                params.append({'params': [self.shape], 'lr': self.config.lr_id * lr_scale, 'name': ['shape']})
        #params.append({'params': [self.shape], 'lr': 0.0, 'name': ['shape']})
        params.append({'params': [self.jaw], 'lr': self.config.lr_jaw * lr_scale, 'name': ['jaw']})
        if include_neck:
            params.append({'params': [self.neck], 'lr': self.config.lr_neck, 'name': ['neck']})

        # params.append({'params': [self.t], 'lr': 0.001, 'name': ['translation']})
        # params.append({'params': [self.R], 'lr': 0.005, 'name': ['rotation']})
        # params.append({'params': [self.focal_length, self.principal_point], 'lr': 0.01*lr_scale, 'name': ['camera_params']})
        #if not self.config.load_intr:
        if not freeze_cam:
            params.append({'params': [self.focal_length], 'lr': self.config.lr_f * lr_scale_id_related, 'name': ['camera_params']})
            params.append({'params': [self.principal_point], 'lr': self.config.lr_pp * lr_scale_id_related, 'name': ['camera_params']})

        return params


    def clone_params_keyframes_all_joint(self, freeze_id : bool = False, is_joint : bool = False,
                                         include_neck : bool = False):

        lr_scale = 1.0
        lr_scale_id_related = 1.0
        if freeze_id:
            lr_scale_id_related = 0.1
        params = [
            {'params': self.exp.parameters(), 'lr': self.config.lr_exp * lr_scale, 'name': ['exp']},  # 0.025
            {'params': self.eyes.parameters(), 'lr': 0.005 * lr_scale, 'name': ['eyes']},
            {'params': self.eyelids.parameters(), 'lr': 0.002 * lr_scale, 'name': ['eyelids']},
            {'params': self.t.parameters(), 'lr': self.config.lr_t * lr_scale, 'name': ['t']},
            {'params': self.R.parameters(), 'lr': self.config.lr_R * lr_scale, 'name': ['R']},
        ]

        params.append({'params': self.jaw.parameters(), 'lr': self.config.lr_jaw * lr_scale, 'name': ['jaw']})
        if include_neck:
            params.append({'params': self.neck.parameters(), 'lr':  self.config.lr_neck, 'name': ['jaw']})

        if not self.config.global_camera:
            params.append({'params': self.focal_length.parameters(), 'lr': self.config.lr_f * lr_scale_id_related,
                           'name': ['camera_params']})
            params.append({'params': self.principal_point.parameters(), 'lr': self.config.lr_pp * lr_scale_id_related,
                           'name': ['camera_params']})
        #params.append({'params': [self.shape], 'lr': self.config.lr_id * lr_scale * 1, 'name': ['shape']})
        return params


    def reduce_loss(self, losses):
        all_loss = 0.
        for key in losses.keys():
            all_loss = all_loss + losses[key]
        losses['all_loss'] = all_loss
        return all_loss


    def optimize_camera(self, batch, steps=2000, is_first_frame : bool = False
                        ):
        batch = self.to_cuda(batch)

        images, landmarks, lmk_mask = self.parse_landmarks(batch)
        h, w = images.shape[2:4]
        num_keyframes = 1

        uv_mask = batch["uv_mask"]
        uv_map = batch["uv_map"] if "uv_map" in batch else None

        if uv_map is not None:
            uv_map[(1 - uv_mask[:, :, :, :]).bool()] = 0


        self.focal_length.requires_grad = True
        self.principal_point.requires_grad = True

        lr_mult = 1.0

        params = [
            {'params': [self.t], 'lr': lr_mult*0.001}, ##0.05},
            {'params': [self.R], 'lr': lr_mult*0.005}, #0.05},
                  ]

        if is_first_frame:
            params.append({'params': [self.focal_length], 'lr': 0.02})
            params.append({'params': [self.principal_point], 'lr': 0.0001})

        optimizer = torch.optim.Adam(params)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=int(steps*0.75),
                                                    gamma=0.1)

        #self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE_OVERLAY]],
        #                frame_dst='/camera', save=False, dump_directly=True)

        t = tqdm(range(steps), desc='', leave=True, miniters=100)
        num_views = 1 #len(self.R_base.keys())
        bs = 1 #len(self.cam_serials) * num_keyframes

        for k in t:
            vertices_can, lmk68, lmkMP, vertices_can_can, vertices_noneck = self.flame(cameras=torch.inverse(self.R_base[0]),
                                              shape_params=self.shape if self.shape.shape[0] == bs else self.shape.repeat(bs, 1),
                                              expression_params=self.exp.repeat_interleave(num_views, dim=0),
                                              eye_pose_params=self.eyes.repeat_interleave(num_views, dim=0),
                                              jaw_pose_params=self.jaw.repeat_interleave(num_views, dim=0),
                                              neck_pose_params=self.neck.repeat_interleave(num_views, dim=0),
                                              rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(self.R)))).repeat_interleave(num_views, dim=0),
                                                )

            lmk68 = torch.einsum('bny,bxy->bnx', lmk68,
                                 rotation_6d_to_matrix(self.R.repeat_interleave(num_views, dim=0))) + self.t.repeat_interleave(num_views, dim=0).unsqueeze(1)
            verts = torch.einsum('bny,bxy->bnx', vertices_can,
                                 rotation_6d_to_matrix(
                                     self.R.repeat_interleave(num_views, dim=0))) + self.t.repeat_interleave(num_views,
                                                                                                             dim=0).unsqueeze(
                1)


            lmk68_screen_space = project_points_screen_space(lmk68, self.focal_length, self.principal_point, self.R_base, self.t_base, size=self.config.size)
            verts_screen_space = project_points_screen_space(verts, self.focal_length, self.principal_point, self.R_base, self.t_base, size=self.config.size)


            losses = {}
            losses['pp_reg'] = torch.sum(self.principal_point ** 2)
            if k <= steps // 2:
                losses['lmk68'] = util.lmk_loss(lmk68_screen_space[..., :2], landmarks[..., :2], [h, w], lmk_mask) * 3000

            if k == 0:
                self.uv_loss_fn.compute_corresp(uv_map)
            if k > steps // 2:
                uv_loss = self.uv_loss_fn.compute_loss(verts_screen_space)
                losses['uv_loss'] = uv_loss * 1000


            all_loss = 0.
            for key in losses.keys():
                all_loss = all_loss + losses[key]
            losses['all_loss'] = all_loss

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()


            scheduler.step()
            optimizer.zero_grad()

            intrinsics = get_intrinsics(self.focal_length, self.principal_point, use_hack=False, size=self.config.size)

            proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics[0],
                                                             znear=0.1, zfar=5,
                                                             width=self.config.size,
                                                             height=self.config.size)
            for serial in self.cam_pose_nvd.keys():
                extr = get_extrinsics(self.R_base[serial], self.t_base[serial])
                r_mvps = proj_512 @ extr
                self.r_mvps[serial] = r_mvps

            loss = all_loss.item()
            t.set_description(f'Loss for camera {loss:.4f}')
            self.frame += 1
            #if k % 100 == 0:
            #    self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE_OVERLAY, View.COLOR_OVERLAY]], frame_dst='/camera', save=False, dump_directly=True, is_camera=True)
        self.frame = 0


    @torch.compiler.disable
    def get_vars(self, is_joint, selected_frames):
        if not is_joint:
            exp = self.exp
            eyes = self.eyes
            eyelids = self.eyelids
            _R = self.R
            _t = self.t
            jaw = self.jaw
            neck = self.neck
            focal_length = self.focal_length
            principal_point = self.principal_point
        else:
            selected_frames = torch.from_numpy(selected_frames).long().cuda()
            exp = self.exp(selected_frames)
            eyes = self.eyes(selected_frames)
            eyelids = self.eyelids(selected_frames)
            _R = self.R(selected_frames)
            _t = self.t(selected_frames)
            jaw = self.jaw(selected_frames)
            neck = self.neck(selected_frames)
            if not self.config.global_camera:
                focal_length = self.focal_length(selected_frames)
                principal_point = self.principal_point(selected_frames)
            else:
                focal_length = self.focal_length
                principal_point = self.principal_point
        return exp, eyes, eyelids, _R, _t, jaw, neck, focal_length, principal_point


    @torch.compiler.disable
    def data_stuff(self, is_joint, iters, p, image_lmks68, lmk_mask, normal_map, normal_mask, uv_map, uv_mask, left_iris, right_iris, mask_left_iris, mask_right_iris):
        if is_joint:
            with torch.no_grad():
                if ((p < int(iters * 0.15) and (p % 2 == 0)) or not self.config.smooth) or self.config.is_discontinuous:
                    all_frames = np.array(
                        range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP))
                    selected_frames = np.sort(np.random.choice(np.arange(len(all_frames)), size=self.BATCH_SIZE,
                                                               replace=False))  # np.random.choice(
                else:
                    all_frames = np.array(
                        range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP))
                    start = np.min(all_frames)
                    end = np.max(all_frames)
                    rnd_start = np.random.randint(start, end)
                    assert (end - start + 1 )>= self.BATCH_SIZE
                    assert self.BATCH_SIZE % 2 == 0
                    if rnd_start - self.BATCH_SIZE // 2 < 0:
                        rnd_start = self.BATCH_SIZE // 2
                    if rnd_start + self.BATCH_SIZE // 2 + 1 > end:
                        rnd_start = end - self.BATCH_SIZE // 2 + 1
                    selected_frames = np.array(
                        list(range(rnd_start - self.BATCH_SIZE // 2, rnd_start + self.BATCH_SIZE // 2)))

                selected_frames_th = torch.from_numpy(selected_frames).long()
                batch = {k: self.cached_data[k][selected_frames_th, ...] for k in self.cached_data.keys()}
                images, landmarks, lmk_mask = self.parse_landmarks(batch)

                uv_mask = batch["uv_mask"]
                normal_mask = batch["normal_mask"]
                normal_map = batch["normals"] if "normals" in batch else None
                uv_map = batch["uv_map"] if "uv_map" in batch else None
                #TODO check if this was important in any way
                if uv_map is not None:
                    uv_map[(1 - uv_mask[:, :, :, :]).bool()] = 0

                num_views = len(self.R_base.keys())
                bs = batch['normals'].shape[0] * num_views

                image_lmks68 = landmarks
                if landmarks is not None:
                    left_iris = batch['left_iris']
                    right_iris = batch['right_iris']
                    mask_left_iris = batch['mask_left_iris']
                    mask_right_iris = batch['mask_right_iris']
        else:
            selected_frames = None
            bs = 1
            num_views = 1
            batch = None

        return selected_frames, batch, bs, num_views, image_lmks68, lmk_mask, normal_map, normal_mask, uv_map, uv_mask, left_iris, right_iris, mask_left_iris, mask_right_iris



    #TODO: could be improved by compiling all the actuall smooth loss stuff

    #@torch.compile
    def actual_smooth(self, variables, losses):
        reg_smooth_exp = (variables['exp'][:-1, :] - variables['exp'][1:, :]).square().mean()
        reg_smooth_eyes = (variables['eyes'][:-1, :] - variables['eyes'][1:, :]).square().mean()
        reg_smooth_eyelids = (variables['eyelids'][:-1, :] - variables['eyelids'][1:, :]).square().mean()
        reg_smooth_R = (variables['R'][:-1, :] - variables['R'][1:, :]).square().mean()
        reg_smooth_t = (variables['t'][:-1, :] - variables['t'][1:, :]).square().mean()
        reg_smooth_jaw = (variables['jaw'][:-1, :] - variables['jaw'][1:, :]).square().mean()
        reg_smooth_neck = (variables['neck'][:-1, :] - variables['neck'][1:, :]).square().mean()
        if not self.config.global_camera:
            reg_smooth_principal_point = (
                    variables['principal_point'][:-1, :] - variables['principal_point'][1:, :]).square().mean()
            reg_smooth_focal_length = (
                    variables['focal_length'][:-1, :] - variables['focal_length'][1:, :]).square().mean()
        else:
            reg_smooth_principal_point = torch.zeros_like(reg_smooth_jaw)
            reg_smooth_focal_length = torch.zeros_like(reg_smooth_jaw)
        losses['smooth/exp'] = reg_smooth_exp * self.config.reg_smooth_exp * self.config.reg_smooth_mult
        losses['smooth/eyes'] = reg_smooth_eyes * self.config.reg_smooth_eyes * self.config.reg_smooth_mult
        losses['smooth/eyelids'] = reg_smooth_eyelids * self.config.reg_smooth_eyelids * self.config.reg_smooth_mult
        losses['smooth/jaw'] = reg_smooth_jaw * self.config.reg_smooth_jaw * self.config.reg_smooth_mult
        losses['smooth/neck'] = reg_smooth_neck * self.config.reg_smooth_neck * self.config.reg_smooth_mult
        losses['smooth/R'] = reg_smooth_R * self.config.reg_smooth_R * self.config.reg_smooth_mult
        losses['smooth/t'] = reg_smooth_t * self.config.reg_smooth_t * self.config.reg_smooth_mult
        losses['smooth/principal_point'] = reg_smooth_principal_point * self.config.reg_smooth_pp * self.config.reg_smooth_mult
        losses['smooth/focal_length'] = reg_smooth_focal_length * self.config.reg_smooth_fl * self.config.reg_smooth_mult
        return losses

    @torch.compiler.disable
    def add_smooth_loss(self, losses, is_joint, p, iters, variables):
        if is_joint and self.config.smooth and ((p >= int(iters * 0.15) and (p % 2 == 1)) ):  # and p % 2 != 0 and False:
            losses = self.actual_smooth(variables, losses)

        return losses


    def opt_pre(self, is_joint, iters, p, no_lm, image_lmks68, lmk_mask, normal_mask, normal_map, uv_map, uv_mask, left_iris, right_iris, mask_left_iris, mask_right_iris):

        image_size = [self.config.size, self.config.size]

        selected_frames, batch, bs, num_views, image_lmks68, lmk_mask, normal_map, normal_mask, uv_map, uv_mask, left_iris, right_iris, mask_left_iris, mask_right_iris = self.data_stuff(is_joint, iters, p, image_lmks68, lmk_mask, normal_map, normal_mask, uv_map, uv_mask, left_iris, right_iris, mask_left_iris, mask_right_iris)

        self.diff_renderer.reset()
        losses = {}
        exp, eyes, eyelids, _R, _t, jaw, neck, focal_length, principal_point = self.get_vars(is_joint, selected_frames)

        variables = {
            'exp': exp,
            'eyes': eyes,
            'eyelids': eyelids,
            'R': _R,
            't': _t,
            'jaw': jaw,
            'neck': neck,
            'principal_point': principal_point,
            'focal_lenght': focal_length,
        }

        intrinsics = get_intrinsics(focal_length, principal_point, use_hack=False, size=self.config.size)

        proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics,
                                                         znear=0.1, zfar=5,
                                                         width=self.config.size,
                                                         height=self.config.size)
        for serial in self.cam_pose_nvd.keys():
            extr = get_extrinsics(self.R_base[serial], self.t_base[serial])
            r_mvps = torch.matmul(proj_512, extr.repeat(bs, 1, 1))
            self.r_mvps[serial] = r_mvps

        vertices_can, lmk68, lmkMP, vertices_can_can, vertices_noneck = self.flame(
            cameras=torch.inverse(self.R_base[0]).repeat(bs, 1, 1),
            shape_params=self.shape if self.shape.shape[0] == bs else self.shape.repeat(bs, 1).cuda(),
            expression_params=exp.repeat_interleave(num_views, dim=0),  # .repeat(bs, 1),
            eye_pose_params=eyes.repeat_interleave(num_views, dim=0),  # .repeat(bs, 1),
            jaw_pose_params=jaw.repeat_interleave(num_views, dim=0),  # .repeat(bs, 1),
            neck_pose_params=neck.repeat_interleave(num_views, dim=0),  # .repeat(bs, 1),
            eyelid_params=eyelids.repeat_interleave(num_views, dim=0),  # .repeat(bs, 1),
            rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(_R)))).repeat_interleave(
                num_views, dim=0),  # .repeat(bs, 1)
        )

        verts_can_can_mirrored = vertices_can_can[:, self.mirror_order, :]
        vertices_can_can_mirrored = torch.zeros_like(verts_can_can_mirrored)
        vertices_can_can_mirrored[:, :, 0] = -verts_can_can_mirrored[:, :, 0]
        vertices_can_can_mirrored[:, :, 1:] = verts_can_can_mirrored[:, :, 1:]
        mirror_loss = (vertices_can_can_mirrored - vertices_can_can).square().sum(-1)
        mirror_loss = mirror_loss.mean()

        lmk68 = torch.einsum('bny,bxy->bnx', lmk68,
                             rotation_6d_to_matrix(_R.repeat_interleave(num_views, dim=0))) + _t.repeat_interleave(
            num_views, dim=0).unsqueeze(1)

        vertices = torch.einsum('bny,bxy->bnx', vertices_can,
                                rotation_6d_to_matrix(_R.repeat_interleave(num_views, dim=0))) + _t.repeat_interleave(
            num_views, dim=0).unsqueeze(1)
        vertices_noneck = torch.einsum('bny,bxy->bnx', vertices_noneck,
                               rotation_6d_to_matrix(_R.repeat_interleave(num_views, dim=0))) + _t.repeat_interleave(
            num_views, dim=0).unsqueeze(1)

        proj_lmks68 = project_points_screen_space(lmk68, focal_length, principal_point, self.R_base, self.t_base,
                                                  size=self.config.size)
        proj_vertices = project_points_screen_space(vertices, focal_length, principal_point, self.R_base, self.t_base,
                                                    size=self.config.size)

        right_eye, left_eye = eyes[:, :6], eyes[:, 6:]


        # landmark loss
        if not no_lm:
            lmk_scale = 1.0  # 0.0001
            # Landmarks sparse term
            # losses[('loss/lmk_oval')] = util.oval_lmk_loss(proj_lmks68[..., :2], image_lmks68, image_size, lmk_mask) * self.config.w_lmks_oval * lmk_scale
            # losses['loss/lmk_68'] = util.lmk_loss(proj_lmks68[:, 17:, :2], image_lmks68[:, 17:, :], image_size, lmk_mask[:, 17:, :]) * self.config.w_lmks * lmk_scale
            # if self.config.use_eyebrows:
            # losses['loss/lmk_eyebrows'] = util.lmk_loss(proj_lmks68[:, 17:27, :2], image_lmks68[:, 17:27, :], image_size, lmk_mask[:, 17:27, :]) * self.config.w_lmks * lmk_scale * 5.0
            losses['loss/lmk_eye2'] = util.lmk_loss(proj_lmks68[:, 36:48, :2], image_lmks68[:, 36:48, :], image_size,
                                                    lmk_mask[:, 36:48,
                                                    :]) * self.config.w_lmks * lmk_scale * 5 #10  # 0 #2.0 #0.5 #0.0 #100
            if self.config.use_mouth_lmk:
                losses['loss/lmk_mouth'] = util.lmk_loss(proj_lmks68[:, 48:68, :2], image_lmks68[:, 48:68, :],
                                                         image_size,
                                                         lmk_mask[:, 48:68, :]) * self.config.w_lmks_mouth * lmk_scale * 0.25
                losses['loss/lmk_mouth_closure'] = util.mouth_closure_lmk_loss(proj_lmks68[..., :2], image_lmks68,
                                                                               image_size,
                                                                               lmk_mask) * self.config.w_lmks_mouth * lmk_scale * 2.5

            losses['loss/lmk_eye'] = util.eye_closure_lmk_loss(proj_lmks68[..., :2], image_lmks68, image_size,
                                                               lmk_mask) * self.config.w_lmks_lid * lmk_scale * 500  # 0 #500 #0.0 #10
            losses['loss/lmk_iris_left'] = util.lmk_loss(proj_vertices[:, left_iris_flame[:1], ..., :2], left_iris,
                                                         image_size,
                                                         mask_left_iris) * self.config.w_lmks_iris * lmk_scale * 50.00
            losses['loss/lmk_iris_right'] = util.lmk_loss(proj_vertices[:, right_iris_flame[:1], ..., :2], right_iris,
                                                          image_size,
                                                          mask_right_iris) * self.config.w_lmks_iris * lmk_scale * 50.0

        # Reguralizers
        losses['reg/exp'] = torch.sum(exp ** 2, dim=-1).mean() * self.config.w_exp
        losses['reg/sym'] = torch.sum((right_eye - left_eye) ** 2, dim=-1).mean() * 0.1  # 8.0 #*5.0
        losses['reg/jaw'] = torch.sum((I6D - jaw) ** 2, dim=-1).mean() * self.config.w_jaw
        losses['reg/neck'] = torch.sum((I6D - neck) ** 2, dim=-1).mean() * self.config.w_neck
        # losses['reg/eye_lids'] = torch.sum((eyelids[:, 0] - eyelids[:, 1]) ** 2, dim=-1).mean() * 0.1
        losses['reg/eye_left'] = torch.sum((I6D - left_eye) ** 2, dim=-1).mean() * 0.01
        losses['reg/eye_right'] = torch.sum((I6D - right_eye) ** 2, dim=-1).mean() * 0.01

        losses['reg/shape'] = torch.sum((self.shape - self.mica_shape) ** 2, dim=-1).mean() * self.config.w_shape
        losses['reg/shape_general'] = torch.sum((self.shape) ** 2, dim=-1).mean() * self.config.w_shape_general

        losses['reg/mirror'] = mirror_loss * 5000
        if not (self.config.n_fine and p >= iters // 2):
            losses['reg/pp'] = torch.sum(principal_point ** 2, dim=-1).mean()

        return batch, losses, vertices, vertices_noneck, vertices_can, vertices_can_can, proj_vertices, proj_lmks68, selected_frames, variables, num_views, normal_mask, normal_map, uv_map, uv_mask


    def opt_post(self, variables, ops, proj_vertices, proj_lmks68, batch, is_joint, is_first_step, losses, uv_map, selected_frames, p, iters, num_views, normal_mask, normal_map):
        grabbed_depth = ops['actual_rendered_depth'][:, 0,
                                                     torch.clamp(proj_vertices[:, :, 1].long(), 0,
                                                                 self.config.size - 1),
                                                     torch.clamp(proj_vertices[:, :, 0].long(), 0,
                                                                 self.config.size - 1),
        ][:, 0, :]

        is_visible_verts_idx = grabbed_depth < (proj_vertices[:, :, 2] + 1e-2)
        if not self.config.occ_filter:
            is_visible_verts_idx = torch.ones_like(is_visible_verts_idx)

        valid_bg_classes = batch['valid_bg']  # bg-class or neck-class
        if self.config.sil_super > 0:
            if is_joint or (not is_first_step):  # and p > 50 and p < int(iters*0.85): # 100
                # losses['loss/sil'] =((1-upper_forehead[:, None, :, :]) * (batch['fg_mask'] - ops['fg_images'])).abs().mean() * self.config.sil_super#0
                losses['loss/sil'] = ((valid_bg_classes[:, None, :, :]) * (
                            batch['fg_mask'] - ops['fg_images'])).abs().mean() * self.config.sil_super  # 0
            else:
                losses['loss/sil'] = ((valid_bg_classes[:, None, :, :]) * (
                            batch['fg_mask'] - ops['fg_images'])).abs().mean() * self.config.sil_super / 10  # 0

        if self.config.uv_map_super:  # and p > iters // 2:
            gt_uv = uv_map[:, :2, :, :].permute(0, 2, 3, 1)
            if self.config.uv_l2:
                uv_loss = ((gt_uv - ops['uv_images']) * batch["uv_mask"][:, 0, ...].unsqueeze(-1)).square().mean() * 100
            else:
                uv_loss = ((gt_uv - ops['uv_images']) * batch["uv_mask"][:, 0, ...].unsqueeze(-1)).abs().mean()
            # TODO: outlier filtering!!!
            losses['loss/uv_pixel'] = uv_loss * self.config.uv_map_super

        if self.config.uv_map_super > 0.0:  # and (p < iters // 2 or self.config.keep_uv) and not self.config.no2d_verts:
            # uv_loss = get_uv_loss(uv_map, proj_vertices)
            if self.uv_loss_fn.gt_2_verts is None:
                self.uv_loss_fn.compute_corresp(uv_map, selected_frames=selected_frames)


            uv_loss = self.uv_loss_fn.compute_loss(proj_vertices, selected_frames=selected_frames, uv_map=uv_map,
                                                   l2_loss=self.config.uv_l2, is_visible_verts_idx=is_visible_verts_idx)
            losses['loss/uv'] = uv_loss * self.config.uv_map_super  # 000

        skip_normals = False
        if self.config.n_fine and p < iters // 2:
            skip_normals = True

        if (self.config.normal_super > 0.0 or self.config.normal_super_can > 0.0) and not skip_normals:
            # normal_loss_map = normal_loss_map * dilated_eye_mask[:, 0, ...] * (1 - ops['mask_images_eyes_region'][:, 0, ...])
            # use dilated eye mask only
            # maybe also applie eyemask in image not rendering
            dilated_eye_mask = 1 - (gaussian_blur(ops['mask_images_eyes'],
                                                  [self.config.normal_mask_ksize, self.config.normal_mask_ksize],
                                                  sigma=[self.config.normal_mask_ksize,
                                                         self.config.normal_mask_ksize]) > 0).float()
            pred_normals = ops['normal_images']  # 1 3 512 512 normals in world space
            rot_mat = rotation_6d_to_matrix(variables["R"].repeat_interleave(num_views, dim=0))  # 1 3 3

            pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals)
            if normal_map is not None:
                l_map = (normal_map - pred_normals_flame_space)
                valid = ((l_map.abs().sum(dim=1) / 3) < self.config.delta_n).unsqueeze(1)
                normal_loss_map = l_map * valid.float() * normal_mask * dilated_eye_mask
                if self.config.normal_l2:
                    losses['loss/normal'] = normal_loss_map.square().mean() * self.config.normal_super
                else:
                    losses['loss/normal'] = normal_loss_map.abs().mean() * self.config.normal_super
            else:
                losses['loss/normal'] = 0.0


        if not self.config.is_discontinuous:
            # smoothness loss
            losses = self.add_smooth_loss(losses, is_joint, p, iters, variables)

        all_loss = self.reduce_loss(losses)

        return all_loss


    def optimize_color(self, batch, params_func,
                       no_lm : bool = False,
                       save_timestep=0,
                       is_joint : bool = False,
                       is_first_step : bool = False,
                       ):

        iters = self.config.iters
        if not is_joint:
            images, landmarks, lmk_mask = self.parse_landmarks(batch)

            uv_mask = batch["uv_mask"]
            normal_mask = batch["normal_mask"]

            normal_map = batch["normals"] if "normals" in batch else None
            uv_map = batch["uv_map"] if "uv_map" in batch else None

            if uv_map is not None:
                uv_map[(1-uv_mask[:, :, :, :]).bool()] = 0


        # Optimizer per step
        if is_joint:
            optimizer = torch.optim.SparseAdam(params_func())
            params_global = [
                {'params': [self.shape], 'lr': self.config.lr_id * 1.0, 'name': ['shape']}
            ]
            if self.config.global_camera:
                params_global.append({'params': [self.focal_length], 'lr': self.config.lr_f * 1.0,
                               'name': ['camera_params']})
                params_global.append({'params': [self.principal_point], 'lr': self.config.lr_pp * 1.0,
                               'name': ['camera_params']})
            optimizer_id = torch.optim.Adam(params_global)

            optimizer_id.zero_grad()
        else:
            optimizer = torch.optim.Adam(params_func())

        optimizer.zero_grad()


        if not is_joint:
            num_views = len(self.R_base.keys())
            bs = batch['normals'].shape[0] * num_views

            image_lmks68 = landmarks
            if landmarks is not None:
                left_iris = batch['left_iris']
                right_iris = batch['right_iris']
                mask_left_iris = batch['mask_left_iris']
                mask_right_iris = batch['mask_right_iris']
        else:
            image_lmks68 = None
            lmk_mask, normal_mask, normal_map, uv_map, uv_mask = None, None, None, None, None
            left_iris, right_iris, mask_left_iris, mask_right_iris = None, None, None, None

        self.diff_renderer.reset()

        best_loss = np.inf

        n_steps_stagnant = 0
        stagnant_window_size = 10
        past_k_steps = np.array([100.0 for _ in range(stagnant_window_size)])

        iterator = tqdm(range(iters), desc='', leave=True, miniters=100)

        for p in iterator:

            if is_joint and p == int(iters*0.5):

                for pgroup in optimizer.param_groups:
                    if pgroup['name'] in ['t', 'R', 'jaw']:
                        pgroup['lr'] = pgroup['lr'] / 10
                        print(f'LR Reduce at iter {p}, for pgroup {pgroup["name"]}')
                    else:
                        pgroup['lr'] = pgroup['lr'] / 2
            if is_joint and p == int(iters *0.75):
                for pgroup in optimizer.param_groups:
                    if pgroup['name'] in ['t', 'R', 'jaw']:
                        pgroup['lr'] = pgroup['lr'] / 5
                        print(f'LR Reduce at iter {p}, for pgroup {pgroup["name"]}')
                    else:
                        pgroup['lr'] = pgroup['lr'] / 2

            if is_joint and p == int(iters *0.9):
                for pgroup in optimizer.param_groups:
                    if pgroup['name'] in ['t', 'R', 'jaw']:
                        pgroup['lr'] = pgroup['lr'] / 2
                        print(f'LR Reduce at iter {p}, for pgroup {pgroup["name"]}')
                    else:
                        pgroup['lr'] = pgroup['lr'] / 5


            batch_joint, losses, vertices, vertices_noneck, vertices_can, vertices_can_can, proj_vertices, proj_lmks68, selected_frames, variables, num_views, normal_mask, normal_map, uv_map, uv_mask = self.opt_pre(is_joint, iters, p, no_lm, image_lmks68, lmk_mask, normal_mask, normal_map, uv_map, uv_mask, left_iris, right_iris, mask_left_iris, mask_right_iris)

            if is_joint:
                batch = batch_joint

            timestep = 0
            ops = self.diff_renderer(vertices, None, None,
                                     self.r_mvps[timestep], self.R_base[timestep], self.t_base[timestep],
                                     texture_observation_mask=self.texture_observation_mask,
                                     verts_can=vertices_can,
                                     verts_noneck=vertices_noneck,
                                     verts_can_can=vertices_can_can,
                                     verts_depth=proj_vertices[:, :, 2:3],
                                     )


            all_loss = self.opt_post(variables, ops, proj_vertices, proj_lmks68, batch, is_joint, is_first_step, losses, uv_map, selected_frames, p, iters, num_views, normal_mask, normal_map)

            #vertices.retain_grad()
            #if not self.init_done:
            all_loss.backward()#retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            if is_joint:
                optimizer_id.step()
                optimizer_id.zero_grad()


            #if p == 0 or p == iters-1:
            #if p == iters-1:# and not self.config.low_overhead and False:
                #wandb.log(losses)

            self.global_step += 1
            loss_color = all_loss.item()

            if loss_color < best_loss - 1.0:
                best_loss = loss_color
                n_steps_stagnant = 0
            elif p > 25: # only start counting after n steps
                n_steps_stagnant += 1

            if p > 0:
                past_k_steps[p%stagnant_window_size] = np.abs(all_loss.item() - prev_loss)
            prev_loss = all_loss.item()


            if (self.frame % 99 == 0 or p < 10)  and is_joint:
                pass
                #with torch.no_grad():
                #    intrinsics = get_intrinsics(focal_length, principal_point, use_hack=False)

                    #proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics,
                    #                                                 znear=0.1, zfar=5,
                    #                                                width=512,
                    #                                                 height=512)
                    #for serial in self.cam_pose_nvd.keys():
                    #    extr = get_extrinsics(self.R_base[serial], self.t_base[serial])
                    #    r_mvps = torch.matmul(proj_512, extr.repeat(bs, 1, 1))
                    #    self.r_mvps[serial] = r_mvps
                #self.checkpoint(batch, visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.SHAPE_OVERLAY]],
                #                frame_dst='/debug_joint', save=False, dump_directly=True, timestep=p, selected_frames=selected_frames, is_final=True)
            self.frame += 1

            iterator.set_description(f'Timestep {save_timestep}; Loss {all_loss.item():.4f}')

            #if n_steps_stagnant > 35 and not is_joint:
            #    print('Early Stopping, go to next frame!')
            #    #break
            if not is_joint and not is_first_step:
                if p > stagnant_window_size and np.mean(past_k_steps) < self.config.early_stopping_delta: #3.0: #3.0:
                    print('Early Stopping, go to next frame!')
                    #losses['early_stopping'] = past_k_steps
                    #wandb.log(losses)
                    #wandb.log({'early_stopping': wandb.Histogram(past_k_steps)})

                    break
            #print('rate of change', np.mean(past_k_steps))


    def render_and_save(self, batch,
                        visualizations=[[View.GROUND_TRUTH, View.LANDMARKS, View.HEATMAP], [View.COLOR_OVERLAY, View.SHAPE_OVERLAY, View.SHAPE]],
                        frame_dst='/video', save=True, dump_directly=False,
                        outer_iter = None,
                        is_camera : bool = False,
                        all_keyframes : bool = False,
                        timestep : int = 0,
                        is_final : bool = False,
                        selected_frames = None,
                        ):
        batch = self.to_cuda(batch)
        images, landmarks, _ = self.parse_landmarks(batch)

        if 'uv_map' in batch:
            uv_map = batch['uv_map']
            uv_mask = batch['uv_mask']
            uv_map[(1-uv_mask).bool()] = 0
        else:
            uv_map = None
            uv_mask = None

        if 'normals' in batch:
            normal_map = batch['normals']
        else:
            normal_map = None
        if 'normal_map_can' in batch:
            normal_map_can = batch['normal_map_can']
        else:
            normal_map_can = None


        savefolder = self.save_folder + self.actor_name + frame_dst
        num_keyframes = 1#1 + len(self.config.keyframes)

        with torch.no_grad():
            self.diff_renderer.reset()
            num_views = len(self.R_base.keys())
            bs = batch['normals'].shape[0] * num_keyframes #self.shape.shape[0]

            if selected_frames is None:
                exp = self.exp
                eyes = self.eyes
                eyelids = self.eyelids
                R = self.R
                t = self.t
                jaw = self.jaw
                neck = self.neck
                focal_length = self.focal_length
                principal_point = self.principal_point
            else:
                exp = self.exp(selected_frames)
                eyes = self.eyes(selected_frames)
                eyelids = self.eyelids(selected_frames)
                R = self.R(selected_frames)
                t = self.t(selected_frames)
                jaw = self.jaw(selected_frames)
                neck = self.neck(selected_frames)
                if not self.config.global_camera:
                    focal_length = self.focal_length(selected_frames)
                    principal_point = self.principal_point(selected_frames)
                else:
                    focal_length = self.focal_length
                    principal_point = self.principal_point

            with torch.no_grad():
                intrinsics = get_intrinsics(focal_length, principal_point, use_hack=False, size=self.config.size)

                proj_512 = nvdiffrast_util.intrinsics2projection(intrinsics,
                                                                 znear=0.1, zfar=5,
                                                                 width=self.config.size,
                                                                 height=self.config.size)
                for serial in self.cam_pose_nvd.keys():
                    extr = get_extrinsics(self.R_base[serial], self.t_base[serial])
                    r_mvps = torch.matmul(proj_512, extr.repeat(bs, 1, 1))
                    self.r_mvps[serial] = r_mvps
            vertices_can, _lmk68, lmkMP, vertices_can_can, vertices_noneck = self.flame(
                #cameras=torch.inverse(self.R_base[0]),
                cameras=torch.inverse(self.R_base[0]).repeat(bs, 1, 1),
                shape_params=self.shape.repeat(bs, 1),
                expression_params=exp.repeat_interleave(num_views, dim=0), #torch.from_numpy(self.expression_params[:1, :]).cuda().repeat(bs, 1), #self.exp,
                eye_pose_params=eyes.repeat_interleave(num_views, dim=0),
                #euler_angles_to_matrix(x_opts['rotation'][i], 'XYZ')
                jaw_pose_params=jaw.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).cuda(), 'XYZ')).repeat(bs, 1), #self.jaw,
                neck_pose_params=neck.repeat_interleave(num_views, dim=0), #matrix_to_rotation_6d(euler_angles_to_matrix(torch.from_numpy(self.jaw_params[:1, :]).cuda(), 'XYZ')).repeat(bs, 1), #self.jaw,
                eyelid_params=eyelids.repeat_interleave(num_views, dim=0),
                rot_params_lmk_shift=(matrix_to_rotation_6d(torch.inverse(rotation_6d_to_matrix(R)))).repeat_interleave(num_views, dim=0),
            )


            lmk68 = torch.einsum('bny,bxy->bnx', _lmk68, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0))) + t.repeat_interleave(num_views, dim=0).unsqueeze(1)
            vertices = torch.einsum('bny,bxy->bnx', vertices_can, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0))) + t.repeat_interleave(num_views, dim=0).unsqueeze(1)
            vertices_noneck = torch.einsum('bny,bxy->bnx', vertices_noneck, rotation_6d_to_matrix(R.repeat_interleave(num_views, dim=0))) + t.repeat_interleave(num_views, dim=0).unsqueeze(1)


            lmk68 =  project_points_screen_space(lmk68, focal_length, principal_point, self.R_base, self.t_base, size=self.config.size)
            proj_vertices = project_points_screen_space(vertices, focal_length, principal_point, self.R_base, self.t_base, size=self.config.size)


            _timestep = 0
            ops = self.diff_renderer(vertices, None, None,
                                     self.r_mvps[_timestep], self.R_base[_timestep], self.t_base[_timestep],
                                     verts_can=vertices_can,
                                     verts_noneck=vertices_noneck,
                                     verts_depth=proj_vertices[:, :, 2:3],
                                     is_viz=True
                                     )
            mask = (self.parse_mask(ops, batch, visualization=True) > 0).float()
            grabbed_depth = ops['actual_rendered_depth'][0, 0,
            torch.clamp(proj_vertices[0, :, 1].long(), 0, self.config.size-1),
            torch.clamp(proj_vertices[0, :, 0].long(), 0, self.config.size-1),
            ]
            is_visible_verts_idx = grabbed_depth < proj_vertices[0, :, 2] + 1e-2
            if not self.config.occ_filter:
                is_visible_verts_idx = torch.ones_like(is_visible_verts_idx)


            all_final_views = []
            for b_i in range(bs):
                final_views = []

                for views in visualizations:
                    row = []
                    for view in views:
                        if view == View.COLOR_OVERLAY:
                            row.append((ops['normal_images'][b_i].cpu().numpy() + 1)/2)
                        if view == View.GROUND_TRUTH:
                            row.append(images[b_i].cpu().numpy())
                        if (view == View.LANDMARKS and not self.no_lm) or is_camera:
                            gt_lmks = images[b_i:b_i+1].clone()
                            gt_lmks = util.tensor_vis_landmarks(gt_lmks, landmarks[b_i:b_i+1, :, :], color='g')
                            gt_lmks = util.tensor_vis_landmarks(gt_lmks, batch['left_iris'][b_i:b_i+1, ...], color='g')
                            gt_lmks = util.tensor_vis_landmarks(gt_lmks, batch['right_iris'][b_i:b_i+1, ...], color='g')
                            gt_lmks = util.tensor_vis_landmarks(gt_lmks, proj_vertices[b_i:b_i+1, left_iris_flame, ...], color='r')
                            gt_lmks = util.tensor_vis_landmarks(gt_lmks, proj_vertices[b_i:b_i+1, right_iris_flame, ...], color='r')
                            gt_lmks = util.tensor_vis_landmarks(gt_lmks, lmk68[b_i:b_i+1, :, :], color='r')
                            row.append(gt_lmks[0].cpu().numpy())

                    if True:
                        nvd_mask = gaussian_blur(ops['mask_images_rendering'].detach(),
                                                 kernel_size=[self.config.normal_mask_ksize, self.config.normal_mask_ksize],
                                                 sigma=[self.config.normal_mask_ksize, self.config.normal_mask_ksize])
                        nvd_mask = (nvd_mask > 0.5).float()
                        nvd_mask_clone = nvd_mask.clone()


                        eyebrow_level = torch.min(lmk68[:, :, 1], dim=1).indices

                        for _i in range(eyebrow_level.shape[0]):
                            nvd_mask_clone[_i, :, :eyebrow_level[_i], :] = 0


                    final_views.append(row)


                # VIDEO
                final_views = util.merge_views(final_views)
                all_final_views.append(final_views)
            final_views = np.concatenate(all_final_views, axis=0)

            if outer_iter is None:
                frame_id = str(self.frame).zfill(5)
            else:
                frame_id = str(self.frame + 10*outer_iter).zfill(5)

            if uv_map is not None and is_final:
                # uv losses visualizations
                proj_vertices = proj_vertices[:, self.uv_loss_fn.valid_vertex_index, :]
                can_uv = torch.from_numpy(np.load(env_paths.FLAME_UV_COORDS)).cuda().unsqueeze(0).float()[:, self.uv_loss_fn.valid_vertex_index, :]
                valid_verts_visibility = is_visible_verts_idx[self.uv_loss_fn.valid_vertex_index]
                #can_uv[..., 0] = (can_uv[..., 0] * -1) + 1
                can_uv[..., 1] = (can_uv[..., 1] * -1) + 1
                #can_uv = can_uv[:, ::50, :]
                gt_uv = uv_map[:, :2, :, :].permute(0, 2, 3, 1)
                gt_uv = gt_uv.reshape(gt_uv.shape[0], -1, 2)  # B x n_pixel x 2
                can_uv = can_uv.repeat(gt_uv.shape[0], 1, 1)
                knn_result = knn_points(can_uv, gt_uv)

                pixel_position_width = knn_result.idx % uv_map.shape[-1]
                pixel_position_height = knn_result.idx // uv_map.shape[-2]

                dists = knn_result.dists.clone()

                gt_2_verts = torch.cat([pixel_position_width, pixel_position_height], dim=-1)

                pred_normals = ops['normal_images']  # 1 3 512 512 normals in world space
                rot_mat = rotation_6d_to_matrix(R.detach().repeat_interleave(num_views, dim=0))  # 1 3 3
                pred_normals_flame_space = torch.einsum('bxy,bxhw->byhw', rot_mat, pred_normals)

                delta = self.config.uv_loss.delta_uv
                catted_uv_rows = []
                for b_i in range(images.shape[0]):
                    empty = images[b_i].detach().cpu().numpy().copy().transpose(1, 2, 0)
                    is_valid_uv_corresp = (dists[b_i, :, 0] < delta) & valid_verts_visibility
                    valid_pred_2d = proj_vertices[b_i, is_valid_uv_corresp, :]
                    valid_gt_2d = gt_2_verts[b_i, is_valid_uv_corresp, :]
                    pixels_pred = torch.stack(
                        [
                            torch.clamp(valid_pred_2d[:, 0], 0, images.shape[-1] - 1),
                            torch.clamp(valid_pred_2d[:, 1], 0, images.shape[-2] - 1),
                        ], dim=-1
                    ).int()
                    pixels_gt = torch.stack(
                        [
                            torch.clamp(valid_gt_2d[:, 0], 0, images.shape[-1] - 1),
                            torch.clamp(valid_gt_2d[:, 1], 0, images.shape[-2] - 1),
                        ], dim=-1
                    ).int()

                    if self.config.draw_uv_corresp:
                        empty = plot_points(empty, pts=pixels_pred.detach().cpu().numpy(), pts2=pixels_gt.detach().cpu().numpy())


                    gt_uv = uv_map[:, :2, :, :].permute(0, 2, 3, 1)

                    upper_forehead = ((uv_map[:, 0, :, :].abs() < 0.85) &
                                      (uv_map[:, 0, :, :].abs() > (1 - 0.85)) &
                                      (uv_map[:, 1, :, :] < 0.35) &
                                      (uv_map[:, 1, :, :] > 0.)).float()
                    upper_forehead = (gaussian_blur(upper_forehead, [self.config.normal_mask_ksize, self.config.normal_mask_ksize], sigma=[self.config.normal_mask_ksize, self.config.normal_mask_ksize]) > 0).float()
                    losses_sil = ((1 - upper_forehead[:, None, :, :]) * (batch['fg_mask'] - ops['fg_images'])).abs().permute(0, 2, 3, 1)


                    uv_loss = ((gt_uv - ops['uv_images']) * ops['mask_images'][:, 0, ...].unsqueeze(-1)).abs()
                    #catted_uv = torch.cat([gt_uv[b_i], ops['uv_images'][b_i], uv_loss[b_i]], dim=1).detach().cpu().numpy()
                    catted_uv = torch.cat([losses_sil[b_i][..., :2], uv_loss[b_i]], dim=1).detach().cpu().numpy()
                    catted_uv_I = np.zeros([catted_uv.shape[0], catted_uv.shape[1], 3])
                    catted_uv_I[:, :, :2] = catted_uv
                    catted_uv_I = (catted_uv_I * 255).astype(np.uint8)
                    shape_mask = ((ops['alpha_images'] * ops['mask_images_mesh']) > 0.).int()[b_i]
                    shape = (pred_normals_flame_space[b_i]+1)/2 * shape_mask
                    blend = images[b_i] * (1 - shape_mask) + images[b_i] * shape_mask * 0.3 + shape * 0.7 * shape_mask
                    to_be_catted = [(images[b_i].cpu().permute(1, 2, 0).numpy()*255).astype(np.uint8),
                                                  (blend.permute(1, 2, 0).detach().cpu().numpy()*255).astype(np.uint8),
                                    ]
                    if self.config.draw_uv_corresp:
                        to_be_catted.append(catted_uv_I)
                        to_be_catted.append(empty)
                    catted_uv_I = np.concatenate(to_be_catted, axis=1)
                    catted_uv_rows.append(catted_uv_I)

                if normal_map is None:
                    catted_uv_I = Image.fromarray(np.concatenate(catted_uv_rows, axis=0))

                #pl = pv.Plotter()
                #pl.add_mesh(trim)
                #pl.add_points(visible_verts)
                #pl.show()

            else:
                catted_uv_I = None
                catted_uv_rows = []

            if normal_map is not None:
                dilated_eye_mask = 1 - (gaussian_blur(ops['mask_images_eyes'], [self.config.normal_mask_ksize, self.config.normal_mask_ksize], sigma=[1, 1]) > 0).float()
                l_map = (normal_map - pred_normals_flame_space)
                valid = ((l_map.abs().sum(dim=1)/3) < self.config.delta_n).unsqueeze(1)




                predicted_normal = ((pred_normals_flame_space.permute(0, 2, 3, 1)[...,
                                     :3] + 1) / 2 * 255).detach().cpu().numpy().astype(np.uint8)
                if self.config.draw_uv_corresp:
                    normal_loss_map = l_map * valid.float() * batch["normal_mask"] * dilated_eye_mask
                    pseudo_normal = ((normal_map.permute(0, 2, 3, 1) + 1) / 2 * 255).detach().cpu().numpy().astype(
                        np.uint8)
                    normal_loss_map = (
                        (normal_loss_map.abs().permute(0, 2, 3, 1)) / 2 * 255).detach().cpu().numpy().astype(
                        np.uint8)
                    catted = np.concatenate([pseudo_normal, predicted_normal, normal_loss_map], axis=2)
                else:
                    catted = predicted_normal
                # Image.fromarray(catted).show()
                # print('hi')

                for b_i in range(catted.shape[0]):
                    if len(catted_uv_rows) > 0:
                        catted_uv_rows[b_i] = np.concatenate([catted_uv_rows[b_i], catted[b_i]], axis=1)
                    else:
                        catted_uv_rows.append(catted[b_i])


                catted_uv_I = Image.fromarray(np.concatenate(catted_uv_rows, axis=0))

            #if catted_uv_I is not None:
            #    save_fodler_uv = f'{savefolder}'
            #    os.makedirs(save_fodler_uv, exist_ok=True)
            #    if is_final:
            #        catted_uv_I.save(f'{save_fodler_uv}/{timestep}.png')
            #    else:
            #        catted_uv_I.save(f'{save_fodler_uv}/{self.frame}.png')


            if not save:
                return

            # CHECKPOINT
            self.save_checkpoint(timestep, selected_frames=selected_frames)
        return catted_uv_I



    def parse_landmarks(self, batch):
        images = batch['rgb']
        if 'lmk' in batch:
            landmarks = batch['lmk']
            lmk68 = landmarks[:, WFLW_2_iBUG68, :]
            lmk_mask = ~(lmk68.sum(2, keepdim=True) == 0)
            batch['left_iris'] = landmarks[:, 96:97, :]
            batch['right_iris'] = landmarks[:, 97:98, :]
            batch['mask_left_iris'] = ~(landmarks.sum(2, keepdim=True) == 0)[:, 96:97, :]
            batch['mask_right_iris'] = ~(landmarks.sum(2, keepdim=True) == 0)[:, 97:98, :]

            landmarks = lmk68
        else:
            landmarks = lmk_mask = None

        return images,  landmarks, lmk_mask,


    def read_data(self, timestep):
        DATA_FOLDER = f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}'
        P3DMM_FOLDER = f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/p3dmm/'

        try:
            rgb = np.array(Image.open(f'{DATA_FOLDER}/cropped/{timestep:05d}.jpg').resize((self.config.size, self.config.size))) / 255
        except Exception as ex:
            rgb = np.array(Image.open(f'{DATA_FOLDER}/cropped/{timestep:05d}.png').resize((self.config.size, self.config.size))) / 255

        mica_folder = f'{DATA_FOLDER}/mica'
        mica_files = os.listdir(mica_folder)
        mica_shapes = []
        for mica_file in mica_files:
            mica_shape = np.load(f'{mica_folder}/{mica_file}/identity.npy')
            mica_shapes.append(np.squeeze(mica_shape))
        mica_shapes = np.stack(mica_shapes, axis=0)
        if self.config.early_exit:
            mica_shape = mica_shapes[0, :]
        else:
            mica_shape = np.mean(mica_shapes, axis=0)

        seg = np.array(Image.open(f'{DATA_FOLDER}/seg_og/{timestep:05d}.png').resize((self.config.size, self.config.size), Image.NEAREST))
        if len(seg.shape) == 3:
            seg = seg[..., 0]
        uv_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                   (seg == 10) | (seg == 12) | (seg == 13) |
                   (seg==1) | # neck
                   (seg == 4) | (seg==5) # ears
                   )

        normal_mask = ((seg == 2) | (seg == 6) | (seg == 7) |
                   (seg == 10) | (seg == 12) | (seg == 13)
                   ) | (seg == 11)  # mouth interior
        if self.config.big_normal_mask:
            normal_mask = normal_mask | (seg==1) | (seg == 4) | (seg==5) # add neck and ears

        fg_mask = ((seg == 2) | (seg == 6) | (seg == 7) | (seg == 8) | (seg == 9) | #(seg == 4) | (seg == 5) |
                   (seg == 10) | (seg == 12) | (seg == 13)
                   )

        valid_bg = seg <= 1

        try:
            normals = ((np.array(Image.open(f'{P3DMM_FOLDER}/normals/{timestep:05d}.png').resize((self.config.size, self.config.size))) / 255).astype(np.float32) - 0.5 )*2
            uv_map = (np.array(Image.open(f'{P3DMM_FOLDER}/uv_map/{timestep:05d}png').resize((self.config.size, self.config.size))) / 255).astype(np.float32)
        except Exception as ex:
            normals = ((np.array(Image.open(f'{P3DMM_FOLDER}/normals/{timestep:05d}.png').resize((self.config.size, self.config.size))) / 255).astype(
                np.float32) - 0.5) * 2
            uv_map = (np.array(Image.open(f'{P3DMM_FOLDER}/uv_map/{timestep:05d}.png').resize((self.config.size, self.config.size))) / 255).astype(np.float32)

        try:
            lms = np.load(f'{DATA_FOLDER}/PIPnet_landmarks/{timestep:05d}.npy') * self.config.size
        except Exception as ex:
            lms = np.zeros([98, 2])

        ret_dict = {
            'rgb': rgb,
            'mica_shape': mica_shape,
            'normals': normals,
            'uv_map': uv_map,
            'uv_mask': uv_mask,
            'normal_mask': normal_mask,
            'fg_mask': fg_mask,
            'valid_bg': valid_bg,
        }
        if lms is not None:
            ret_dict['lmk'] = lms


        ret_dict = {k: torch.from_numpy(v).float().unsqueeze(0).cuda() for k,v in ret_dict.items()}

        ret_dict['uv_mask'] = ret_dict['uv_mask'][:, :, :, None].repeat(1, 1, 1, 3)
        ret_dict['normal_mask'] = ret_dict['normal_mask'][:, :, :, None].repeat(1, 1, 1, 3)
        ret_dict['fg_mask'] = ret_dict['fg_mask'][:, :, :, None].repeat(1, 1, 1, 3)

        channels_first  =['rgb', 'uv_mask', 'normal_mask', 'normals', 'uv_map', 'fg_mask']
        for k in channels_first:
            ret_dict[k] = ret_dict[k].permute(0, 3, 1, 2)

        return ret_dict


    def prepare_global_optimization(self, N_FRAMES):
        is_sparse=True
        self.exp = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=100, sparse=is_sparse, ).cuda()
        self.R = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=6, sparse=is_sparse).cuda()
        self.t = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=3, sparse=is_sparse).cuda()
        self.eyes = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=12, sparse=is_sparse).cuda()
        self.eyelids = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=12, sparse=is_sparse).cuda()
        self.jaw = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=6, sparse=is_sparse).cuda()
        self.neck = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=6, sparse=is_sparse).cuda()
        if not self.config.global_camera:
            self.focal_length = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=1, sparse=is_sparse).cuda()
            self.principal_point = nn.Embedding(num_embeddings=N_FRAMES, embedding_dim=2, sparse=is_sparse).cuda()

        exp = torch.cat(self.intermediate_exprs, dim=0)
        R = torch.cat(self.intermediate_Rs, dim=0)
        t = torch.cat(self.intermediate_ts, dim=0)
        eyes = torch.cat(self.intermediate_eyes, dim=0)
        eyelids = torch.cat(self.intermediate_eyelids, dim=0)
        jaw = torch.cat(self.intermediate_jaws, dim=0)
        neck = torch.cat(self.intermediate_necks, dim=0)
        if not self.config.global_camera:
            focal_length = torch.cat(self.intermediate_fls, dim=0)
            principal_point = torch.cat(self.intermediate_pps, dim=0)

        with torch.no_grad():
            self.exp.weight = torch.nn.Parameter(exp)
            self.R.weight = torch.nn.Parameter(R)
            self.t.weight = torch.nn.Parameter(t)
            self.eyes.weight = torch.nn.Parameter(eyes)
            self.eyelids.weight = torch.nn.Parameter(eyelids)
            self.jaw.weight = torch.nn.Parameter(jaw)
            self.neck.weight = torch.nn.Parameter(neck)
            if not self.config.global_camera:
                self.focal_length.weight = torch.nn.Parameter(focal_length)
                self.principal_point.weight = torch.nn.Parameter(principal_point)


    def run(self):
        timestep = self.config.start_frame
        batch = self.read_data(timestep=timestep)

        # Important to initialize
        self.create_parameters(0, batch['mica_shape'])
        self.frame = 0

        print('''
        <<<<<<<< STARTING ONLINE TRACKING PHASE >>>>>>>>
        ''')

        for timestep in range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP):
            batch = self.read_data(timestep=timestep)
            for k in batch.keys():
                if k not in self.cached_data:
                    self.cached_data[k] = [batch[k]]
                else:
                    self.cached_data[k].append(batch[k])
            if timestep == self.config.start_frame:
                self.optimize_camera(batch, steps=500, is_first_frame=True)
                #params = lambda: self.clone_params_keyframes_all(freeze_id=self.config.is_discontinuous, freeze_cam=self.config.global_camera, include_neck=self.config.include_neck)
                params = lambda: self.clone_params_keyframes_all(freeze_id=not (self.MAX_STEPS==1), freeze_cam=self.config.global_camera, include_neck=self.config.include_neck)
                is_first_step = True
            else:
                if self.config.extra_cam_steps:
                    self.optimize_camera(batch, steps=10, is_first_frame=False)
                params = lambda: self.clone_params_keyframes_all(freeze_id=True, freeze_cam=self.config.global_camera, include_neck=self.config.include_neck)
                is_first_step = False


            self.optimize_color(batch, params,
                                no_lm=self.no_lm,
                                save_timestep=timestep,
                                is_first_step=is_first_step
            )

            self.uv_loss_fn.is_next()
            #self.checkpoint(batch,  visualizations=[[View.GROUND_TRUTH, View.COLOR_OVERLAY, View.LANDMARKS, View.SHAPE]], frame_dst='/initialization', outer_iter=0, timestep=timestep, is_final=True, save=True)
            self.frame += 1

            # save results for global optimization later
            self.intermediate_exprs.append(self.exp.detach().clone())
            self.intermediate_Rs.append(self.R.detach().clone())
            self.intermediate_ts.append(self.t.detach().clone())
            self.intermediate_eyes.append(self.eyes.detach().clone())
            self.intermediate_eyelids.append(self.eyelids.detach().clone())
            self.intermediate_jaws.append(self.jaw.detach().clone())
            self.intermediate_necks.append(self.neck.detach().clone())
            if not self.config.global_camera:
                self.intermediate_fls.append(self.focal_length.detach().clone())
                self.intermediate_pps.append(self.principal_point.detach().clone())

            # for temporally discontinuous inputs, donÄt carry over state of the head (except for shape aprameters)
            if self.config.is_discontinuous:
                n_timesteps = 1
                expression_params = np.zeros([n_timesteps, 100])
                jaw_params = np.zeros([n_timesteps, 3])
                neck_params = np.zeros([n_timesteps, 3])
                flame_R = torch.from_numpy(np.stack([np.eye(3) for _ in range(n_timesteps)], axis=0))
                flame_t = torch.from_numpy(np.stack([np.zeros([3]) for _ in range(n_timesteps)], axis=0))
                self.R = nn.Parameter(matrix_to_rotation_6d(flame_R.float().to(self.device)))
                self.t = nn.Parameter(flame_t.float().to(self.device))

                self.expression_params = expression_params
                self.jaw_params = jaw_params.astype(np.float32)
                self.neck_params = neck_params.astype(np.float32)

                self.exp = nn.Parameter(
                    torch.from_numpy(self.expression_params[[0] + self.config.keyframes, ..., :]).float().to(
                        self.device))
                self.jaw = nn.Parameter(matrix_to_rotation_6d(euler_angles_to_matrix(
                    torch.from_numpy(self.jaw_params[[0] + self.config.keyframes, ..., :]).cuda(), 'XYZ')))
                self.neck = nn.Parameter(matrix_to_rotation_6d(euler_angles_to_matrix(
                    torch.from_numpy(self.neck_params[[0] + self.config.keyframes, ..., :]).cuda(), 'XYZ')))

                self.eyes = nn.Parameter(torch.cat([matrix_to_rotation_6d(I), matrix_to_rotation_6d(I)], dim=1).repeat(
                    1 + len(self.config.keyframes), 1))
                self.eyelids = nn.Parameter(torch.zeros(1 + len(self.config.keyframes), 2).float().to(self.device))



            if self.config.early_exit:
                exit()

        for k in self.cached_data.keys():
            self.cached_data[k] = torch.cat(self.cached_data[k], dim=0)

        params = lambda: self.clone_params_keyframes_all_joint(freeze_id=False, is_joint=True, include_neck=self.config.include_neck)


        if self.config.uv_map_super > 0.0:
            self.uv_loss_fn.finish_stage1()

        self.config.iters = self.config.global_iters #self.config.iters * 10

        N_FRAMES = len(self.intermediate_exprs)
        #build optimization targets for global optimization, implement as sparse torch.Embedding
        self.prepare_global_optimization(N_FRAMES=N_FRAMES)


        if COMPILE:
            self.flame = torch.compile(self.flame)
            self.opt_pre = torch.compile(self.opt_pre)
            self.opt_post = torch.compile(self.opt_post)

        print('''
                <<<<<<<< STARTING GLOBAL TRACKING PHASE >>>>>>>>
                ''')

        if N_FRAMES > 1:

            self.optimize_color(None, params,
                                no_lm=self.no_lm,
                                save_timestep=1000, #timestep,
                                is_joint=True,
                                )


        # render result and save it as a video to get some viusal feedback
        video_frames = []
        for it, timestep in enumerate(range(self.config.start_frame, self.MAX_STEPS + self.config.start_frame, self.FRAME_SKIP)):
            selected_frames = []
            selected_frames_loading = []
            batches = []
            batch = self.read_data(timestep=timestep)
            batches.append(batch)
            selected_frames.append(it)
            selected_frames_loading.append(timestep)
            batches = {k: torch.cat([x[k] for x in batches], dim=0) for k in batch.keys()}
            selected_frames = torch.from_numpy(np.array(selected_frames)).long().cuda()

            result_rendering = self.render_and_save(batches, visualizations=[[View.GROUND_TRUTH, View.COLOR_OVERLAY, View.LANDMARKS, View.SHAPE]],
                                                    frame_dst='/joint_initialization', outer_iter=0, timestep=timestep, is_final=True, selected_frames=selected_frames)
            video_frames.append(np.array(result_rendering))
            self.frame += 1

        mediapy.write_video(f'{self.save_folder}/{self.actor_name}/result.mp4', images=video_frames, crf=15, fps=25)


        # Optionally delete all preoprocessing artifacts, once tracking is done (only keep cropped images)
        if self.config.delete_preprocessing:
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/mica')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/p3dmm')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/p3dmm_wGT')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/p3dmm_extraViz')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/pipnet')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/PIPnet_annotated_images')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/PIPnet_landmarks')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/rgb')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/seg_non_crop_annotations')
            shutil.rmtree(f'{env_paths.PREPROCESSED_DATA}/{self.config.video_name}/seg_og')


        print(f'''
                <<<<<<<< DONE WITH TRACKING {self.actor_name} >>>>>>>>
                ''')





