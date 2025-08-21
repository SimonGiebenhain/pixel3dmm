import os
import tyro
import mediapy
import torch
import numpy as np
import pyvista as pv
import trimesh
from PIL import Image

from dreifus.matrix import Intrinsics, Pose, CameraCoordinateConvention, PoseType
from dreifus.pyvista import add_camera_frustum, render_from_camera

from pixel3dmm.utils.utils_3d import rotation_6d_to_matrix
from pixel3dmm.env_paths import PREPROCESSED_DATA, TRACKING_OUTPUT


def main(vid_name : str,
         postfix : str = 'nV1_noPho_uv2000.0_n1000.0',
         HEAD_CENTRIC : bool = True,
         DO_PROJECTION_TEST : bool = False,
         ):
    tracking_dir = f'{TRACKING_OUTPUT}/{vid_name}_{postfix}'

    meshes = [f for f in os.listdir(f'{tracking_dir}/mesh/') if f.endswith('.ply') and not 'canonical' in f]
    meshes.sort()

    ckpts = [f for f in os.listdir(f'{tracking_dir}/checkpoint/') if f.endswith('.frame')]
    ckpts.sort()

    N_STEPS = len(meshes)

    pl = pv.Plotter()
    vid_frames = []
    for i in range(N_STEPS):
        ckpt = torch.load(f'{tracking_dir}/checkpoint/{ckpts[i]}', weights_only=False)

        mesh = trimesh.load(f'{tracking_dir}/mesh/{meshes[i]}', process=False)

        head_rot = rotation_6d_to_matrix(torch.from_numpy(ckpt['flame']['R'])).numpy()[0]

        if not HEAD_CENTRIC:
            # move mesh from FLAME Space into World Space
            mesh.vertices = mesh.vertices @ head_rot.T + (ckpt['flame']['t'])
        else:
            # undo neck rotation
            verts_hom = np.concatenate([mesh.vertices, np.ones_like(mesh.vertices[..., :1])], axis=-1)
            verts_hom = verts_hom @ np.linalg.inv(ckpt['joint_transforms'][0, 1, :, :]).T
            mesh.vertices = verts_hom[..., :3]



        extr_open_gl_world_to_cam = np.eye(4)
        extr_open_gl_world_to_cam[:3, :3] = ckpt['camera']['R_base_0'][0]
        extr_open_gl_world_to_cam[:3, 3] = ckpt['camera']['t_base_0'][0]
        if HEAD_CENTRIC:
            flame2world = np.eye(4)
            flame2world[:3, :3] = head_rot
            flame2world[:3, 3] = np.squeeze(ckpt['flame']['t'])
            extr_open_gl_world_to_cam = extr_open_gl_world_to_cam @ flame2world @ ckpt['joint_transforms'][0, 1, :, :]




        extr_open_gl_world_to_cam = Pose(extr_open_gl_world_to_cam,
                                         camera_coordinate_convention=CameraCoordinateConvention.OPEN_GL,
                                         pose_type=PoseType.WORLD_2_CAM)

        intr = np.eye(3)
        intr[0, 0] = ckpt['camera']['fl'][0, 0] * 256
        intr[1, 1] = ckpt['camera']['fl'][0, 0] * 256
        intr[:2, 2] = ckpt['camera']['pp'][0] * (256/2+0.5) + 256/2 + 0.5

        intr = Intrinsics(intr)



        pl.add_mesh(mesh, color=[(i/N_STEPS), 0, ((N_STEPS-i)/N_STEPS)])
        add_camera_frustum(pl, extr_open_gl_world_to_cam, intr, color=[(i/N_STEPS), 0, ((N_STEPS-i)/N_STEPS)])

        if DO_PROJECTION_TEST:
            pll = pv.Plotter(off_screen=True, window_size=(256, 256))
            pll.add_mesh(mesh)
            img = render_from_camera(pll, extr_open_gl_world_to_cam, intr)

            gt_img = np.array(Image.open(f'{PREPROCESSED_DATA}/{vid_name}/cropped/{i:05d}.jpg').resize((256, 256)))

            alpha = img[..., 3]

            overlay = (gt_img *0.5 + img[..., :3]*0.5).astype(np.uint8)
            vid_frames.append(overlay)




    pl.show()

    if DO_PROJECTION_TEST:
        mediapy.write_video(f'{tracking_dir}/projection_test.mp4', images=vid_frames)



if __name__ == '__main__':
    tyro.cli(main)