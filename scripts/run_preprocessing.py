import os
import tyro

from pixel3dmm import env_paths


def main(video_or_images_path : str):

    if os.path.isdir(video_or_images_path):
        vid_name = video_or_images_path.split('/')[-1]
    else:
        vid_name = video_or_images_path.split('/')[-1][:-4]

    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_cropping.py --video_or_images_path {video_or_images_path}')

    os.system(f'cd {env_paths.CODE_BASE}/src/pixel3dmm/preprocessing/MICA ; python demo.py -video_name {vid_name} -a {env_paths.PREPROCESSED_DATA}/{vid_name}/arcface/')

    os.system(f'cd {env_paths.CODE_BASE}/scripts/ ; python run_facer_segmentation.py --video_name {vid_name}')



if __name__ == '__main__':
    tyro.cli(main)