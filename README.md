#  Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction
[**Paper**](https://arxiv.org/abs/2505.00615) | [**Video**](https://www.youtube.com/watch?v=BwxwEXJwUDc) | [**Project Page**](https://simongiebenhain.github.io/pixel3dmm/) <br>

<div style="text-align: center">
<img src="media/banner.gif" />
</div>

This repository contains the official implementation of the paper:

### Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction
[Simon Giebenhain](https://simongiebenhain.github.io/), 
[Tobias Kirschstein](https://niessnerlab.org/members/tobias_kirschstein/profile.html), 
[Martin Rünz](https://www.martinruenz.de/), 
[Lourdes Agaptio](https://scholar.google.com/citations?user=IRMX4-4AAAAJ&hl=en) and 
[Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html)  



## 1. Installation 

First we need to set up a conda environment. Below, there are two installation options presented:

### Option A: Using `environment.yml`
> Note that this can take quite a while.
```
conda env create --file environment.yml
conda activate p3dmm 
```

### Option B: Manual Installation

```
conda create -n p3dmm python=3.9
conda activate p3dmm

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
conda install nvidia/label/cuda-11.8.0::cuda-nvcc nvidia/label/cuda-11.8.0::cuda-cccl nvidia/label/cuda-11.8.0::cuda-cudart nvidia/label/cuda-11.8.0::cuda-cudart-dev nvidia/label/cuda-11.8.0::libcusparse nvidia/label/cuda-11.8.0::libcusparse-dev nvidia/label/cuda-11.8.0::libcublas nvidia/label/cuda-11.8.0::libcublas-dev nvidia/label/cuda-11.8.0::libcurand nvidia/label/cuda-11.8.0::libcurand-dev nvidia/label/cuda-11.8.0::libcusolver nvidia/label/cuda-11.8.0::libcusolver-dev
```

```
conda env config vars set TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6+PTX"
conda deactivate 
conda activate p3dmm
```


```
pip install git+https://github.com/facebookresearch/pytorch3d.git@stable
pip install git+https://github.com/NVlabs/nvdiffrast.git

pip install -r requirements.txt
```

### 1.1. Final Installation Steps

Finally, you will need to run
```
pip instal -e .
```
to install `pixel3dmm` as a package.

As we use the FLAME 3DMM for tracking, you will need an account for the [FLAME website](https://flame.is.tue.mpg.de/).

Next, you can install all necessary preprocessing repositories and download network weights, by running
```
./install_preprocessing_pipeline.sh
```

### 1.2 Environment Paths

All paths to data / models / infernce are defined by environment variables.
For this we recommend to create a file in your home directory in `~/.config/pixel3dmm/.env` with the following content:
```
PIXEL3DMM_CODE_BASE="{/PATH/TO/THIS/REPOSITORY/}"
PIXEL3DMM_PREPROCESSED_DATA="{/WHERE/TO/STORE/PREPROCESSING/RESULTS/}"
PIXEL3DMM_TRACKING_OUTPUT="{/WHERE/TO/STORE/TRACKING/RESULTS/}"
```
Replace the `{...}` with the locations where data / models / experiments should be located on your machine.

If you do not like creating a config file in your home directory, you can instead hard-code the paths in the env.py. 
Note that using the `.config` folder can be great advantage when working with different machines, e.g. a local PC and a GPU cluster which can have their separate `.env` files.

## 2. Face Tracking

### 2.1 Preprocessing
Before running the tracking you will need to execute

```
PATH_TO_VIDEO="/path/to/video.mp4"
base_name=$(basename $PATH_TO_VIDEO)
VID_NAME="${base_name%%.*}"

python scripts/run_preprocessing.py --video_or_images_path $PATH_TO_VIDEO
````
which will perform cropping, facial landmark detection and segmentation and execute MICA.
Here, `PATH_TO_VIDEO` can either point to an `.mp4` file, or to a folder with images.

> *Note*: You can also run single-image FLAME fitting when providing `.jpg` or `.png` files instead of `.mp4`

### 2.2 Pixel3DMM Inference
Next run normal and uv-map prediction
````
python scripts/network_inference.py model.prediction_type=normals video_name=$VID_NAME
python scripts/network_inference.py model.prediction_type=uv_map video_name=$VID_NAME
````

> *Note*: You can have a look at the preprocessed result and pixel3dmm predictions in `PIXEL3DMM_PREPROCESSED_DATA/{VID_NAME}`. 


> *Note*: This script assumes square images and resizes to 512x512 before the network inference.

### 2.3 Tracking
Finally, run the tracking as such
```
python scripts/track.py video_name=$VID_NAME
```

> *Note*: When `COMPILE=True` (as per default) e.g. the PyCharm debugger won't work in compiled code segments.

> *Note*: You can overwrite the default tracking parameters in `configs/tracking.yaml` using command line arguments. 


> *Note*: It is possible to trade-off fitting fidelity against speed: 
> - increasing `early_stopping_delta` will speed up the online tracking phase, as it controls at what rate of loss-change to skip to the next frame.
> - `global_iters` controls the number of iteration of the global optimization stage.
### 2.4 Visualizations

For convenience we provide a script that shows how to correctly interpret the estimated camera parameters in relation to the FLAME mesh.
Running
````
python scripts/viz_head_centric_cameras.py 
````

### 2.5 Example Inference
You can run our tracker on example videos, by following the steps described in [2. Face Tracking](#2-face-tracking), and setting `PATH_TO_VIDEO="/path/to/this/repo/example_videos/ex1.mp4`.



## Citation

If you find our code or paper useful, please consider citing
```bibtex
@misc{giebenhain2025pixel3dmm,
title={Pixel3DMM: Versatile Screen-Space Priors for Single-Image 3D Face Reconstruction},
author={Simon Giebenhain and Tobias Kirschstein and  Martin R{\"{u}}nz and Lourdes Agapito and Matthias Nie{\ss}ner},
year={2025},
url={https://arxiv.org/abs/2505.00615},
}
```


Contact [Simon Giebenhain](mailto:simon.giebenhain@tum.de) for questions, comments and reporting bugs, or open a GitHub Issue.
