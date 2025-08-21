#!/bin/bash

cd src/pixel3dmm/preprocessing/

# facer repository
git clone git@github.com:FacePerceiver/facer.git
cd facer
cp ../replacement_code/farl.py facer/face_parsing/farl.py
cp ../replacement_code/facer_transform.py facer/transform.py
pip install -e .
cd ..

# MICA
git clone git@github.com:Zielon/MICA.git
cd MICA
cp ../replacement_code/install_mica_download_flame.sh install.sh
cp ../replacement_code/mica_demo.py demo.py
cp ../replacement_code/mica.py micalib/models/mica.py
./install.sh
cd ..

#TODO: Maybe need to copy these flame weights to trackign/FLAME as well, or ideally adjust some paths instead


# PIPnet
git clone https://github.com/jhb86253817/PIPNet.git
cd PIPNet
cd FaceBoxesV2/utils
sh make.sh
cd ../..
mkdir snapshots
mkdir snapshots/WFLW/
mkdir snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/
gdown --id 1nVkaSbxy3NeqblwMTGvLg4nF49cI_99C -O snapshots/WFLW/pip_32_16_60_r18_l2_l1_10_1_nb10/epoch59.pth
#mkdir snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10/
#gdown --id 1Jb97z5Z5ca61-6W2RDOK0e2w_RlbeWgS -O snapshots/WFLW/pip_32_16_60_r101_l2_l1_10_1_nb10/epoch59.pth


cd ../../../../
mkdir pretrained_weights
cd pretrained_weights
gdown --id 1SDV_8_qWTe__rX_8e4Fi-BE3aES0YzJY -O ./uv.ckpt
gdown --id 1KYYlpN-KGrYMVcAOT22NkVQC0UAfycMD -O ./normals.ckpt
