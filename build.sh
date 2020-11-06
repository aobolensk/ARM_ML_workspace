#!/bin/bash

WORKSPACE_DIR=$PWD

# Update submodules
git submodule update --init --recursive

# Build ComputeLibrary
cd ComputeLibrary
scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" internal_only=0 arch=arm64-v8a -j`nproc --all`
cd $WORKSPACE_DIR

echo "Downloading images..."

# Download images
if [ ! -d ./imagenet ]; then
    mkdir imagenet && cd imagenet
    wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train_t3.tar
    tar xf ILSVRC2012_img_train_t3.tar
    rm ILSVRC2012_img_train_t3.tar
    tar xf *.tar
    find -name '*.tar' -exec tar xf {} \;
    rm *.tar
fi
cd $WORKSPACE_DIR

echo "Downloading assets..."

# Download assets
if [ ! -d ./alexnet_assets ]; then
    mkdir alexnet_assets && cd alexnet_assets
    wget https://developer.arm.com//-/media/Arm%20Developer%20Community/Images/Tutorial%20Guide%20Diagrams%20and%20Screenshots/Machine%20Learning/Running%20AlexNet%20on%20Pi%20with%20Compute%20Library/compute_library_alexnet.zip?revision=c1a232fa-f328-451f-9bd6-250b83511e01 -O compute_library_alexnet.zip
    unzip compute_library_alexnet.zip
    rm compute_library_alexnet.zip
fi
cd $WORKSPACE_DIR

echo "Installing Python requirements.txt"

# Install Python requirements
python3 -m pip install -r requirements.txt
