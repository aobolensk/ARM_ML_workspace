#!/usr/bin/env python3

import multiprocessing
import os
import sys

print("Installing requirements...")
os.system(f"{sys.executable} -m pip install -r requirements.txt")
WORKSPACE_DIR = os.getcwd()


def build_compute_library():
    print("Building ComputeLibrary")
    os.chdir("ComputeLibrary")
    os.system('scons Werror=1 debug=0 asserts=0 neon=1 opencl=1 embed_kernels=1 extra_cxx_flags="-fPIC" internal_only=0 arch=arm64-v8a -j`nproc --all`')
    os.chdir(WORKSPACE_DIR)


def download_images():
    if os.path.isdir("imagenet"):
        return
    print("Downloading images...", flush=True)
    os.mkdir("imagenet")
    os.chdir("imagenet")
    os.system(
        "wget http://www.image-net.org/challenges/LSVRC/2012/dd31405981ef5f776aa17412e1f0c112/ILSVRC2012_img_train_t3.tar &> /dev/null")
    print("Unpacking images...")
    os.system("tar xf ILSVRC2012_img_train_t3.tar")
    os.system("rm ILSVRC2012_img_train_t3.tar")
    os.system("tar xf *.tar")
    os.system("find -name '*.tar' -exec tar xf {} \;")
    os.system("rm *.tar")
    os.chdir(WORKSPACE_DIR)


def download_assets():
    if os.path.isdir("alexnet_assets"):
        return
    print("Downloading assets...")
    os.mkdir("alexnet_assets")
    os.chdir("alexnet_assets")
    os.system(
        "wget https://developer.arm.com//-/media/Arm%20Developer%20Community/Images/Tutorial%20Guide%20Diagrams%20and%20Screenshots/Machine%20Learning/Running%20AlexNet%20on%20Pi%20with%20Compute%20Library/compute_library_alexnet.zip?revision=c1a232fa-f328-451f-9bd6-250b83511e01 -O compute_library_alexnet.zip &> /dev/null")
    print("Unpacking assets...")
    os.system("unzip compute_library_alexnet.zip")
    os.system("rm compute_library_alexnet.zip")
    os.chdir(WORKSPACE_DIR)


def main():
    os.system("git submodule update --init --recursive")
    dl_thread = multiprocessing.Process(target=download_images)
    dl_thread.start()
    build_compute_library()
    print("Waiting for image downloading...", flush=True)
    dl_thread.join()
    download_assets()


if __name__ == "__main__":
    main()
