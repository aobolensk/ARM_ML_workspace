#!/usr/bin/env python3

import argparse
import datetime
import os
import shutil
import subprocess

import cv2

IMAGES_DIR = "imagenet"
TMP_DIR = "tmp"
ALEXNET_ASSETS_DIR = "alexnet_assets"

args = None
top1_count = 0
top5_count = 0

def process_image(image_path: str):
    global top1_count, top5_count
    # Resize image and convert it to ppm format
    img = cv2.imread(image_path)
    img = cv2.resize(img, (227, 227))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    wnid = image_name.split("_")[0]
    ppm_path = os.path.join(TMP_DIR, image_name + '.ppm')
    cv2.imwrite(ppm_path, img)

    # Run AlexNet
    proc = subprocess.Popen([
        "ComputeLibrary/build/examples/graph_alexnet",
        f"--image={ppm_path}",
        f"--data={ALEXNET_ASSETS_DIR}",
        f"--labels={os.path.join(ALEXNET_ASSETS_DIR, 'labels.txt')}",
        "--target=neon",
        f"--threads={os.cpu_count()}",
        ], env={
            "LD_LIBRARY_PATH": os.path.join(os.getcwd(), "ComputeLibrary", "build")
        }, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    out, err = out.decode("utf-8"), err.decode("utf-8")
    if proc.returncode != 0:
        print("Error:", err)
        exit(1)
    if args.verbose:
        print(out)

    # Parse predictions
    prediction = 0
    for line in list(filter(None, out.splitlines())):
        if line == "---------- Top 5 predictions ----------":
            prediction += 1
            continue
        if prediction > 0:
            line = line.split()
            got_wnid = line[5]
            if wnid == got_wnid:
                if prediction == 1:
                    print("> Got top 1", flush=True)
                    top1_count += 1
                else:
                    print("> Got top 5", flush=True)
                    top5_count += 1
                break
            prediction += 1
            if prediction > 5:
                break
    else:
        print("> Got nothing", flush=True)

def main():
    global top1_count, top5_count, args
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument("-c", "--count", type=int)
    args = parser.parse_args()
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    images = os.listdir(IMAGES_DIR)
    print(f"Start: {datetime.datetime.now()}")
    for index, image in enumerate(images):
        if index == args.count:
            break
        print(f"Processing image {image}")
        process_image(os.path.join(IMAGES_DIR, image))
    print(f"Stats: Top-1: {top1_count}, Top-5: {top5_count}")
    print(f"Finish: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
