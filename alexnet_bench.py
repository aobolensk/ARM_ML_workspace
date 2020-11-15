#!/usr/bin/env python3

import argparse
import datetime
import os
import shutil
import subprocess
from multiprocessing.pool import ThreadPool

from PIL import Image

IMAGES_DIR = "imagenet"
TMP_DIR = "tmp"
ALEXNET_ASSETS_DIR = "alexnet_assets"

args = None
top1_count = 0
top5_count = 0

def process_image(image_path: str):
    global top1_count, top5_count
    # Resize image and convert it to ppm format
    img = Image.open(image_path)
    img = img.resize((227, 227))
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    wnid = image_name.split("_")[0]
    ppm_path = os.path.join(TMP_DIR, image_name + '.ppm')
    img.save(ppm_path)

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
        return
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

def main():
    global top1_count, top5_count, args
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose")
    parser.add_argument("-c", "--count", type=int)
    parser.add_argument("-j", "--jobs", type=int, default=1)
    args = parser.parse_args()
    if os.path.isdir(TMP_DIR):
        shutil.rmtree(TMP_DIR)
    os.mkdir(TMP_DIR)
    images = os.listdir(IMAGES_DIR)
    print(f"Start: {datetime.datetime.now()}")
    pool = ThreadPool(args.jobs)
    pool.map(process_image, list(os.path.join(IMAGES_DIR, image) for image in images)[:args.count])
    pool.close()
    print(f"Stats: Top-1: {top1_count}, Top-5: {top5_count}")
    print(f"Finish: {datetime.datetime.now()}")

if __name__ == "__main__":
    main()
