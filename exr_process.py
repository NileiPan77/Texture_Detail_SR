import cv2 as cv
import numpy as np
import pyexr
import os
import sys
import time
import argparse
import multiprocessing as mp
from tqdm import tqdm

def exr2png(exr_path, png_path):
    exr = pyexr.open(exr_path)
    img = exr.get()
    img = (img * 255).astype(np.uint8)
    img = np.clip(img, 0, 255)
    cv.imwrite(png_path, img)

def png2exr(png_path, exr_path):
    img = cv.imread(png_path)
    img = img.astype(np.float32) / 255
    pyexr.write(exr_path, img)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='.\exrs', help='input folder')
    parser.add_argument('--output', type=str, default='.\pngs', help='output folder')
    parser.add_argument('--mode', type=str, default='exr2png', help='exr2png or png2exr')
    parser.add_argument('--block_size', type=int, default=1024, help='block size for multi-process')
    parser.add_argument('--multi_process', type=bool, default=False, help='multi-process or not')
    args = parser.parse_args()
    print("Multi-process: ", args.multi_process)
    start = time.time()
    if args.mode == 'exr2png':
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        # walk through all files in input folder
        for root, dirs, files in os.walk(args.input):
            # if multi-process
            if args.multi_process:
                pool = mp.Pool(mp.cpu_count())
                for file in files:
                    # if file is exr file
                    if file.endswith('.exr'):
                        print('Processing: ', file)
                        png_path = os.path.join(args.output, file.replace('.exr', '.png'))
                        # if png file already exists, skip
                        if os.path.exists(png_path):
                            continue
                        exr_path = os.path.join(root, file)
                        pool.apply_async(exr2png, args=(exr_path, png_path))
                pool.close()
                pool.join()
            # if single-process
            else:
                for file in files:
                    # if file is exr file
                    if file.endswith('.exr'):
                        print('Processing: ', file)
                        png_path = os.path.join(args.output, file.replace('.exr', '.png'))
                        # if png file already exists, skip
                        if os.path.exists(png_path):
                            continue
                        exr_path = os.path.join(root, file)
                        exr2png(exr_path, png_path)
    elif args.mode == 'png2exr':
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        # walk through all files in input folder
        for root, dirs, files in os.walk(args.input):
            if args.multi_process:
                pool = mp.Pool(mp.cpu_count())
                for file in files:
                    # if file is png file
                    if file.endswith('.png'):
                        print('Processing: ', file)
                        png_path = os.path.join(root, file)
                        exr_path = os.path.join(args.output, file.replace('.png', '.exr'))
                        # if exr file already exists, skip
                        if os.path.exists(exr_path):
                            continue
                        pool.apply_async(png2exr, args=(png_path, exr_path))
                pool.close()
                pool.join()
            else:
                for file in files:
                    # if file is png file
                    if file.endswith('.png'):
                        png_path = os.path.join(root, file)
                        exr_path = os.path.join(args.output, file.replace('.png', '.exr'))
                        # if exr file already exists, skip
                        if os.path.exists(exr_path):
                            continue
                        png2exr(png_path, exr_path)
    else:
        print('mode error')
        sys.exit(0)
    print('Time used: ', time.time() - start)

if __name__ == '__main__':
    main()