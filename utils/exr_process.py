import cv2 as cv
import numpy as np
import pyexr
import os
import sys
import time
import argparse
import tifffile as tiff
import multiprocessing as mp
from tqdm import tqdm


def exr2tiff(exr_path, tiff_path):
    exr = pyexr.open(exr_path)
    img = exr.get()
    tiff.imwrite(tiff_path, img.astype(np.float16)[:, :, 0])

def tiff2exr(tiff_path, exr_path):
    img = tiff.imread(tiff_path)
    pyexr.write(exr_path, img.astype(np.float32))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='.\exrs', help='input folder')
    parser.add_argument('--output', type=str, default='.\tiffs', help='output folder')
    parser.add_argument('--mode', type=str, default='exr2tiff', help='exr2tiff or tiff2exr')
    parser.add_argument('--block_size', type=int, default=1024, help='block size for multi-process')
    parser.add_argument('--multi_process', type=bool, default=False, help='multi-process or not')
    args = parser.parse_args()
    print("Multi-process: ", args.multi_process)
    start = time.time()
    if args.mode == 'exr2tiff':
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
                        tiff_path = os.path.join(args.output, file.replace('.exr', '.tiff'))
                        # if tiff file already exists, skip
                        if os.path.exists(tiff_path):
                            continue
                        exr_path = os.path.join(root, file)
                        pool.apply_async(exr2tiff, args=(exr_path, tiff_path))
                pool.close()
                pool.join()
            # if single-process
            else:
                for file in files:
                    # if file is exr file
                    if file.endswith('.exr'):
                        print('Processing: ', file)
                        tiff_path = os.path.join(args.output, file.replace('.exr', '.tiff'))
                        # if tiff file already exists, skip
                        if os.path.exists(tiff_path):
                            continue
                        exr_path = os.path.join(root, file)
                        exr2tiff(exr_path, tiff_path)
    elif args.mode == 'tiff2exr':
        if not os.path.exists(args.output):
            os.makedirs(args.output)
        # walk through all files in input folder
        for root, dirs, files in os.walk(args.input):
            if args.multi_process:
                pool = mp.Pool(mp.cpu_count())
                for file in files:
                    # if file is tiff file
                    if file.endswith('.tiff'):
                        print('Processing: ', file)
                        tiff_path = os.path.join(root, file)
                        exr_path = os.path.join(args.output, file.replace('.tiff', '.exr'))
                        # if exr file already exists, skip
                        if os.path.exists(exr_path):
                            continue
                        pool.apply_async(tiff2exr, args=(tiff_path, exr_path))
                pool.close()
                pool.join()
            else:
                for file in files:
                    # if file is tiff file
                    if file.endswith('.tiff'):
                        tiff_path = os.path.join(root, file)
                        exr_path = os.path.join(args.output, file.replace('.tiff', '.exr'))
                        # if exr file already exists, skip
                        if os.path.exists(exr_path):
                            continue
                        tiff2exr(tiff_path, exr_path)
    else:
        print('mode error')
        sys.exit(0)
    print('Time used: ', time.time() - start)

if __name__ == '__main__':
    main()