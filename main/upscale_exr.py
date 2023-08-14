# upscale a exr image

import argparse
import cv2 as cv
import numpy as np
import os
import cProfile
import pyexr

def bicubic_upscale(img, scale):
    return cv.resize(img.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)

def nearest_upscale(img, scale):
    return cv.resize(img.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv.INTER_NEAREST)

def bilinear_upscale(img, scale):
    return cv.resize(img.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv.INTER_LINEAR)

def lanczos_upscale(img, scale):
    return cv.resize(img.astype(np.float32), None, fx=scale, fy=scale, interpolation=cv.INTER_LANCZOS4)

def main():
    parser = argparse.ArgumentParser(description='Upscale a exr image')
    parser.add_argument('--input', type=str, help='input image')
    parser.add_argument('--output', type=str, help='output image')
    parser.add_argument('--scale', type=float, default=2.0, help='scale factor')
    parser.add_argument('--method', type=str, default='bicubic', help='method for upscaling')
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print('Input file does not exist')
        return

    # read a exr image
    img = pyexr.open(args.input).get()

    if args.method == 'bicubic':
        new_img = bicubic_upscale(img, args.scale)
    elif args.method == 'nearest':
        new_img = nearest_upscale(img, args.scale)
    elif args.method == 'bilinear':
        new_img = bilinear_upscale(img, args.scale)
    elif args.method == 'lanczos':
        new_img = lanczos_upscale(img, args.scale)
    else:
        print('Invalid method')
        return

    #conver to 3 channels
    new_img = np.stack((new_img, new_img, new_img), axis=-1)
    pyexr.write(args.output, new_img)

if __name__ == '__main__':
    cProfile.run('main()', sort='cumtime')