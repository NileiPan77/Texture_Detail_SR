# Combined script for feature transfer
# 1. Extract landmarks from both images
# 2. Align images using landmarks (optional)
# 3. Transfer features from img2 to img1

from texture_align import *
from extract_facial_landmark import *
from fourier_filtering import *
import cv2
import numpy as np
import os
import argparse
import sys
import pyexr
import time

def feature_transfer(args):
    # load images
    print("Loading images...")
    base_cavity = pyexr.open(args.base_cavity).get('B')[:, :, 0]
    target_height = pyexr.open(args.target_height).get('B')[:, :, 0]

    # shape check
    print("Checking image shapes...")
    if base_cavity.shape != target_height.shape:
        print('Image shape not match')
        print('base_cavity.shape: ', base_cavity.shape)
        print('target_height.shape: ', target_height.shape)
        return

    # align images
    print("Aligning images...")
    if args.align:
        base_diffuse = pyexr.open(args.base_diffuse).get()
        target_diffuse = pyexr.open(args.target_diffuse).get()
        # align images
        base_diffuse, base_cavity = align_using_landmarks(base_diffuse, target_diffuse, base_cavity)
        del target_diffuse
    
    # transfer features
    print("Transferring features...")
    start = time.time()
    new_cavity = cufourier_filtering(target_height, base_cavity, args.cutoff_low, args.cutoff_high, args.degree)
    print(f'Time used on fourier: {time.time() - start}')
    del base_cavity, target_height

    # save result
    print("Saving result...")
    
    if not os.path.exists(args.output_dir):
        print("Creating output directory...")
        os.makedirs(args.output_dir)
    
    output_name = 'combined_cavity_' + str(args.cutoff_low) + '_' + str(args.cutoff_high) + '_' + str(args.degree) + '_' +args.output_name
    if os.path.exists(args.output_dir + output_name):
        output_name = 'combined_cavity_' + str(args.cutoff_low) + '_' + str(args.cutoff_high) + '_' + str(args.degree) + '_1_' + args.output_name
    
    # convert to 3 channels
    new_cavity = np.stack((new_cavity,)*3, axis=-1)
    
    pyexr.write(args.output_dir + output_name, new_cavity)
    

if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser(description='Transfer features from img2 to img1')
    parser.add_argument('--base_diffuse', type=str, help='path to base diffuse map', default='.\target\base_diffuse.png')
    parser.add_argument('--base_cavity', type=str, help='path to base cavity map', default='.\target\base_cavity.exr')
    parser.add_argument('--target_diffuse', type=str, help='path to target diffuse map', default='.\target\target_diffuse.png')
    parser.add_argument('--target_height', type=str, help='path to target height map', default='.\target\target_height.exr')
    parser.add_argument('--output_dir', type=str, help='path to output directory', default='./target/results/')
    parser.add_argument('--align', type=bool, help='align images using landmarks', default=False)
    parser.add_argument('--cutoff_high', type=float, help='cutoff high frequency for fourier filtering', default=60)
    parser.add_argument('--cutoff_low', type=float, help='cutoff low frequency for fourier filtering', default=120)
    parser.add_argument('--degree', type=int, help='degree of polynomial for fourier filtering', default=10)
    parser.add_argument('--output_name', type=str, help='output name', default='out_cavi.exr')
    args = parser.parse_args()

    start = time.time()
    # run feature transfer
    feature_transfer(args)

    print(f'Time used in total: {time.time() - start}')