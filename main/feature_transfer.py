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
import configparser
from upscale_exr import *

# Define a struct for parameters
class Parameters:
    def __init__(self, config, section='DEFAULT'):
        self.base_diffuse = config.get(section, 'base_diffuse')
        self.base_cavity = config.get(section, 'base_cavity')
        self.target_diffuse = config.get(section, 'target_diffuse')
        self.target_height = config.get(section, 'target_height')
        self.output_dir = config.get(section, 'output_dir')
        self.align = config.getboolean(section, 'align')
        self.cutoff_high = config.getint(section, 'cutoff_high')
        self.cutoff_low = config.getint(section, 'cutoff_low')
        self.degree = config.getint(section, 'degree')
        self.output_name = config.get(section, 'output_name')

def feature_transfer(args):
    # load images
    print("Loading images...")
    base_cavity = pyexr.open(args.base_cavity).get()[:, :, 0]
    target_height = pyexr.open(args.target_height).get()[:, :, 0]

    # shape check
    print("Checking image shapes...")
    if base_cavity.shape != target_height.shape:
        scale = target_height.shape[0] / base_cavity.shape[0]
        print('Image shape not match')
        print('base_cavity.shape: ', base_cavity.shape)
        print('target_height.shape: ', target_height.shape)
        print('Scaling base_cavity by ', scale)
        base_cavity = bicubic_upscale(base_cavity, scale)
        

    # align images
    print("Aligning images...")
    if args.align:
        base_diffuse = pyexr.open(args.base_diffuse).get()
        target_diffuse = pyexr.open(args.target_diffuse).get()
        # align images
        base_diffuse, base_cavity = align_using_landmarks(base_diffuse, target_diffuse, base_cavity)
        del target_diffuse, base_diffuse
    
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

    # check sys args
    if len(sys.argv) != 2:
        print('Usage: python feature_transfer.py [section_name]')
        exit()

    # get section name
    section = sys.argv[1]

    # parse arguments
    config = configparser.ConfigParser()
    config.read('./config.ini')
    args = Parameters(config, section)

    # run feature transfer
    feature_transfer(args)