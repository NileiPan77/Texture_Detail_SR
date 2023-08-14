# Covert input image to grayscale and write out

import pyexr
import numpy as np
import argparse
import sys
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Covert input image to grayscale and write out')
    parser.add_argument('--input', type=str, help='input image path', default='.\input.exr')
    parser.add_argument('--output', type=str, help='output image path', default='.\output.exr')
    args = parser.parse_args()
    default_output = args.input.split('.')[0] + '_gray.exr'
    # Read in image and convert to grayscale
    img = pyexr.open(args.input).get('B')[:, :, 0]
    # Write out image
    pyexr.write(args.output, img, channel_names=['B'])