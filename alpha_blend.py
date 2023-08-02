# merge 2 images with alpha blending
import cv2
import numpy as np
import os
import argparse

def alpha_blend(img1, img2, output_dir, alpha=1):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    # make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # resize img2 to img1's size
    if np.shape(img1) != np.shape(img2):
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # print shapes
    print('img1 shape: ', np.shape(img1))
    print('img2 shape: ', np.shape(img2))

    # blend
    blended = cv2.addWeighted(img1, alpha, img2, 1-alpha, 0)
    cv2.imwrite(os.path.join(output_dir, 'fix.png'), blended)

# write out one image on top of another, overwriting the pixels
def overwrite(img1, img2):
    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)
    
    # overwrite part of img1 with img2
    img1[img2 != 0] = img2[img2 != 0]

    cv2.imwrite('.\\data\\overwrite.png', img1)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', help='path to image 1', default='.\\data\\out_1080p.png')
    parser.add_argument('--img2', help='path to image 2', default='.\\data\\warped_image.jpg')
    parser.add_argument('--output_dir', help='path to output directory', default='.\\data\\')
    parser.add_argument('--alpha', help='alpha value for blending', default=0.5)
    args = parser.parse_args()
    # alpha_blend(args.img1, args.img2, args.output_dir, args.alpha)
    overwrite(args.img1, args.img2)