# Align 2 face images using points detected by face-recognition library
import cv2 as cv
import numpy as np
import os
import argparse
import sys
import face_recognition
from extract_facial_landmark import *
from matplotlib import pyplot as plt
# extract facial landmarks from image
# return: list of landmarks
def get_landmarks(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            face_landmarks[facial_feature] = face_landmarks[facial_feature][0:4]

    # only use the mean of the points
    new_face_landmarks_list = []
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            new_face_landmarks_list.append({facial_feature: np.mean(face_landmarks[facial_feature], axis=0)})
    
    return new_face_landmarks_list

# find landmarks on low resolution image
# return: list of landmarks
def get_landmarks_low(img):
    # resize image
    img = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
    face_landmarks_list = get_landmarks(img)
    # scale back
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            face_landmarks[facial_feature] = face_landmarks[facial_feature] * 4
    return face_landmarks_list


def get_alignment_matrix(src, dest):
    # Create matrix for source points
    ones = np.ones(len(src))
    points_src_matrix = np.hstack([src, ones[:, None]])

    # Solve the linear system
    transformation_matrix, _, _, _ = np.linalg.lstsq(points_src_matrix, dest, rcond=None)

    return transformation_matrix

def align_using_landmarks(img1, img2, img_disp):
    # shape check
    if img1.shape != img2.shape:
        print('Image shape not match')
        return
    height, width, channels = img1.shape

    # get landmarks
    landmarks1 = mean_of_landmarks_np(img1)
    landmarks2 = mean_of_landmarks_np(img2)

    # get transformation matrix
    transformation_matrix = get_alignment_matrix(landmarks1, landmarks2)

    # 2nd method: use cv.homography
    # transformation_matrix, mask = cv.findHomography(landmarks1, landmarks2, cv.RANSAC, 5.0)
    
    print('transformation_matrix: ', transformation_matrix)
    # apply transformation matrix
    img1_aligned = cv.warpAffine(img1, transformation_matrix.T, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    
    print("first warp done")
    width, height, channels = img_disp.shape
    # apply transformation matrix to displacement map of img1
    img_disp_aligned = cv.warpAffine(img_disp, transformation_matrix.T, (width, height), flags=cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)
    print("second warp done")
    # img1_aligned = cv.warpPerspective(img1, transformation_matrix, (width, height))
    return (img1_aligned, img_disp_aligned)



# return img1_aligned 
def tex_align(img1_path, img2_path, out_path, img3_path):
    # load images
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)
    img3 = cv.imread(img3_path)

    # align images
    img1_aligned, disp = align_using_landmarks(img1, img2, img3)

    print("aligned")
    # save image
    cv.imwrite(out_path, img1_aligned)
    cv.imwrite(r'.\target\results\aligned_tar_disp.png', disp)
    return img1_aligned

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Align 2 face images using points detected by face-recognition library')

    parser.add_argument('--img1', type=str, required=True, help='src diffuse image')
    parser.add_argument('--img2', type=str, required=True, help='dest diffuse image')
    parser.add_argument('--img3', type=str, help='Path dest displacement map', default=r'.\target\tar_disp_16k.png')
    parser.add_argument('--out', type=str, help='Path to the output image', default=r'.\target\results\aligned_tar.png')
    args = parser.parse_args()
    tex_align(args.img1, args.img2, args.out, args.img3)