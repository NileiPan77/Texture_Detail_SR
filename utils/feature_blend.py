import cv2 as cv
import numpy as np
import pyexr
import os
import sys
import time
import argparse
import multiprocessing as mp
from tqdm import tqdm
import face_recognition

# extract facial landmarks from image
# return: list of landmarks
def get_landmarks(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
    return face_landmarks_list

# function to blend two pixels
def blend_f(p1, p2, alpha):
    # Linear blending
    p1 = p1 * (1 - alpha) + p2 * alpha
    return p1

def highlight_landmarks(img, landmarks):
    # highlight landmarks
    if len(landmarks) == 0:
        return img
    for landmark in landmarks:
        for key in landmark.keys():
            # fill in the polygon
            if key == 'top_lip' or key == 'bottom_lip' or key == 'left_eyebrow' or key == 'right_eyebrow' or key == 'left_eye' or key == 'right_eye':
                points = np.array(landmark[key], np.int32)
                points = points.reshape((-1, 1, 2))
                cv.fillPoly(img, [points], (255, 255, 255), lineType=cv.LINE_AA)
    # combine points of nose_bridge and nose_tip
    # leave only the top point in nose_bridge

    landmark['nose_bridge'] = [landmark['nose_bridge'][0]]
    points = np.array(landmark['nose_bridge'] + landmark['nose_tip'], np.int32)
    points = points.reshape((-1, 1, 2))
    cv.fillPoly(img, [points], (255, 255, 255), lineType=cv.LINE_AA)
    return img

# Only blend the face regions that are not part of the landmarks
# return: blended image
def blend_face(img1, img2):
    height, width, channels = img1.shape
    
    # get landmarks
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)
    
    # Store a copy of the landmark regions
    img1_landmark = np.zeros((height, width, channels), np.uint8)
    img2_landmark = np.zeros((height, width, channels), np.uint8)
    img1_cp = img1.copy()
    img2_cp = img2.copy()
    highlight_landmarks(img1_cp, landmarks1)
    highlight_landmarks(img2_cp, landmarks2)
    
    # Landmark only images
    img1_landmark[img1_cp == 255] = img1[img1_cp == 255]
    img2_landmark[img2_cp == 255] = img2[img2_cp == 255]

    # write out landmark only images for debugging
    cv.imwrite('img1_landmark.png', img1_landmark)
    cv.imwrite('img2_landmark.png', img2_landmark)

    # Blend the non-landmark regions
    # img1 = blend_f(img1, img2, 0.5)

    
    return img1


def blend_feature(img1, img2):
    # read images

    # The image to blend
    img_to_blend = cv.imread(img1)

    # The image to blend with
    target_img = cv.imread(img2)

    # A diffuse/albedo image that has the same size as the input image
    # and contains clear face features
    # img_to_extract_landmarks = cv.imread(args[3])
    
    # TODO: blend images
    blend_face(img_to_blend, target_img)

if __name__ == '__main__':
    blend_feature(sys.argv[1], sys.argv[2])