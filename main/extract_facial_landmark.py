import cv2 as cv
import numpy as np
import os
import argparse
import face_recognition

# extract facial landmarks from image
# return: list of landmarks
def get_landmarks(img):
    face_landmarks_list = face_recognition.face_landmarks(img)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
    return face_landmarks_list

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
    landmark['nose_bridge'] = [landmark['nose_bridge'][0]]
    points = np.array(landmark['nose_bridge'] + landmark['nose_tip'], np.int32)
    points = points.reshape((-1, 1, 2))
    cv.fillPoly(img, [points], (255, 255, 255), lineType=cv.LINE_AA)
    return img

def mean_of_landmarks_np(img):
    scale = False
    if img.shape[0] > 4096:
        img = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_AREA)
        scale = True
    
    face_landmarks_list = face_recognition.face_landmarks(img)
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            face_landmarks[facial_feature] = face_landmarks[facial_feature][0:4]

    # only use the mean of the points
    new_face_landmarks_list = []
    for face_landmarks in face_landmarks_list:
        for facial_feature in face_landmarks.keys():
            new_face_landmarks_list.append({facial_feature: np.mean(face_landmarks[facial_feature], axis=0)})
    
    if scale:
        for face_landmarks in new_face_landmarks_list:
            for facial_feature in face_landmarks.keys():
                face_landmarks[facial_feature] = face_landmarks[facial_feature] * 4
    
    new_landmarks1 = np.array([])
    for i in range(len(new_face_landmarks_list)):
        for facial_feature in new_face_landmarks_list[i].keys():
            if new_landmarks1.size == 0:
                new_landmarks1 = np.array(new_face_landmarks_list[i][facial_feature])
            else:
                new_landmarks1 = np.concatenate((new_landmarks1, np.array(new_face_landmarks_list[i][facial_feature])), axis=0)

    new_face_landmarks_list = new_landmarks1.reshape(-1,2)
    return new_face_landmarks_list

# write out 4 images: img1_wo_landmark, img2_wo_landmark, img1_landmark_only, img2_landmark_only
def write_landmark_images(img1, img2, output_dir):
    height, width, channels = img1.shape
    # shape check
    if img1.shape != img2.shape:
        print('Image shape not match')
        return

    # get landmarks
    landmarks1 = get_landmarks(img1)
    landmarks2 = get_landmarks(img2)
    
    # Store a copy of the landmark regions
    img1_landmark_only = np.zeros((height, width, channels), np.uint8)
    img2_landmark_only = np.zeros((height, width, channels), np.uint8)
    img1_wo_landmarks = img1.copy()
    img2_wo_landmarks = img2.copy()
    highlight_landmarks(img1_wo_landmarks, landmarks1)
    highlight_landmarks(img2_wo_landmarks, landmarks2)
    
    # Landmark only images
    img1_landmark_only[img1_wo_landmarks == 255] = img1[img1_wo_landmarks == 255]
    img2_landmark_only[img2_wo_landmarks == 255] = img2[img2_wo_landmarks == 255]

    # write out landmark only images for debugging
    cv.imwrite(os.path.join(output_dir, 'img1_landmark_only.png'), img1_landmark_only)
    cv.imwrite(os.path.join(output_dir, 'img2_landmark_only.png'), img2_landmark_only)

    # write out images without landmarks
    cv.imwrite(os.path.join(output_dir, 'img1_wo_landmarks.png'), img1_wo_landmarks)
    cv.imwrite(os.path.join(output_dir, 'img2_wo_landmarks.png'), img2_wo_landmarks)

def landmark_extarct(img1_path, img2_path, output_dir):
    # read images
    img1 = cv.imread(img1_path)
    img2 = cv.imread(img2_path)

    # make output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # make landmark images
    write_landmark_images(img1, img2, output_dir)



if __name__ == '__main__':
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--img1', type=str, required=True, help='path to image 1')
    parser.add_argument('--img2', type=str, required=True, help='path to image 2')
    parser.add_argument('--output_dir', type=str, required=True, help='path to output directory')
    args = parser.parse_args()
    landmark_extarct(args.img1, args.img2, args.output_dir)
    