
import numpy as np
import cv2
import os
from tps import ThinPlateSpline
import scipy.ndimage
from mmpose.apis import MMPoseInferencer
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'


source_img = cv2.imread('.\\data\\out_1080p.png')
target_img = cv2.imread('.\\data\\diffuse_1080p.png')

# These points should be detected using a facial landmark detection model
# target_points = [dlib.dpoint(x, y) for (x, y) in [[15, 15], [25, 25], [35, 35], [45, 45], [55, 55]]]

# warped_img = apply_tps(source_img, source_points, target_points)

# cv2.imwrite('warped_image.jpg', cv2.cvtColor(warped_img, cv2.COLOR_RGB2BGR))

source_path = '.\\data\\out_1080p.png'
target_path = '.\\data\\diffuse_1080p.png'


# inference
inferencer = MMPoseInferencer('face')

source_generator = inferencer(source_path, show=False)
target_generator = inferencer(target_path, show=False)

source_result = next(source_generator)
target_result = next(target_generator)

source_points = source_result['predictions'][0][0]['keypoints']
target_points = target_result['predictions'][0][0]['keypoints']

# save points in txt
np.savetxt('.\\data\\source_points.txt', source_points)
np.savetxt('.\\data\\target_points.txt', target_points)
