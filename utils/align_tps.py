from skimage.transform import PiecewiseAffineTransform, warp
import numpy as np
import cv2
import matplotlib.pyplot as plt
source_img = cv2.imread('.\\data\\out_1080p.png')
target_img = cv2.imread('.\\data\\diffuse_1080p.png')

source_path = '.\\data\\out_1080p.png'
target_path = '.\\data\\diffuse_1080p.png'

# load points in txt
target_points = np.loadtxt('.\\data\\target_points.txt')
source_points = np.loadtxt('.\\data\\source_points.txt')


# plot points, connect source and target points whose indices are the same
plt.figure(figsize=(10, 10))
plt.imshow(source_img)
plt.scatter(target_points[:, 0], target_points[:, 1], c='r')
plt.scatter(source_points[:, 0], source_points[:, 1], c='b')
for i in range(len(target_points)):
    plt.plot([target_points[i, 0], source_points[i, 0]], [target_points[i, 1], source_points[i, 1]], c='g')
plt.savefig('.\\data\\points.png')


# Define the TPS transform function
tps = PiecewiseAffineTransform()
tps.estimate(target_points, source_points)

# Warp source image so it fits target_image
out_image = warp(source_img, tps)

# plot points, connect source and target points whose indices are the same after warping
plt.figure(figsize=(10, 10))
plt.imshow(out_image)
plt.scatter(target_points[:, 0], target_points[:, 1], c='r')
plt.scatter(source_points[:, 0], source_points[:, 1], c='b')
for i in range(len(target_points)):
    plt.plot([target_points[i, 0], source_points[i, 0]], [target_points[i, 1], source_points[i, 1]], c='g')
plt.savefig('.\\data\\points_warped.png')
out_image = (out_image * 255).astype(np.uint8)
cv2.imwrite('.\\data\\warped_image.jpg', out_image)