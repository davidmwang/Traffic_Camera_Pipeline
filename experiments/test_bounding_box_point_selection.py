from tcp.object_detection.video_labeler import VideoLabeler
from tcp.configs.alberta_config import Config
from tcp.registration.homography import Homography

import cPickle as pickle
import cv2
import glob
import os
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import time
import pylab as pl

VIDEO_NAME = 'alberta_cam_original_2017-10-26_16-33-45'

bboxes = pickle.load(open("{0}/{0}_bboxes.cpkl".format(VIDEO_NAME), 'r'))
classes = pickle.load(open("{0}/{0}_classes.cpkl".format(VIDEO_NAME), 'r'))
init_labels = pickle.load(open("{0}/{0}_init_labels.cpkl".format(VIDEO_NAME), 'r'))

img = cv2.imread(os.path.join(VIDEO_NAME + "_images", '%s_%07d.jpg' % (VIDEO_NAME, 0)))
# cv2.imshow('img', img)
# cv2.waitKey(100000)

print(len(bboxes))
print(bboxes[0])

print(len(classes))
print(classes[0])

print(len(init_labels))
print(init_labels[0])
print(init_labels[1])

cnfg = Config()
hm = Homography(cnfg)

# test_bboxes = bboxes[0]
# points = []
# for i in range(len(test_bboxes)):
#   print(test_bboxes[i])
#   x_min, y_min, x_max, y_max = test_bboxes[i]
#   result = hm.transform_point((x_min + x_max) / 2.0, y_max, use_offset=True)
#   print(result)
#   points.append(result)

# points = np.array(points)
# print(points.shape)
# print(points[:0])
# print(points[:1])

# img = cv2.imread("alberta_cam_original_2017-10-26_16-33-45_images/alberta_cam_original_2017-10-26_16-33-45_0000000.jpg")
# img_warped = hm.apply_homography_on_img(img)
# plt.imshow(img_warped)

# plt.show()

fig, axarr = plt.subplots(2)

img = plt.imread(os.path.join(VIDEO_NAME + "_images", '%s_%07d.jpg' % (VIDEO_NAME, 0)))
img_plot = axarr[0].imshow(img)

axes = axarr[1]
axes.set_xlim([-100,1100])
axes.set_ylim([-100,1100])
axes.invert_yaxis()

axes.plot([400, 400], [0, 1000], color='k')
axes.plot([500, 500], [0, 1000], color='k')
axes.plot([600, 600], [0, 1000], color='k')

axes.plot([0, 1000], [400, 400], color='k')
axes.plot([0, 1000], [500, 500], color='k')
axes.plot([0, 1000], [600, 600], color='k')

test_bboxes = bboxes[0]
points = []
for j in range(len(test_bboxes)):
    x_min, y_min, x_max, y_max = test_bboxes[j]
    result = hm.transform_point((x_min + x_max) / 2.0, y_max * 3.0 / 4 + y_min * 1.0 / 4, use_offset=False)
    points.append(result)

points = np.array(points)
scatter_plot = axes.scatter(points[:,0], points[:,1])

def animate(i):
    print("Frame", i)

    img = plt.imread(os.path.join(VIDEO_NAME + "_images", '%s_%07d.jpg' % (VIDEO_NAME, i)))
    img_plot.set_data(img)

    test_bboxes = bboxes[i]
    points = []
    for j in range(len(test_bboxes)):
        x_min, y_min, x_max, y_max = test_bboxes[j]
        result = hm.transform_point((x_min + x_max) / 2.0, y_max * 3.0 / 4 + y_min * 1.0 / 4, use_offset=False)
        points.append(result)

    points = np.array(points)
    scatter_plot.set_offsets(points)

    return img_plot, scatter_plot

ani = animation.FuncAnimation(fig, animate, np.arange(1, 100), interval=5, repeat=False)
plt.show()
