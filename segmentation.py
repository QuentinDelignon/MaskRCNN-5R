import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

path = 'C:\\Users\\Quentin Delignon\\Documents\\Python\\PJE\\segmentation\\TestingImageDetection\\Images1\\WhatsApp Image 2020-04-29 at 18.39.31 (1).jpeg'
img = cv.imread('coins.png')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(gray,0,255,cv.THRESH_BINARY_INV+cv.THRESH_OTSU)
plt.imshow(img)
plt.show()
plt.imshow(thresh)
plt.show()

# noise removal
kernel = np.ones((3,3),np.uint8)
opening = cv.morphologyEx(thresh,cv.MORPH_OPEN,kernel, iterations = 2)
# sure background area
sure_bg = cv.dilate(opening,kernel,iterations=3)
# Finding sure foreground area
dist_transform = cv.distanceTransform(opening,cv.DIST_L2,5)
ret, sure_fg = cv.threshold(dist_transform,0.7*dist_transform.max(),255,0)
# Finding unknown region
sure_fg = np.uint8(sure_fg)
unknown = cv.subtract(sure_bg,sure_fg)
plt.imshow(sure_fg)
plt.show()
plt.imshow(dist_transform)
plt.show()

# Marker labelling
ret, markers = cv.connectedComponents(sure_fg)
# Add one to all labels so that sure background is not 0, but 1
markers = markers+1
# Now, mark the region of unknown with zero
markers[unknown==255] = 0
plt.imshow(markers,colormap='jet')
plt.show()

markers = cv.watershed(img,markers)
img[markers == -1] = [255,0,0]
plt.imshow(markers,colormap='jet')
plt.show()
plt.imshow(img)
plt.show()
