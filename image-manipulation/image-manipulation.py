import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

checkers = cv2.imread(r'C:\openCV-bootcamp\image-manipulation\checkerboard_18x18.png')
print(checkers)

## ACCESSING INDIVIDUAL PIXELS ##
# First pixel at origin
print(checkers[0, 0])
# First pixel of the next white box
print(checkers[0, 6])

## MODIFY IMAGE PIXELS ##
# Make a copy of OG image
checkers_copy = checkers.copy()
checkers_copy[0, 0] = 150
checkers_copy[0, 1] = 150
checkers_copy[1, 0] = 200
checkers_copy[1, 1] = 200

plt.imshow(checkers_copy, cmap='gray')
print(checkers_copy)

## CROPPING ##
img_NZ_BGR = cv2.imread(r'C:\openCV-bootcamp\image-manipulation\New_Zealand_Boat.jpg', cv2.IMREAD_COLOR)
img_NZ_RGB = cv2.cvtColor(img_NZ_BGR, cv2.COLOR_BGR2RGB) # Bootcamp used img_NZ_rgb = img_NZ_bgr[:, :, ::-1]
plt.imshow(img_NZ_RGB)

cropped = img_NZ_RGB[200:400, 300:600]
plt.imshow(cropped)


## RESIZE IMAGE ##
# resize( src, dsize[, dst[, fx[, fy[, interpolation]]]] )
# src: image, dsize: output image size, fx: horizontal scale, fy: vertical scale

# Scale by 4
cropped_4x = cv2.resize(cropped, None, fx=4, fy=4)
plt.imshow(cropped_4x)

# Exact size
cropped_exact = cv2.resize(cropped, dsize=(150, 275))
plt.imshow(cropped_exact)

# Aspect Ratio
cropped_aspect_ratio = 150 / cropped.shape[1] # new width / cropped width
aspect_height = int(cropped.shape[0] * cropped_aspect_ratio)

cropped_aspect = cv2.resize(cropped, dsize=(150, aspect_height))
plt.imshow(cropped_aspect)

## FLIPPING IMAGES ##
# cv.flip( src, flipCode ) -> returns an output array
# src: image, flipCode: 0 1 or -1
# 0: x-axis, 1: y-axis, -1: both axes

img_NZ_flipped_horz = cv2.flip(img_NZ_RGB, 1)
img_NZ_flipped_vert = cv2.flip(img_NZ_RGB, 0)
img_NZ_flipped_both = cv2.flip(img_NZ_RGB, -1)

plt.figure(figsize=(20, 10))

plt.subplot(131)
plt.imshow(img_NZ_flipped_horz)
plt.title('Flipped Sideways')

plt.subplot(132)
plt.imshow(img_NZ_flipped_vert)
plt.title('Flipped Upside-down')

plt.subplot(133)
plt.imshow(img_NZ_flipped_both)
plt.title('Flipped Both')

plt.show()