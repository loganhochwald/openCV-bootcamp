# %%
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

## READING IMAGES ##

# Syntax: cv2.imread( <filename as string>, <flags that denote colour> )
# Flags: 1 is colour ignoring any transparency, 0 is greyscale, and -1 is unchanged
# Returns: pixel values in 8-bit in a 2D array
img = cv2.imread(r'C:\openCV-bootcamp\getting-started\image.jpg')
print(img)

## IMAGE ATTRIBUTES ##
# Size:
print('Height and Width: ', img.shape[:2])
# Data-type:
print('Data type: ', img.dtype)

## DISPLAY WITH MATPLOTLIB ##
# OpenCV uses BGR format but Matplotlib expects RGB
img_in_rgb = img[:, :, ::-1]
# Default Colour
plt.imshow(img_in_rgb)


## COLOUR CHANNELS ##
# You can split into the three colour channels then plot / display using Matplotlib
# Split
img_NZ = cv2.imread(r'C:\openCV-bootcamp\getting-started\New_Zealand_Lake.jpg')
b, g, r = cv2.split(img_NZ)

# Create figure that will contain the plotted channels, size in inches
plt.figure(figsize=[40, 8])

# Subplots of Red, Green, and Blue
plt.subplot(141) # 1 row, 4 columns, 1st subplot
plt.imshow(r, cmap='gray') # greyscale, remove cmap for reds in image
plt.title('Red')

plt.subplot(142) # 1 row, 4 columns, 2nd subplot
plt.imshow(g, cmap='gray')
plt.title('Green')

plt.subplot(143) # 1 row, 4 columns, 3rd subplot
plt.imshow(b, cmap='gray')
plt.title('Blue')

# Merge the individual channels into a BGR image again
imgMerged = cv2.merge((b, g, r))

# Show the merged image
plt.subplot(144) # 1 row, 4 columns, 4th subplot
plt.imshow(imgMerged[:, :, ::-1])
plt.title('Merged')

## RGB to BGR Shortcut ##
img_NZ_RGB = cv2.cvtColor(img_NZ, cv2.COLOR_BGR2RGB)
plt.imshow(img_NZ_RGB)

## SAVE IMAGE ##
# cv2.imwrite( filename, img[, params] )
cv2.imwrite('NZ_Merge_Save.png', imgMerged)

plt.show()