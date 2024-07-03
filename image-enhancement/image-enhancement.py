import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

from IPython.display import Image

img_NZ_BGR = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\New_Zealand_Coast.jpg', cv2.IMREAD_ANYCOLOR)
img_NZ_RGB = cv2.cvtColor(img_NZ_BGR, cv2.COLOR_BGR2RGB)

## BRIGHTNESS ##
# add or subtract the images ontop of one another to look brighter or darker
# due to changing the number of pixels on the screen

matrix = np.ones(img_NZ_RGB.shape, dtype='uint8') * 50

img_NZ_RGB_brighter = cv2.add(img_NZ_RGB, matrix)
img_NZ_RGB_darker = cv2.subtract(img_NZ_RGB, matrix)


# Show the images
plt.figure(figsize=[16, 4])

plt.subplot(132)
plt.imshow(img_NZ_RGB)
plt.title('Original')

plt.subplot(131)
plt.imshow(img_NZ_RGB_darker)
plt.title('Darker')

plt.subplot(133)
plt.imshow(img_NZ_RGB_brighter)
plt.title('Brighter')

## CONTRAST ##
# difference in the intensity values of the pixels in an image
# multiplication rather than addition / subtraction
matrix1 = np.ones(img_NZ_RGB.shape) * 0.8
matrix2 = np.ones(img_NZ_RGB.shape) * 1.2

img_NZ_RGB_lower  = np.uint8(cv2.multiply(np.float64(img_NZ_RGB), matrix1))
img_NZ_RGB_higher = np.uint8(np.clip(cv2.multiply(np.float64(img_NZ_RGB), matrix2), 0, 255))

# Show the images
plt.figure(figsize=[16, 4])

plt.subplot(132)
plt.imshow(img_NZ_RGB)
plt.title('Original')

plt.subplot(131)
plt.imshow(img_NZ_RGB_lower)
plt.title('Lower Contrast')

plt.subplot(133)
plt.imshow(img_NZ_RGB_higher)
plt.title('Higher Contrast')


## IMAGE THRESHOLDING ##
# useful for creating masks
# retval, dst = cv2.threshold( src, thresh, maxval, type[, dst] )
# src: input array, threshold value, max val for thresholding types, thresholding type
img_windows = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\building-windows.jpg', cv2.IMREAD_GRAYSCALE)
retval, img_thresh = cv2.threshold(img_windows, 100, 255, cv2.THRESH_TRIANGLE)

img_adaptive_thresh = cv2.adaptiveThreshold(img_windows,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
# Show the images
plt.figure(figsize=[16, 4])

plt.subplot(131)
plt.imshow(img_windows, cmap='gray')
plt.title('Original')

plt.subplot(132)
plt.imshow(img_thresh, cmap='gray')
plt.title('Thresholded')

plt.subplot(133)
plt.imshow(img_adaptive_thresh, cmap='gray')
plt.title('Adaptive Gaussian Thresholded')

print(img_thresh.shape)


## SHEET MUSIC READER ##
# Read the original image
img_original = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\Piano_Sheet_Music.png', cv2.IMREAD_GRAYSCALE)

# Perform global thresholding at different levels
retval1, img_thresh_global_1 = cv2.threshold(img_original, 50, 255, cv2.THRESH_BINARY)
retval2, img_thresh_global_2 = cv2.threshold(img_original, 130, 255, cv2.THRESH_BINARY)

# Perform adaptive thresholding
img_thresh_adaptive = cv2.adaptiveThreshold(img_original, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 7)

# Show the images
plt.figure(figsize=[18, 6])
plt.subplot(221)
plt.imshow(img_original, cmap='gray')
plt.title('Original')

plt.subplot(222)
plt.imshow(img_thresh_global_1, cmap='gray')
plt.title('Global: 50 Thresholded')

plt.subplot(223)
plt.imshow(img_thresh_global_2, cmap='gray')
plt.title('Global: 130 Thresholded')

plt.subplot(224)
plt.imshow(img_thresh_adaptive, cmap='gray')
plt.title('Adaptive Thresholded')

## BITWISE OPERATIONS ##
img_rec = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\rectangle.jpg', cv2.IMREAD_GRAYSCALE)
img_cir = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\circle.jpg', cv2.IMREAD_GRAYSCALE)

bitwise_AND = cv2.bitwise_and(img_rec, img_cir, mask=None)
bitwise_OR = cv2.bitwise_or(img_rec, img_cir, mask=None)
bitwise_XOR = cv2.bitwise_xor(img_rec, img_cir, mask=None)

plt.figure(figsize=[18, 10])

plt.subplot(221)
plt.imshow(img_rec, cmap='gray')
plt.title('Original Rectangle')

plt.subplot(222)
plt.imshow(img_cir, cmap='gray')
plt.title('Original Circle')

plt.subplot(223)
plt.imshow(bitwise_AND, cmap='gray')
plt.title('Bitwise AND Operator')

plt.subplot(224)
plt.imshow(bitwise_OR, cmap='gray')
plt.title('Bitwise OR Operator')

plt.subplot(325)
plt.imshow(bitwise_XOR, cmap='gray')
plt.title('Bitwise XOR Operator')

## LOGO MANIPULATION ##

# Read and convert logo image
img_logo_bgr = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\coca-cola-logo.png')
img_logo_rgb = cv2.cvtColor(img_logo_bgr, cv2.COLOR_BGR2RGB)

# Get logo width and height
logo_width = img_logo_rgb.shape[1]
logo_height = img_logo_rgb.shape[0]

# Read and convert background image
img_background_bgr = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\checkerboard_color.png')
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)

# Resize background image to match logo dimensions while maintaining aspect ratio
aspect_ratio = logo_width / img_background_rgb.shape[1]
dim = (logo_width, int(img_background_rgb.shape[0] * aspect_ratio))
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)

# Convert logo image to grayscale
img_logo_gray = cv2.cvtColor(img_logo_rgb, cv2.COLOR_RGB2GRAY)

# Apply global thresholding to create a binary mask of the logo
retval, img_mask = cv2.threshold(img_logo_gray, 127, 255, cv2.THRESH_BINARY)

# Create an inverse mask
img_mask_inv = cv2.bitwise_not(img_mask)

# Create colorful background "behind" the logo lettering
img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask)

# Isolate foreground (logo) using the inverse mask
img_foreground = cv2.bitwise_and(img_logo_rgb, img_logo_rgb, mask=img_mask_inv)

# Combine background and foreground to obtain the final result
result = cv2.add(img_background, img_foreground)

# Display the final result and save it
plt.imshow(result)
cv2.imwrite('masked_logo.png', result[:, :, ::-1])

plt.show()