import cv2
import matplotlib.pyplot as plt

## LOGO MANIPULATION ##

# Read and convert logo image
img_logo_bgr = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\practice\panda.png')
img_logo_rgb = cv2.cvtColor(img_logo_bgr, cv2.COLOR_BGR2RGB)

# Get logo width and height
logo_width = img_logo_rgb.shape[1]
logo_height = img_logo_rgb.shape[0]

# Read and convert background image
img_background_bgr = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\practice\green.jpg')
img_background_rgb = cv2.cvtColor(img_background_bgr, cv2.COLOR_BGR2RGB)

# Resize background image to match logo dimensions while maintaining aspect ratio
aspect_ratio = logo_width / img_background_rgb.shape[1]
dim = (logo_width, int(img_background_rgb.shape[0] * aspect_ratio))
img_background_rgb = cv2.resize(img_background_rgb, dim, interpolation=cv2.INTER_AREA)

# Convert panda image to grayscale
img_logo_gray = cv2.cvtColor(img_logo_rgb, cv2.COLOR_RGB2GRAY)

# Apply global thresholding to create a binary mask of the panda
retval, img_mask = cv2.threshold(img_logo_gray, 127, 255, cv2.THRESH_BINARY)

# Create an inverse mask
img_mask_inv = cv2.bitwise_not(img_mask)

# Create colorful background "behind" the panda
img_background = cv2.bitwise_and(img_background_rgb, img_background_rgb, mask=img_mask_inv)

# Isolate foreground (panda) using the inverse mask
img_foreground = cv2.bitwise_and(img_logo_rgb, img_logo_rgb, mask=img_mask)

# Combine background and foreground to obtain the final result
result = cv2.add(img_background, img_foreground)

# Display the final result and save it
plt.imshow(result)
cv2.imwrite('cooler_panda.png', result[:, :, ::-1])

plt.show()

# Panda source: https://i.pinimg.com/originals/41/05/d5/4105d5622c8a68d05ab861702b29d487.jpg
# Background source: https://st2.depositphotos.com/3224051/9271/i/950/depositphotos_92710762-stock-photo-vivid-green-coloured-splatter-cool.jpg