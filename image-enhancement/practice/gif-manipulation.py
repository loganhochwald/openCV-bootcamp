from PIL import Image, ImageSequence
import cv2
import numpy as np

# Gir gif, must use Pillow because cv2.imread() can only handle photos, not gifs
with Image.open(r'C:\openCV-bootcamp\image-enhancement\practice\gir-dancing.gif') as gif:
    frames = []

    # Extract each frame with Pillow Iterator: https://pillow.readthedocs.io/en/stable/reference/ImageSequence.html
    for frame in ImageSequence.Iterator(gif):
        frames.append(frame.copy())

    processed_frames = []

    for frame in frames:
        # Convert PIL image to OpenCV format (BGR)
        frame_BGR = np.array(frame)

        # Skip frames that are not BGR (3 channels)
        # https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
        if frame_BGR.ndim != 3 or frame_BGR.shape[2] != 3:
            continue

        # Read and convert background image to RGB
        img_background_BGR = cv2.imread(r'C:\openCV-bootcamp\image-enhancement\practice\green.jpg')
        img_background_RGB = cv2.cvtColor(img_background_BGR, cv2.COLOR_BGR2RGB)

        # Ratio of gif frame to image background
        aspect_ratio = frame_BGR.shape[1] / img_background_RGB.shape[1]

        # Dimensions of the background image
        dim = (frame_BGR.shape[1], int(img_background_RGB.shape[0] * aspect_ratio))

        # Resize with INTER_AREA for shrinking images
        img_background_RGB = cv2.resize(img_background_RGB, dim, interpolation=cv2.INTER_AREA)

        # It misses a few frames maybe, code doesn't work without this?
        img_background_RGB = cv2.resize(img_background_RGB, (frame_BGR.shape[1], frame_BGR.shape[0]))

        # Create masks for background and foreground

        # HSV green lower and upper boundaries: https://www.geeksforgeeks.org/color-identification-in-images-using-python-opencv/
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([90, 255, 255])

        # Convert BGR to HSV and create a mask and inverse of mask for green pixels
        mask = cv2.inRange(cv2.cvtColor(frame_BGR, cv2.COLOR_BGR2HSV), lower_green, upper_green)
        mask_inv = cv2.bitwise_not(mask)

        # Apply masks
        img_background = cv2.bitwise_and(img_background_RGB, img_background_RGB, mask=mask)
        img_foreground = cv2.bitwise_and(frame_BGR, frame_BGR, mask=mask_inv)

        # Combine the background and foreground
        result = cv2.add(img_background, img_foreground)

        # Convert back to RGB so it can be cropped
        result_pil = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        # Crop by removing 50 pixels from each side
        cropped_result = result_pil.crop((50, 50, result_pil.width - 50, result_pil.height - 50))

        # Append the cropped frame to processed_frames
        processed_frames.append(cropped_result)

# Save the processed frames as a new GIF
processed_frames[0].save(r'C:\openCV-bootcamp\image-enhancement\practice\processed_gir_dancing_green.gif', save_all=True, append_images=processed_frames[1:], loop=0, duration=gif.info['duration'])

# Gif source: https://i.pinimg.com/originals/96/52/3f/96523f2c3d53787e051d596041b52074.gif
# Background source: https://st2.depositphotos.com/3224051/9271/i/950/depositphotos_92710762-stock-photo-vivid-green-coloured-splatter-cool.jpg