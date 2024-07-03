import cv2
import sys

# Default camera source
s = 0
# If there are any command line arguments, set source to that
if len(sys.argv) > 1:
    s = sys.argv[1]

# Opencv captures the video from the source
source = cv2.VideoCapture(s)

# Creates a named window that is resizable
window_name = 'Camera Feed'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

# Until the Escape key is pressed
while cv2.waitKey(1) != 27:
    # Boolean if frame is read, frame is frame data
    has_frame, frame = source.read()
    if not has_frame:
        break
    cv2.imshow(window_name, frame) # Displays the frame in window
# Releases the video source and destroys the frame no matter what
source.release()
cv2.destroyWindow(window_name)