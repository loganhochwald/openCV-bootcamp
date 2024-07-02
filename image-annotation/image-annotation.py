import os
import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from zipfile import ZipFile
from urllib.request import urlretrieve

matplotlib.rcParams['figure.figsize'] = (9.0, 9.0)

image = cv2.imread(r'C:\openCV-bootcamp\image-annotation\Apollo_11_Launch.jpg', cv2.IMREAD_COLOR)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
print(image.shape)

## DRAWING A LINE ##

# cv2.line(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
# img: the image, pt1: first point (x,y) of the line segment, pt2: second point of line, color: colour of the drawn line
# optional are thickness and lineType

image_with_line = image.copy() # will draw line across pic
cv2.line(image_with_line, (0,0), (image.shape[1], image.shape[0]), (20, 20, 180), thickness=4)
plt.imshow(cv2.cvtColor(image_with_line, cv2.COLOR_BGR2RGB))

## DRAWING A CIRCLE ##
# cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
# thickness: positive is thickness of circle outline, negative will fill circle

image_with_circle = image.copy()
cv2.circle(image_with_circle, (((int)(image.shape[1] / 2)), ((int)(image.shape[0] / 2))), 300, (20, 20, 180), thickness=20)
cv2.circle(image_with_circle, (((int)(image.shape[1] / 2)), ((int)(image.shape[0] / 2))), 100, (20, 20, 180), thickness=-2)
plt.imshow(cv2.cvtColor(image_with_circle, cv2.COLOR_BGR2RGB))


## DRAWING A RECTANGLE ##
# cv2.rectangle(img, pt1, pt2, color[, thickness[, lineType[, shift]]])
# pt1 is top left vertex and pt2 is bottom right vertex
image_with_rect = image.copy() # will frame it
cv2.rectangle(image_with_rect, (0,0), (image.shape[1], image.shape[0]), (20, 20, 180), thickness=50)
plt.imshow(cv2.cvtColor(image_with_rect, cv2.COLOR_BGR2RGB))


## ADDING TEXT ##
# img = cv2.putText(img, text, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
image_with_txt = image.copy()
text = 'Billie Jean is not my lover'
# text will be upside down bc negative number
cv2.putText(image_with_txt, text, (((int)(image.shape[1])), ((int)(image.shape[0] / 2))), cv2.FONT_HERSHEY_TRIPLEX, -2, (0, 255, 0), thickness=3)
#plt.imshow(cv2.cvtColor(image_with_txt, cv2.COLOR_BGR2RGB))


plt.show()