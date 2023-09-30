import numpy as np

import argparse
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "Path to the image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
cv2.imshow("Original", image)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
h,s,v = cv2.split(hsv)

mask = cv2.inRange(hsv, (0,0,0), (180, 50, 130))
dst1 = cv2.bitwise_and(image, image, mask=mask)

th, threshed = cv2.threshold(v, 100, 200, cv2.THRESH_BINARY_INV)
dst2 = cv2.bitwise_and(image, image, mask=threshed)

th, threshed2 = cv2.threshold(s, 30, 255, cv2.THRESH_BINARY_INV)
dst3 = cv2.bitwise_and(image, image, mask=threshed2)

cv2.imshow("Hue", dst1)
cv2.imshow("Saturation", dst2)
cv2.imshow("Value", dst3)

#cv2.imwrite("dst1.png", dst1)
#cv2.imwrite("dst2.png", dst2)
#cv2.imwrite("dst3.png", dst3)

cv2.waitKey(0)
