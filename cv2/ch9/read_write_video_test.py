import numpy as np
import cv2 as cv
import mahotas

cap = cv.VideoCapture('krunker_test.mov')
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    else:

        frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame = cv.GaussianBlur(frame, (5, 5), 0)

        canny = cv.Canny(frame, 30, 150)
        cv.imshow('frame', canny)

    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
cv.destroyAllWindows()
