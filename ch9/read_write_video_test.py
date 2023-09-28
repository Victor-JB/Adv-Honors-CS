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
        blurred = cv.GaussianBlur(frame, (5, 5), 0)

        T = mahotas.thresholding.otsu(blurred)

        # print("Otsu’s threshold: {}".format(T))
        thresh = frame.copy()
        thresh[thresh > T] = 255

        thresh[thresh < 255] = 0
        thresh = cv.bitwise_and(frame, thresh)
        cv.imshow('frame', frame)

    if cv.waitKey(1) == ord('q'):
        break
# Release everything if job is finished
cap.release()
cv.destroyAllWindows()
