import imutils
import time
import cv2
import numpy as np
#import airsim
import time
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import os
script_path = os.path.dirname(os.path.realpath(__file__))

vs = cv2.VideoCapture(f'{script_path}/output_video.avi')

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0
detected = False

inv = False
thresh = cv2.THRESH_BINARY
if inv: thresh = cv2.THRESH_BINARY_INV

while True:

    frame = vs.read()[1]
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    _, thresholded = cv2.threshold(grey, 200, 255, thresh)
    #contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circles = cv2.HoughCircles(thresholded, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                               param1=100, param2=16, minRadius=10, maxRadius=100)

    # If some circles are detected, process them
    if circles is not None:
        circles = np.uint16(np.around(circles))  # Round and convert to integer
        for circle in circles[0, :]:
            # Draw the outer circle
            cv2.circle(thresholded, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)
            # Draw the center of the circle
            cv2.circle(thresholded, (circle[0], circle[1]), 2, (0, 0, 255), 3)

            center_x, center_y, radius = circle[0], circle[1], circle[2]

        # Calculate the size of the square bounding box
            size = int(2 * radius)  # The box size is twice the radius

        # Ensure the size is odd to keep the center precise (optional)
            if size % 2 == 0:
                size += 1

        # Crop the square region around the circle
            cropped_image = cv2.getRectSubPix(thresholded, (size, size), (center_x, center_y))
            _, thresholdedi = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY_INV)

            mask = 255 * np.ones(thresholdedi.shape, thresholdedi.dtype)

# Center of the location where the cropped image will be placed
            center_location = (center_x, center_y)
            cv2.imshow("thresholdedi", thresholdedi)
            #cv2.imshow("blended_image", blended_image)

            #blended_image = cv2.seamlessClone(thresholdedi, thresholded, mask, center_location, cv2.NORMAL_CLONE)

            cv2.imshow("thresholdedi", thresholdedi)
            #cv2.imshow("blended_image", blended_image)





    cv2.putText(grey,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,255,255),thickness)
    cv2.putText(thresholded,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,255,255),thickness)
    cv2.imshow("Frame1", thresholded)
    cv2.imshow("Frame2", grey)
    frameCount = frameCount  + 1
    endTime = time.time()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
# if we are using a webcam, release the pointer
vs.release()
# close all windows
cv2.destroyAllWindows()