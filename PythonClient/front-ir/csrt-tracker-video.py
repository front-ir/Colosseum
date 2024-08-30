import imutils
import time
import cv2
import numpy as np
#import airsim
import time
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import os

def load_classifier(cascade_path):
    # Load the Haar cascade file
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError("Cannot load cascade classifier from file: " + cascade_path)
    return cascade
script_path = os.path.dirname(os.path.realpath(__file__))

cascade_type = "RAB_HAAR"
cascade = load_classifier(f'{script_path}/{cascade_type}/cascade.xml')

tracker = None
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

margin = 10

while True:
    frame = cv2.cvtColor(imutils.resize(vs.read()[1], width=600), cv2.COLOR_BGR2GRAY)
    if frame is None:
        break
    _, thresholded = cv2.threshold(frame, 200, 255, thresh)

    if not detected:
        detections = cascade.detectMultiScale(thresholded, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10), maxSize=(55, 55))
        if len(detections) > 0:
            (x, y, w, h) = detections[0]
            print(f"{x + w * 0.5} {y + h * 0.5}")
            print(x-margin, y-margin, w+margin, h+margin)
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x-margin, y-margin, w+margin, h+margin))
            detected = True
        else:
            continue

    if detected:
        (success, box) = tracker.update(thresholded)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_tracker = cv2.getRectSubPix(thresholded, (w, h), (x + w / 2, y + h / 2))

            circles = cv2.HoughCircles(cropped_tracker, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                       param1=100, param2=16, minRadius=10, maxRadius=100)
            if circles is not None:
                print("Round shape detected. Refreshing")
                tracker = None
                detected = None
                continue

            contours, _ = cv2.findContours(cropped_tracker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours is None:
                print("Nothing is inside the tracker. Refreshing")
                tracker = False
                detected = None
                continue
            else:
                offset_contours = [contour + np.array([x, y], dtype=np.int32) for contour in contours]
                cv2.drawContours(frame, offset_contours, -1, (0, 255, 0), 1)
        else:
            print("Tracker lost")
            tracker = None
            detected = None

    cv2.imshow("thresholded", thresholded)


    # show the output frame
    cv2.putText(frame,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,255,255),thickness)
    cv2.imshow("Frame", frame)
    frameCount = frameCount  + 1
    endTime = time.time()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        box = cv2.selectROI("Frame", frame, fromCenter=False,
            showCrosshair=True)

        print(box)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker

    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
# if we are using a webcam, release the pointer
vs.release()
# close all windows
cv2.destroyAllWindows()