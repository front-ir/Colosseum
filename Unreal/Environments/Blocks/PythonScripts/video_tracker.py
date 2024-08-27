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

#OPENCV_OBJECT_TRACKERS = {
#	"csrt": cv2.legacy.TrackerCSRT_create,
#	"kcf": cv2.TrackerKCF_create,
#	"boosting": cv2.TrackerBoosting_create,
#	"mil": cv2.TrackerMIL_create,
#	"tld": cv2.TrackerTLD_create,
#	"medianflow": cv2.TrackerMedianFlow_create,
#	"mosse": cv2.TrackerMOSSE_create
#}

trackers = cv2.legacy.MultiTracker_create()
vs = cv2.VideoCapture('output_video.avi')

#init = vs.read()[1]
#detections = cascade.detectMultiScale(init, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10), maxSize=(55, 55))
#if len(detections) > 0:
#    (x, y, w, h) = detections[0]
#    #cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 1)
#    print(f"{x + w * 0.5} {y + h * 0.5}")
#(x, y, w, h) = detections[0]

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0
detected = False

margin = 10

while True:
	frame = imutils.resize(vs.read()[1], width=600)

	if not detected:
		detections = cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10), maxSize=(55, 55))
		if len(detections) > 0:
			(x, y, w, h) = detections[0]
			print(f"{x + w * 0.5} {y + h * 0.5}")
			print(x-margin, y-margin, w+margin, h+margin)
			tracker = cv2.legacy.TrackerCSRT_create()
			#trackers.add(tracker, frame, (x-15, y-15, w+15, h+15))
			trackers.add(tracker, frame, (x-margin, y-margin, w+margin, h+margin))
			detected = True

	# grab the current frame, then handle if we are using a
	# VideoStream or VideoCapture object
	# check to see if we have reached the end of the stream
	if frame is None:
		break
	# resize the frame (so we can process it faster)
	frame = imutils.resize(frame, width=600)
    # grab the updated bounding box coordinates (if any) for each
	# object that is being tracked
	(success, boxes) = trackers.update(frame)
	# loop over the bounding boxes and draw then on the frame
	for box in boxes:
		(x, y, w, h) = [int(v) for v in box]
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

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