import cv2
import numpy as np
import airsim
import time
import os
import math

script_path = os.path.dirname(p=os.path.realpath(__file__))
textSize, _ = cv2.getTextSize(text="FPS", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

cascade_size=50

wait_time = 0

tracker = None
detected = False

DEBUG = False

cascadePath = os.path.join(script_path, "RAB_HAAR", "cascade.xml")
cascade = cv2.CascadeClassifier(filename=cascadePath)
if cascade.empty():
    raise IOError(f"Cannot load cascade classifier from file: {cascadePath}")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

camera_pose = airsim.Pose(position_val=airsim.Vector3r(0, 0, 0), orientation_val=airsim.to_quaternion(1.5, 0, 0))
client.simSetCameraPose(camera_name="0", pose=camera_pose)

while True:
    response = client.simGetImages(requests=[
        airsim.ImageRequest(camera_name="0", image_type=airsim.ImageType.Scene, pixels_as_float=False, compress=False)])[0]
    image = np.frombuffer(buffer=response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    grey = cv2.cvtColor(src=image, code=cv2.COLOR_RGBA2GRAY)
    _, thresholded = cv2.threshold(src=grey, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    if time.time() - wait_time > 1:
        wait_time = 0

    if wait_time == 0:
        if not detected:
            detections = cascade.detectMultiScale(image=thresholded, scaleFactor=1.05, minNeighbors=55, minSize=(int(cascade_size*0.25), int(cascade_size*0.25)), maxSize=(cascade_size, cascade_size))
            if len(detections) == 1:
                detected = True
                cascade_coords = detections[0]
                if DEBUG: print(f"Cascade center: {detections[0][0]+detections[0][2]*0.5} {detections[0][1]+detections[0][3]*0.5}")
            else: 
                detected = False
                cascade_coords = (None, None, None, None)

            if detected:
                tracker = None
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(thresholded, cascade_coords)

        if detected:
            (success, box) = tracker.update(thresholded)
            if success:
                tracker_coords = [int(v) for v in box]
                tracker_center = (tracker_coords[0]+tracker_coords[2]*0.5, tracker_coords[1]+tracker_coords[3]*0.5)

                if DEBUG:
                    print("Tracker center:", *tracker_center)
                    cv2.rectangle(grey, (tracker_coords[0], tracker_coords[1]),
                                  (tracker_coords[0] + tracker_coords[2], tracker_coords[1] + tracker_coords[3]),
                                  (0, 0, 255), 1)

                cropped_tracker = cv2.getRectSubPix(image=thresholded, patchSize=(tracker_coords[2], tracker_coords[3]),
                                                    center=tracker_center)
                circles = cv2.HoughCircles(image=cropped_tracker, method=cv2.HOUGH_GRADIENT, dp=1.1, minDist=100,
                                           param1=110, param2=25, minRadius=10, maxRadius=100)
                if circles is not None:
                    if DEBUG:
                        circle = np.uint16(np.around(circles))[0, :][0]
                        circle_center = (circle[0]+tracker_coords[0], circle[1]+tracker_coords[1])
                        circle_radius = circle[2]
                        print("Circle center and r", *circle_center, circle_radius)
                        print("Round shape detected inside the tracker. Stopping tracking for 2 sec")
                    tracker = None
                    detected = False
                    wait_time = time.time()
                else:
                    contours, _ = cv2.findContours(image=cropped_tracker, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                    if contours is None:
                        tracker = None
                        detected = False
                        wait_time = time.time()
                        if DEBUG: print("Nothing is inside the tracker. Refreshing")
                    elif len(contours) > 1:
                        detections = cascade.detectMultiScale(image=thresholded, scaleFactor=1.05, minNeighbors=55, minSize=(int(cascade_size*0.25), int(cascade_size*0.25)), maxSize=(cascade_size, cascade_size))
                        if len(detections) == 1:
                            detected = True
                            cascade_coords = detections[0]
                            if DEBUG: print(f"Cascade center: {detections[0][0]+detections[0][2]*0.5} {detections[0][1]+detections[0][3]*0.5}")
                        else: 
                            detected = False
                            cascade_coords = (None, None, None, None)

                        if not detected:
                            tracker = None
                            wait_time = time.time()
                            if DEBUG:print("Multiple contours detected in box. Refresh")
                    elif len(contours) == 1:
                        contour = contours[0] + np.array([tracker_coords[0], tracker_coords[1]], dtype=np.int16)

                        contour_coords = [int(x) for x in cv2.boundingRect(array=contour)]
                        contour_coords[0] -= 5
                        contour_coords[1] -= 5
                        contour_coords[2] += 10
                        contour_coords[3] += 10

                        if DEBUG:
                            contour_center = (contour_coords[0]+contour_coords[2]*0.5, contour_coords[1]+contour_coords[3]*0.5)
                            print("Contour center:", *contour_center)
                            cv2.rectangle(grey, (contour_coords[0], contour_coords[1]),
                                          (contour_coords[0] + contour_coords[2], contour_coords[1] + contour_coords[3]),
                                          (255, 255, 255), 1)

                        if math.fabs(contour_coords[2] - tracker_coords[2]) > 1 or math.fabs(contour_coords[3] - tracker_coords[3]) > 1:
                            cascade_size=max(contour_coords[2], 50)
                            tracker = None
                            tracker = cv2.legacy.TrackerCSRT_create()
                            tracker.init(thresholded, contour_coords)
                        cv2.drawContours(image=grey, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=1)
            else:
                print("Tracker lost")
                tracker = None
                detected = False

    cv2.putText(img=grey, text=f"{fps}", org=textOrg, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,255), thickness=2)
    cv2.imshow("Grey", grey)
    cv2.imshow("thresholded", thresholded)
    frameCount += 1
    endTime = time.time()
    diff = endTime - startTime
    if (diff > 1):
        fps = frameCount
        frameCount = 0
        startTime = endTime
    key = cv2.waitKey(1) & 0xFF
    if (key == 27 or key == ord('q') or key == ord('x')):
        break
    if (key == ord('r')):
        print("Reset tracker")
        detected = False
        tracker = None
