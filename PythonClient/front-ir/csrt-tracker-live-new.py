import cv2
import numpy as np
import airsim
import time
import os

script_path = os.path.dirname(os.path.realpath(__file__))

def load_classifier(cascade_path):
    # Load the Haar cascade file
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError("Cannot load cascade classifier from file: " + cascade_path)
    return cascade

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# Path to the Haar cascade file
cascade40 = load_classifier('U:/front-ir/Colosseum/Unreal/Environments/Blocks/PythonScripts/RAB_HAAR/cascade.xml')
cascade100 = load_classifier(f'{script_path}/screenshots/cascade/cascade.xml')

camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(1.5, 0, 0))
client.simSetCameraPose("0", camera_pose)

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

image_type = 0
pixels_as_float = False
compress = False

tracker = None
detected = False
margin = 10
last_cascade_time = time.time()
inv = False
thresh = cv2.THRESH_BINARY
if inv: thresh = cv2.THRESH_BINARY_INV
wait_time = 0
gw, gh = 50, 50


while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    response = client.simGetImages([airsim.ImageRequest("0", image_type, pixels_as_float, compress)])[0]
    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    frame = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    _, thresholded = cv2.threshold(frame, 200, 255, thresh)

    #state = client.getMultirotorState()
    #altitude = -state.kinematics_estimated.position.z_val
    #print(altitude)


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

    if time.time() - wait_time > 2:
        wait_time = 0

    if not detected and wait_time == 0:
        #if altitude < 220:
        #    detections = cascade40.detectMultiScale(thresholded, scaleFactor=1.1, minNeighbors=25, minSize=(10, 10), maxSize=(50, 50))
        #else:
        detections = cascade100.detectMultiScale(thresholded, scaleFactor=1.1, minNeighbors=25, minSize=(10, 10), maxSize=(455, 455))
        if len(detections) > 0:
            (x, y, w, h) = detections[0]
            #gw, gh = w, h
            #print(w, h)
            #print(f"{x + w * 0.5} {y + h * 0.5}")
            #print(x-margin, y-margin, w+margin, h+margin)
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x-margin, y-margin, w+margin, h+margin))
            detected = True

    if detected:
        (success, box) = tracker.update(thresholded)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cropped_tracker = cv2.getRectSubPix(thresholded, (w, h), (x + w / 2, y + h / 2))

            circles = cv2.HoughCircles(cropped_tracker, cv2.HOUGH_GRADIENT, dp=1.2, minDist=100,
                                       param1=100, param2=55, minRadius=15, maxRadius=100)
            if circles is not None:
                circles = np.uint16(np.around(circles))  # Round and convert to integer
                for circle in circles[0, :]:
                    #Draw the outer circle
                    cv2.circle(frame, (circle[0]+x, circle[1]+y), circle[2], (0, 0, 0), 2)
                    #Draw the center of the circle
                    cv2.circle(frame, (circle[0]+x, circle[1]+y), 2, (0, 0, 255), 3)
                print("Round shape detected. Stopping tracking for 2 sec")
                tracker = None
                detected = False
                wait_time = time.time()
                continue

            #print(time.time() - last_cascade_time)
            #if time.time() - last_cascade_time > 2:
            #    print("im here")
            #    if altitude < 220:
            #        print("small box")
            #        detections = cascade40.detectMultiScale(thresholded, scaleFactor=1.1, minNeighbors=25, minSize=(10, 10), maxSize=(50, 50))
            #    else:
            #        print("big box")
            #        #detections = cascade100.detectMultiScale(thresholded, scaleFactor=1.2, minNeighbors=25, minSize=(10, 10), maxSize=(w+10, h+10))
            #    if len(detections) == 0:
            #        print("No shahed in box")
            #            # tracker = None
            #            # detected = False
            #    last_cascade_time = time.time()

            # contours, _ = cv2.findContours(cropped_tracker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # if contours is None:
                # print("Nothing is inside the tracker. Refreshing")
                # tracker = False
                # detected = None
                # continue
            # else:
                # offset_contours = [contour + np.array([x, y], dtype=np.int32) for contour in contours]
                # cv2.drawContours(frame, offset_contours, -1, (0, 255, 0), 1)
        #else:
        #    print("Tracker lost")
        #    tracker = None
        #    detected = None

    cv2.putText(frame,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
    cv2.imshow("Grey", frame)
    cv2.imshow("thresholded", thresholded)
