import cv2
import numpy as np
import airsim
import time
import os
import math

script_path = os.path.dirname(os.path.realpath(__file__))

# def load_classifier(cascade_path):
#    Load the Haar cascade file
    # cascade = cv2.CascadeClassifier(cascade_path)
    # if cascade.empty():
        # raise IOError("Cannot load cascade classifier from file: " + cascade_path)
    # return cascade

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

# Path to the Haar cascade file
#cascade = load_classifier('U:/front-ir/Colosseum/Unreal/Environments/Blocks/PythonScripts/RAB_HAAR/cascade.xml')

camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(1.5, 0, 0))
client.simSetCameraPose("0", camera_pose)

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

image_type = airsim.ImageType.Infrared
pixels_as_float = False
compress = False

#tracker = None
#detected = False
#margin = 10
#last_cascade_time = time.time()
#inv = False
thresh = cv2.THRESH_BINARY
#if inv: thresh = cv2.THRESH_BINARY_INV
#wait_time = 0

#def create_tracker(frame, x,y,w,h):
#    tracker = cv2.legacy.TrackerCSRT_create()
#    tracker.init(frame, (x, y, w, h))
#    return tracker

while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    response = client.simGetImages([
            airsim.ImageRequest("0", image_type, pixels_as_float, compress)])[0]
    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    grey = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    _, thresholded = cv2.threshold(grey, 210, 255, thresh)

    #if time.time() - wait_time > 2:
    #    wait_time = 0
#
    #if not detected and wait_time == 0:
    #    detections = cascade.detectMultiScale(thresholded, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10), maxSize=(55, 55))
    #    if len(detections) > 0:
    #        (dx, dy, dw, dh) = detections[0]
    #        #print(f"{dx + dw * 0.5} {dy + dh * 0.5}")
    #        #print(dx-margin, dy-margin, dw+margin, dh+margin)
    #        tracker = create_tracker(thresholded, dx-margin, dy-margin, dw+margin, dh+margin)
    #        #tracker = cv2.legacy.TrackerCSRT_create()
    #        #tracker.init(frame, (dx-margin, dy-margin, dw+margin, dh+margin))
    #        detected = True
#
    #if detected:
    #    (success, box) = tracker.update(thresholded)
    #    if success:
    #        (tx, ty, tw, th) = [int(v) for v in box]
    #        tracker_x = tx
    #        tracker_y = ty
    #        tracker_w = tw
    #        tracker_h = th
#
    #        cropped_tracker = cv2.getRectSubPix(thresholded, (tw, th), (tx + tw / 2, ty + th / 2))
#
    #        circles = cv2.HoughCircles(cropped_tracker, cv2.HOUGH_GRADIENT, dp=1.1, minDist=100,
    #                                   param1=110, param2=25, minRadius=10, maxRadius=100)
    #        if circles is not None:
    #            circles = np.uint16(np.around(circles))  # Round and convert to integer
    #            for circle in circles[0, :]:
    #                #Draw the outer circle
    #                #cv2.circle(frame, (circle[0]+tx, circle[1]+ty), circle[2], (0, 0, 0), 2)
    #                #Draw the center of the circle
    #                #cv2.circle(frame, (circle[0]+tx, circle[1]+ty), 2, (0, 0, 0), 2)
#
    #                #print(f"Circle center - {circle[0]+tx}")
    #                #print(f"Tracker center - {tx + (tw/2)}")
#
    #                if (circle[0]+tx) - (tx + (tw/2)) < tw/4:
    #                    print("Round shape detected in center of the tracker. Stopping tracking for 2 sec")
    #                    tracker = None
    #                    detected = False
    #                    wait_time = time.time()
    #                    continue
    #        if not detected and tracker == None:
    #            continue
    #            #continue
#
    #        # current_time = time.time()
    #        # if current_time - last_cascade_time > 2:
    #            # detections = cascade.detectMultiScale(cropped_tracker, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10), maxSize=(55, 55))
    #            # if len(detections) == 0:
    #                # print("No shahed in box")
    #                # tracker = None
    #                # detected = False
    #            # last_cascade_time = current_time
#
    #        contours, _ = cv2.findContours(cropped_tracker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #        if contours is None:
    #            print("Nothing is inside the tracker. Refreshing")
    #            tracker = None
    #            detected = False
    #        elif len(contours) > 1:
    #            print("Multiple contours detected in box. Refresh")
    #            tracker = None
    #            detected = False
    #        elif len(contours) == 1:
    #            contour = contours[0]
    #            #is_closed = np.array_equal(contour[0], contour[-1])
    #            #if not is_closed:
    #            #    print("Contour is not closed (has gaps), removing tracker")
    #            #    tracker = None
    #            #    detected = False
#
    #            (cx, cy, cw, ch) = cv2.boundingRect(contour)
    #            cx -= margin
    #            cy -= margin
    #            cw += 2*margin
    #            ch += 2*margin
    #            #print(cx, cy, cw, ch)
    #            #cv2.rectangle(frame, (cx+tx, cy+ty), (cx+tx + cw, cy+ty + ch), (0, 255, 0), 2)
#
    #            if math.fabs(cw - tw) > math.log(tw)*2:
    #                tracker_x = cx+tx
    #                tracker_y = cy+ty
    #                tracker_w = cw
    #                tracker_h = ch
    #                tracker = create_tracker(thresholded, tracker_x, tracker_y, tracker_w, tracker_h)
#
    #            # if cw - tw >= margin/2 :
    #                # print(f"Enlarging tracker - {i}")
    #                # tracker = None
##
    #                # tracker_x = cx+tx
    #                # tracker_y = cy+ty
    #                # tracker_w = cw
    #                # tracker_h = ch
##
    #                # tracker = create_tracker(thresholded, cx+tx, cy+ty, cw, ch)
    #            # elif tw - cw >= margin/2:
    #                # print(f"Making tracker smaller - {i}")
    #                # tracker = None
    #                # tracker = create_tracker(thresholded, cx+tx, cy+ty, cw, ch)
#
    #            cv2.rectangle(frame, (tracker_x, tracker_y), (tracker_x + tracker_w, tracker_y + tracker_h), (0, 255, 0), 2)
    #            offset_contours = [contour + np.array([tracker_x, tracker_y], dtype=np.int32) for contour in contours]
    #            cv2.drawContours(frame, offset_contours, -1, (0, 255, 0), 1)
    #    else:
    #        print("Tracker lost")
    #        tracker = None
    #        detected = None

    cv2.putText(grey,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)
    #cv2.putText(img,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)

    cv2.imshow("Grey", grey)
    cv2.imshow("image", image)
    cv2.imshow("thresholded", thresholded)
    #cv2.imshow("thresholded", thresholded)
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
    #if (key == ord('r')):
    #    print("Reset tracker")
    #    detected = False
    #    tracker = None
