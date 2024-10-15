import cv2
import numpy as np
import airsim
import time
import os
import math
import ultralytics

script_path = os.path.dirname(p=os.path.realpath(__file__))
textSize, _ = cv2.getTextSize(text="FPS", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=2)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

tracker_size=0

wait_time = 0
upd_time = 0


tracker = None
useTracker = True
detected = False

DEBUG = True

#18 avg
#5 pretty good

model = ultralytics.YOLO(r"U:\front-ir\Colosseum\runs\detect\train18\weights\best.pt")

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

camera_pose = airsim.Pose(position_val=airsim.Vector3r(0, 0, 0), orientation_val=airsim.to_quaternion(1.5, 0, 0))
client.simSetCameraPose(camera_name="0", pose=camera_pose)
upd_time = time.time()

while True:
    response = client.simGetImages(requests=[
        airsim.ImageRequest(camera_name="0", image_type=airsim.ImageType.Scene, pixels_as_float=False, compress=False)])[0]
    image = np.frombuffer(buffer=response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    grey = cv2.cvtColor(src=image, code=cv2.COLOR_RGB2GRAY)
    norm = cv2.cvtColor(src=grey, code=cv2.COLOR_GRAY2RGB)
    _, thresholded = cv2.threshold(src=grey, thresh=200, maxval=255, type=cv2.THRESH_BINARY)

    if time.time() - wait_time > 1:
        wait_time = 0

    if wait_time == 0:
        if not detected:
            results = model.predict(source=norm)
            if len(results) == 1:
                if len(results[0].boxes.data) == 1:
                    box = results[0].boxes[0]
                    detected = True
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    yolo_coords = (int(x1), int(y1), int(x2), int(y2))
                    if DEBUG: print(f"Yolo1 center: {x1 + ((x2 - x1)*0.5)} {y1 + ((y2 - y1)*0.5)}")
                else: 
                    detected = False
                    yolo_coords = (None, None, None, None)

            if detected and useTracker:
                tracker = None
                tracker = cv2.legacy.TrackerCSRT_create()
                tracker.init(thresholded, (x1,y1,x2-x1, y2-y1))

        if detected:
            if useTracker:
                (success, box) = tracker.update(thresholded)
                if success:
                    tracker_coords = [int(v) for v in box]
                    tracker_center = (tracker_coords[0]+tracker_coords[2]*0.5, tracker_coords[1]+tracker_coords[3]*0.5)
                    cv2.rectangle(grey, (tracker_coords[0], tracker_coords[1]),
                                (tracker_coords[0] + tracker_coords[2], tracker_coords[1] + tracker_coords[3]),
                                (0, 0, 255), 1)
                    
                    if DEBUG: print("Tracker center:", *tracker_center)

                    if time.time() - upd_time > 1:
                        results = model.predict(source=norm)
                        if len(results) == 1:
                            if len(results[0].boxes.data) == 1:
                                box = results[0].boxes[0]
                                detected = True
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                yolo_coords = (int(x1), int(y1), int(x2), int(y2))

                                if not (x1 <= tracker_center[0] <= x2) or not (y1 <= tracker_center[1] <= y2):
                                    if DEBUG: print(f"Tracker works outside shahed bounding box found by YOLO. Resetting!")
                                    detected = False
                                    tracker = None

                                if DEBUG: print(f"Yolo2 center: {x1 + ((x2 - x1)*0.5)} {y1 + ((y2 - y1)*0.5)}")
                            else: 
                                detected = False
                                yolo_coords = (None, None, None, None)
                                wait_time = time.time()
                                if DEBUG: print(f"Yolo couldn't find shahed in frame. Resetting in 2 sec!")
                        upd_time = time.time()
                        
                    useTracker = x2 - x1 < 200
                    if not useTracker: tracker = None
                else:
                    print("Tracker lost")
                    tracker = None
                    detected = False
            else:
                results = model.predict(source=norm)
                for result in results:
                    if len(result.boxes.data) == 1:
                        x1, y1, x2, y2 = result.boxes[0].xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cv2.rectangle(grey, (x1, y1), (x2, y2), (0, 255, 0), 1)

                        if DEBUG: print("Yolo2 center:", x1, y1, x2, y2)

    cv2.putText(img=grey, text=f"{fps}", org=textOrg, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,255), thickness=2)
    cv2.imshow("Grey", grey)
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
