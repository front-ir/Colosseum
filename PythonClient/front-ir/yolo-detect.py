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

cascade_size=50

wait_time = 0

tracker = None
detected = False

DEBUG = False

model = ultralytics.YOLO(r"U:\front-ir\Colosseum\runs\detect\train13\weights\best.pt")

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
    norm = cv2.cvtColor(src=grey, code=cv2.COLOR_GRAY2RGB)
    #thresholded = cv2.cvtColor(cv2.threshold(src=grey, thresh=200, maxval=255, type=cv2.THRESH_BINARY)[1], code=cv2.COLOR_GRAY2RGB)

    results = model(norm)
    #results.show()
    for result in results:
       for box in result.boxes:  # Iterate over each box
       # Get bounding box coordinates (in pixels)
           if (box.conf.item() > 0.1):
               x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # Coordinates in the format (x1, y1, x2, y2)
               x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

               color = (0, 255, 0)  # Green color for the bounding box
               thickness = 2  # Thickness of the bounding box lines
               cv2.rectangle(grey, (x1, y1), (x2, y2), color, thickness)

               confidence = box.conf.item()  # Confidence score for the detection
               class_id = int(box.cls)  # Class ID of the detected object

               # Print the coordinates and other details
               print(f"Bounding Box: ({x1}, {y1}, {x2}, {y2})")
               print(f"Confidence: {confidence}")
               print(f"Class ID: {class_id}")

    cv2.putText(img=grey, text=f"{fps}", org=textOrg, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,255), thickness=2)
    cv2.imshow("Grey", grey)
    # cv2.imshow("norm", norm)
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
