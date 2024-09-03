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
cascade = load_classifier('U:/front-ir/Colosseum/Unreal/Environments/Blocks/PythonScripts/RAB_HAAR/cascade.xml')
#cascade = load_classifier(f'{script_path}/screenshots/cascade/cascade.xml')

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

image_type = 0
pixels_as_float = False
compress = False
thresh = cv2.THRESH_BINARY


while True:
    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
    response = client.simGetImages([
            airsim.ImageRequest("0", image_type, pixels_as_float, compress)])[0]
    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    _, thresholded = cv2.threshold(image, 200, 255, thresh)

    detections = cascade.detectMultiScale(thresholded, scaleFactor=1.1, minNeighbors=5, minSize=(10, 10))

    if len(detections) > 0:
        (x, y, w, h) = detections[0]
        cv2.rectangle(thresholded, (x, y), (x + w, y + h), (255, 255, 255), 1)
        #print(f"{x + w * 0.5} {y + h * 0.5}")
        print(w, h)

    cv2.putText(thresholded,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)

    cv2.imshow("Grey", thresholded)
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
