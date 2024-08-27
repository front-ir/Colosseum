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

#client = airsim.MultirotorClient()
#client.confirmConnection()
#client.enableApiControl(True)
#
## Path to the Haar cascade file
#
#camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(1.5, 0, 0))
#client.simSetCameraPose("0", camera_pose)
#
#image_type = 0
#pixels_as_float = False
#compress = False
#
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output_video.avi', fourcc, 20.0, (640, 640), False)

#while True:
#    # because this method returns std::vector<uint8>, msgpack decides to encode it as a string unfortunately.
#    response = client.simGetImages([
#            airsim.ImageRequest("0", image_type, pixels_as_float, compress)])[0]
#    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
#    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
#
#    out.write(image)
#
#    cv2.imshow("Grey", image)
#    key = cv2.waitKey(1) & 0xFF
#    if (key == 27 or key == ord('q') or key == ord('x')):
#        break
#
#out.release()
#cv2.destroyAllWindows()
cascade_type = "RAB_HAAR"
cascade = load_classifier(f'{script_path}/{cascade_type}/cascade.xml')

#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
input_video = cv2.VideoCapture('output_video.avi')
#output_video = cv2.VideoWriter(f'{cascade_type}.mp4', fourcc, 20.0, (640, 640), False)

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

detected = False
(x,y,w,h) = 0,0,0,0

while input_video.isOpened():
    ret, frame = input_video.read()
    if ret:
        # Write the frame into the output file
        if not detected:
            (x, y, w, h) = cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10))[0]
            s = np.linspace(0, 2 * np.pi, 150)
            x_center = x + w / 2
            y_center = y + h / 2
            x_init = x_center + w / 4 * np.cos(s)
            y_init = y_center + h / 4 * np.sin(s)
            init = np.array([y_init, x_init]).T


            x_center_cropped = w / 2
            y_center_cropped = h / 2
            x_init = x_center_cropped + w / 4 * np.cos(s)
            y_init = y_center_cropped + h / 4 * np.sin(s)
            init_cropped = np.array([y_init, x_init]).T

        #cropped_image = cv2.getRectSubPix(frame, (w, h), (x + w / 2, y + h / 2))

        #cv2.rectangle(cropped_image, (x, y), (x + w, y + h), (255, 255, 255), 1)
#
        # s = np.linspace(0, 2 * np.pi, 150)
        # x_center = x + w / 2
        # y_center = y + h / 2
        # x_init = x_center + w / 4 * np.cos(s)
        # y_init = y_center + h / 4 * np.sin(s)
        # init = np.array([y_init, x_init]).T
#
#
        # x_center_cropped = w / 2
        # y_center_cropped = h / 2
        # x_init = x_center_cropped + w / 4 * np.cos(s)
        # y_init = y_center_cropped + h / 4 * np.sin(s)
        # init_cropped = np.array([y_init, x_init]).T
#
        ### Apply the Snake algorithm
        snake = active_contour(frame, init_cropped, alpha=0.015, beta=19, gamma=0.01, max_num_iter=10000)
#
        for i in range(len(init)):
            #cv2.circle(frame, (int(init[i, 1]), int(init[i, 0])), 1, (0, 0, 255), 1)
            cv2.circle(frame, (int(snake[i, 1] + x), int(snake[i, 0]) + y), 1, (0, 255, 0), 1)
        ## Display the resulting frame
        #output_video.write(frame)
        #cv2.putText(frame,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,255,255),thickness)

        cv2.imshow('frame', frame)

        frameCount = frameCount  + 1
        endTime = time.time()
        diff = endTime - startTime
        if (diff > 1):
            fps = frameCount
            frameCount = 0
            startTime = endTime

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

input_video.release()
#output_video.release()
cv2.destroyAllWindows()
