import cv2
import numpy as np
from skimage import img_as_float
from skimage.segmentation import active_contour
from skimage.filters import gaussian
import os
import time


# Kalman filter initialization
def initialize_kalman():
    kalman = cv2.KalmanFilter(4, 2)
    kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                         [0, 1, 0, 0]], np.float32)
    kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                        [0, 1, 0, 1],
                                        [0, 0, 1, 0],
                                        [0, 0, 0, 1]], np.float32)
    kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
    kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    kalman.errorCovPost = np.eye(4, dtype=np.float32)
    return kalman

def load_classifier(cascade_path):
    # Load the Haar cascade file
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise IOError("Cannot load cascade classifier from file: " + cascade_path)
    return cascade
script_path = os.path.dirname(os.path.realpath(__file__))

cascade_type = "RAB_HAAR"

# Load video
cap = cv2.VideoCapture('output_video.avi')

# Get the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Convert to grayscale
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Initial detection using Haar Cascade (Assume cascade is already trained)
drone_cascade = load_classifier(f'{script_path}/{cascade_type}/cascade.xml')
drones = drone_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)

# Initialize Kalman filter
kalman = initialize_kalman()

# Initialize snake contour
if len(drones) > 0:
    x, y, w, h = drones[0]
    kalman.statePre = np.array([x + w // 2 + w // 4, y + h // 2 + h // 4, 0, 0], np.float32)
    s = np.linspace(0, 2 * np.pi, 150)
    r = h // 2  # Approximate radius
    init = np.array([y + r + (r /2) * np.sin(s), x + r + (r /2) * np.cos(s)]).T

fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
thickness = 2
textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
print(textSize)
textOrg = (10, 10 + textSize[1])
frameCount = 0
startTime = time.time()
fps = 0

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian filter to smooth the image
    smoothed_frame = gaussian(gray_frame, 1)

    # Normalize image for active contouring
    image = img_as_float(smoothed_frame)

    # Predict the next position of the drone using Kalman filter
    prediction = kalman.predict()
    predicted_center = (int(prediction[0]), int(prediction[1]))

    # Use the predicted position to adjust the snake contour initialization
    init = np.array([predicted_center[1] + (r/2) * np.sin(s), predicted_center[0] + (r/2) * np.cos(s)]).T

    # Evolve snake contour
    snake = active_contour(image, init, alpha=0.015, beta=10, gamma=0.001)

    # Correct Kalman filter with the new measurement
    measured_center = np.array([np.mean(snake[:, 1]), np.mean(snake[:, 0])], np.float32)
    kalman.correct(measured_center)

    # Draw the contour on the frame
    frame_with_contour = frame.copy()
    cv2.circle(frame_with_contour, predicted_center, 5, (255, 0, 0), -1)
    for i in range(len(snake)):
        cv2.circle(frame_with_contour, (int(snake[i][1]), int(snake[i][0])), 1, (0, 255, 0), -1)
        cv2.circle(frame_with_contour, (int(init[i][1]), int(init[i][0])), 1, (0, 255, 0), -1)

    cv2.putText(frame_with_contour,'FPS ' + str(fps),textOrg, fontFace, fontScale,(255,0,255),thickness)

    # Show the frame
    cv2.imshow('Kalman + Snake Contour Tracking', frame_with_contour)

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

# Release resources
cap.release()
cv2.destroyAllWindows()
