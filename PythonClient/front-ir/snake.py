import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from skimage import data, img_as_float, img_as_uint
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import os

script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_path = os.path.normpath(f"{script_path}/../screenshots")

# Load an example image from scikit-image
#image = img_as_uint(data.astronaut())
image_path = os.path.normpath(f"{screenshots_path}/shahed-for-snake.png")
cascade_path = os.path.normpath(f"{script_path}/day/cascade.xml")

cascade = cv2.CascadeClassifier(cascade_path)
grey_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGRA2GRAY)

(x, y, w, h) = cascade.detectMultiScale(grey_image, scaleFactor=1.15, minNeighbors=11, minSize=(10, 10))[0]

st = time.time()

rect = False
contour = np.array([])

if rect:

    rect_contour = np.array([
        [x, y],  # Top-left corner
        [x + w, y],  # Top-right corner
        [x + w, y + h],  # Bottom-right corner
        [x, y + h],  # Bottom-left corner
        [x, y]  # Close the rectangle
    ])

    smooth_contour = np.linspace(rect_contour[0], rect_contour[1], 100)
    smooth_contour = np.vstack([smooth_contour, np.linspace(rect_contour[1], rect_contour[2], 100)])
    smooth_contour = np.vstack([smooth_contour, np.linspace(rect_contour[2], rect_contour[3], 100)])
    contour = np.vstack([smooth_contour, np.linspace(rect_contour[3], rect_contour[0], 100)])

else:

    # Define the number of points for the ellipse
    num_points = 400
    s = np.linspace(0, 2*np.pi, num_points)

    # Create an elliptical contour
    center_x = x + w / 2
    center_y = y + h / 2
    a = w / 2  # Semi-major axis
    b = h / 2  # Semi-minor axis
    contour = np.array([center_x + a*np.cos(s), center_y + b*np.sin(s)]).T

print(f"Calculation = {time.time() - st}")

#cv2.rectangle(grey_image, (x, y), (x + w, y + h), (255, 255, 0), 1)
#print (x + (w * 0.5), y + (h * 0.5))
#cv2.rectangle(grey_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#print(f"Detected at: x={x + w * 0.5}, y={y + h * 0.5}")
# Display the result
#cv2.imshow('Detected Objects', grey_image)
#cv2.waitKey(0)
#
#exit(0)

# Apply a Gaussian filter for smoothing
#smoothed_image = gaussian(gray_image, 3)
#st = time.time()
# 309 105 22 22
# Define an initial circular contour
#s = np.linspace(0, 2*np.pi, 400)
#x = 220 + 100*np.cos(s)
#y = 100 + 100*np.sin(s)
#init = np.array([x, y]).T
#print(f"Calculation = {time.time() - st}")
#print(s)
#print(x)
#print(s)
#print(init)
#exit(0)


# Perform the snake algorithm
sn = time.time()
snake = active_contour(grey_image, contour, alpha=0.015, beta=10, gamma=0.001)

print(f"Snake algo = {time.time() - sn}")

# Plot the results
fig, ax = plt.subplots(figsize=(7, 7))
ax.imshow(grey_image, cmap=plt.cm.gray)
ax.plot(contour[:, 0], contour[:, 1], '--r', lw=1, label='Initial Contour')
ax.plot(snake[:, 0], snake[:, 1], '-b', lw=1, label='Snake Contour')
ax.set_xticks([]), ax.set_yticks([])
ax.legend(loc='best')
plt.show()
