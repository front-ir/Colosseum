import airsim
import numpy as np
import os
import cv2
import os

script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_dir = os.path.normpath(f"{script_path}/../screenshots/shahed/negative/")



for f in os.listdir(screenshots_dir):
    f = os.path.join(screenshots_dir, f)
    img = 255 - cv2.imread(f)
    cv2.imwrite(f, img)