import airsim
import numpy as np
import os
import cv2

script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_dir = f"{script_path}/../screenshots/shahed/positive/"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(1.5, 0, 0))
client.simSetCameraPose("0", camera_pose)

image_type = airsim.ImageType.Scene
pixels_as_float = False
compress = False

i = 0

while i < 475:

    response = client.simGetImages([
        airsim.ImageRequest("0", image_type, pixels_as_float, compress)])[0]  # segmentation view

    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    cv2.imwrite(os.path.normpath(screenshots_dir + f'{image_type}_xyz_{i}_int.png'), image) # write to png
    i += 1
