import airsim
import numpy as np
import os
import cv2
import unreal

script_path = os.path.dirname(os.path.realpath(__file__))
screenshots_dir = f"{script_path}/screenshots/negative/"

# connect to the AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)

camera_pose = airsim.Pose(airsim.Vector3r(0, 0, 0), airsim.to_quaternion(1.5, 0, 0))
client.simSetCameraPose("0", camera_pose)

image_type = 0
pixels_as_float = False
compress = False

i = 0

while i < 3000:

    response = client.simGetImages([
        airsim.ImageRequest("0", image_type, pixels_as_float, compress)])[0]

    image = np.frombuffer(response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    _, thresholded = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)

    cv2.imwrite(os.path.normpath(screenshots_dir + f'{image_type}_self_rotation_{i}_int.png'), thresholded) # write to png
    #cv2.imwrite(os.path.normpath(screenshots_dir + f'asd{i}.png'), thresholded) # write to png
    i += 1
