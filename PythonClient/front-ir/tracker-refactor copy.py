import cv2
import numpy as np
import airsim
import time
import os
import math
import asyncio

script_path = os.path.dirname(p=os.path.realpath(__file__))
camera_pose = airsim.Pose(position_val=airsim.Vector3r(0, 0, 0), orientation_val=airsim.to_quaternion(1.5, 0, 0))

class DroneFinder:
    def __init__(self, cascade_name, camera_position):
        self.running = True
        self.DEBUG = True

        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.simSetCameraPose(camera_name="0", pose=camera_position)

        self.grey = None
        self.thresholded = None
        self.processed = None
        self.thresholdedcopy = None

        self.cascade = self.load_classifier(cascade_path=os.path.join(script_path, cascade_name, "cascade.xml"))

    def load_classifier(self, cascade_path):
        # Load the Haar cascade file
        cascade = cv2.CascadeClassifier(filename=cascade_path)
        if cascade.empty():
            raise IOError(f"Cannot load cascade classifier from file: {cascade_path}")
        return cascade

    async def run_cascade(self, minSize, maxSize):
        detections = self.cascade.detectMultiScale(image=self.thresholded, scaleFactor=1.05, minNeighbors=55, minSize=minSize, maxSize=maxSize)
        if len(detections) == 1:
            if self.DEBUG: print(f"Cascade center: {detections[0][0]+detections[0][2]*0.5} {detections[0][1]+detections[0][3]*0.5}")
            return (True, detections[0])
        return (False, (None, None, None, None))

    async def get_images(self):
        while self.running:
            response = self.client.simGetImages(requests=[
                airsim.ImageRequest(camera_name="0", image_type=airsim.ImageType.Scene, pixels_as_float=False, compress=False)])[0]
            image = np.frombuffer(buffer=response.image_data_uint8, dtype=np.uint8).reshape(response.height, response.width, 3)
            self.grey = cv2.cvtColor(src=image, code=cv2.COLOR_RGBA2GRAY)
            _, self.thresholded = cv2.threshold(src=self.grey, thresh=200, maxval=255, type=cv2.THRESH_BINARY)
            await asyncio.sleep(0)

    async def show_frame(self):
        textSize, _ = cv2.getTextSize(text="FPS", fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, thickness=1)
        textOrg = (10, 10 + textSize[1])
        frameCount = 0
        startTime = time.time()
        fps = 0

        while self.running:
            #frame_to_display = self.grey
            cv2.putText(img=self.grey, text=f"{fps}", org=textOrg, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255,0,255), thickness=1)
            cv2.imshow("Grey", self.grey)
            #cv2.imshow("thresholded", self.thresholded)
            frameCount += 1
            endTime = time.time()
            diff = endTime - startTime
            if (diff > 1):
                fps = frameCount
                frameCount = 0
                startTime = endTime
            key = cv2.waitKey(1) & 0xFF
            if (key == 27 or key == ord('q') or key == ord('x')):
                self.running = False
            if (key == ord('r')):
                print("Reset tracker")
                self.detected = False
                self.tracker = None
            await asyncio.sleep(0)

    async def process_image(self):
        detected = False
        tracker = None
        wait_time = 0
        cascade_size = 50
        while True:
            if self.grey is None or self.thresholded is None:
                continue

            thresholded_to_process = self.thresholded.copy()
            grey_to_process = self.grey

            if detected and (tracker is not None):
                print("a")
                (success, box) = tracker.update(thresholded_to_process)
                if success:
                    tracker_coords = [int(v) for v in box]
                    tracker_center = (tracker_coords[0]+tracker_coords[2]*0.5,
                                      tracker_coords[1]+tracker_coords[3]*0.5)
                    if self.DEBUG:
                        print("Tracker center:", *tracker_center)
                        cv2.rectangle(img=grey_to_process, pt1=(tracker_coords[0], tracker_coords[1]),
                                      pt2=(tracker_coords[0] + tracker_coords[2], tracker_coords[1] + tracker_coords[3]),
                                      color=(0, 0, 255), thickness=1)

                    cropped_tracker = cv2.getRectSubPix(image=thresholded_to_process, patchSize=(tracker_coords[2], tracker_coords[3]),
                                                        center=tracker_center)
                    circles = cv2.HoughCircles(image=cropped_tracker, method=cv2.HOUGH_GRADIENT, dp=1.1, minDist=100,
                                               param1=110, param2=25, minRadius=10, maxRadius=100)
                    if circles is not None:
                        if self.DEBUG:
                            circle = np.uint16(np.around(circles))[0, :][0]
                            circle_center = (circle[0]+tracker_coords[0], circle[1]+tracker_coords[1])
                            circle_radius = circle[2]
                            print("Circle center and r", *circle_center, circle_radius)
                            print("Round shape detected inside the tracker. Stopping tracking for 2 sec")
                        tracker = None
                        detected = False
                        wait_time = time.time()
                    else:
                        contours, _ = cv2.findContours(image=cropped_tracker, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
                        if contours is None:
                            tracker = None
                            detected = False
                            wait_time = time.time()
                            if self.DEBUG: print("Nothing is inside the tracker. Refreshing")
                        elif len(contours) > 1:
                            detected, cascade_coords = await self.run_cascade(minSize=(int(cascade_size*0.25), int(cascade_size*0.25)),
                                                   maxSize=(cascade_size, cascade_size))
                            if not detected:
                                tracker = None
                                wait_time = time.time()
                                if self.DEBUG:print("Multiple contours detected in box. Refresh")
                        elif len(contours) == 1:
                            contour = contours[0] + np.array([tracker_coords[0], tracker_coords[1]], dtype=np.int16)

                            contour_coords = [int(x) for x in cv2.boundingRect(array=contour)]
                            contour_coords[0] -= 5
                            contour_coords[1] -= 5
                            contour_coords[2] += 10
                            contour_coords[3] += 10

                            if self.DEBUG:
                                contour_center = (contour_coords[0]+contour_coords[2]*0.5, contour_coords[1]+contour_coords[3]*0.5)
                                print("Contour center:", *contour_center)
                                cv2.rectangle(img=grey_to_process, pt1=(contour_coords[0], contour_coords[1]),
                                            pt2=(contour_coords[0] + contour_coords[2], contour_coords[1] + contour_coords[3]),
                                            color=(255, 255, 255), thickness=1)

                            if math.fabs(contour_coords[2] - tracker_coords[2]) > 3 or math.fabs(contour_coords[3] - tracker_coords[3]) > 3:
                                cascade_size=contour_coords[2]
                                tracker = None
                                tracker = cv2.legacy.TrackerCSRT_create()
                                tracker.init(thresholded_to_process, contour_coords)
                            cv2.drawContours(image=grey_to_process, contours=[contour], contourIdx=-1, color=(0, 255, 0), thickness=1)
                            self.processed = grey_to_process
                else:
                    print("Tracker lost")
                    tracker = None
                    detected = None

            if not detected and time.time() - wait_time > 1:
                wait_time = 0
                detected, cascade_coords = await self.run_cascade(minSize=(int(cascade_size*0.25), int(cascade_size*0.25)),
                                       maxSize=(cascade_size, cascade_size))
                if detected:
                    tracker = cv2.legacy.TrackerCSRT_create()
                    tracker.init(thresholded_to_process, cascade_coords)

    async def run(self):
        await asyncio.gather(self.get_images(), self.process_image(), self.show_frame())

if __name__ == "__main__":
    loop = asyncio.get_event_loop()
    df = DroneFinder(cascade_name="RAB_HAAR", camera_position=camera_pose)
    try:
        loop.run_until_complete(df.run())
    except KeyboardInterrupt:
        print("Stopping")
        loop.close()
