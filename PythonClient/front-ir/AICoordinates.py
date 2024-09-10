import cv2
import numpy as np
import socket
import base64
import time
import os
import math
import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32MultiArray


class Coordinates(Node):

    def __init__(self):
        super().__init__('AICoordinates')
        self.coordinatesPub = self.create_publisher(Int32MultiArray, 'coordinates', 10)
        timer_period = 1.0  # seconds
        self.main()
       
    def main(self):
        # Path to the Haar cascade file (you need to set this path correctly)
        script_path = os.path.dirname(os.path.realpath(__file__))
        cascade_path = '/root/ros2_ws/src/drone_controlling/drone_controlling/RAB_HAAR/cascade.xml'

        BUFF_SIZE = 65536
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, BUFF_SIZE)
        host_ip = '192.168.137.1'  # IP address of the server sending the video stream
        port = 9999
        message = b'Hello'

        client_socket.sendto(message, (host_ip, port))

        # Load Haar cascade for object detection
        def load_classifier(cascade_path):
            cascade = cv2.CascadeClassifier(cascade_path)
            if cascade.empty():
                raise IOError("Cannot load cascade classifier from file: " + cascade_path)
            return cascade

        cascade = load_classifier(cascade_path)

        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        thickness = 2
        textSize, baseline = cv2.getTextSize("FPS", fontFace, fontScale, thickness)
        textOrg = (10, 10 + textSize[1])

        frameCount = 0
        startTime = time.time()
        fps = 0
        print("1")
        tracker = None
        detected = False
        margin = 10
        inv = False
        thresh = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
        wait_time = 0

        def create_tracker(frame, x, y, w, h):
            tracker = cv2.legacy.TrackerCSRT_create()
            tracker.init(frame, (x, y, w, h))
            return tracker

        while True:
            # Receive video stream from UDP
            packet, _ = client_socket.recvfrom(BUFF_SIZE)
            data = base64.b64decode(packet, ' /')
            npdata = np.fromstring(data, dtype=np.uint8)
            frame = cv2.imdecode(npdata, 1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2GRAY)

            # Process the received frame
            _, thresholded = cv2.threshold(frame, 210, 255, thresh)

            if time.time() - wait_time > 2:
                wait_time = 0

            if not detected and wait_time == 0:
                detections = cascade.detectMultiScale(thresholded, scaleFactor=1.05, minNeighbors=55, minSize=(10, 10), maxSize=(55, 55))
                if len(detections) > 0:
                    (dx, dy, dw, dh) = detections[0]
                    tracker = create_tracker(thresholded, dx-margin, dy-margin, dw+margin, dh+margin)
                    detected = True

            if detected:
                (success, box) = tracker.update(thresholded)
                if success:
                    (tx, ty, tw, th) = [int(v) for v in box]
                    tracker_x, tracker_y, tracker_w, tracker_h = tx, ty, tw, th

                    cropped_tracker = cv2.getRectSubPix(thresholded, (tw, th), (tx + tw / 2, ty + th / 2))

                    circles = cv2.HoughCircles(cropped_tracker, cv2.HOUGH_GRADIENT, dp=1.1, minDist=100,
                                            param1=110, param2=25, minRadius=10, maxRadius=100)
                    if circles is not None:
                        circles = np.uint16(np.around(circles))  # Round and convert to integer
                        for circle in circles[0, :]:
                            if abs((circle[0]+tx) - (tx + (tw / 2))) < tw / 4:
                                print("Round shape detected in center of the tracker. Stopping tracking for 2 sec")
                                tracker = None
                                detected = False
                                wait_time = time.time()
                                continue

                    contours, _ = cv2.findContours(cropped_tracker, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours is None or len(contours) != 1:
                        print("Tracker lost or multiple contours. Refreshing")
                        tracker = None
                        detected = False
                    else:
                        contour = contours[0]
                        (cx, cy, cw, ch) = cv2.boundingRect(contour)
                        cx -= margin
                        cy -= margin
                        cw += 2 * margin
                        ch += 2 * margin

                        if abs(cw - tw) > math.log(tw) * 2:
                            tracker_x, tracker_y, tracker_w, tracker_h = cx + tx, cy + ty, cw, ch
                            tracker = create_tracker(thresholded, tracker_x, tracker_y, tracker_w, tracker_h)

                        cv2.rectangle(frame, (tracker_x, tracker_y), (tracker_x + tracker_w, tracker_y + tracker_h), (0, 255, 0), 2)
                        print(f"X coordinate {tracker_x + tracker_w * 0.5}")
                        print(f"Y coordinate {tracker_y + tracker_h *0.5}")
                        # Create and publish an array for x_trg and y_trg
                        array_msg = Int32MultiArray()
                        array_msg.data = [int(tracker_x + tracker_w * 0.5), int(tracker_y + tracker_h * 0.5)]
                        self.coordinatesPub.publish(array_msg)
                        
                        offset_contours = [contour + np.array([tracker_x, tracker_y], dtype=np.int32) for contour in contours]
                        cv2.drawContours(frame, offset_contours, -1, (0, 255, 0), 1)
                else:
                    print("Tracker lost")
                    tracker = None
                    detected = False

            # Display FPS and frames
            #cv2.putText(frame, 'FPS ' + str(fps), textOrg, fontFace, fontScale, (255, 0, 255), thickness)
            #cv2.imshow("Grey", frame)
            #cv2.imshow("Thresholded", thresholded)
            
            frameCount += 1
            endTime = time.time()
            diff = endTime - startTime
            if diff > 1:
                fps = frameCount
                frameCount = 0
                startTime = endTime

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
            if key == ord('r'):
                print("Reset tracker")
                detected = False
                tracker = None
        client_socket.close()
        cv2.destroyAllWindows()

def main(args=None):
    rclpy.init(args=args)
    node = Coordinates()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
