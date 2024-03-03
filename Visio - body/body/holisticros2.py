import cv2 
import mediapipe as mp
import numpy as np
import time
import pyrealsense2 as rs
import argparse
import rclpy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class detectorhros:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("image_topic",Image, queue_size=10)

    def run(self):
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        pipeline.start(config)

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                results = holistic.process(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                f1 = mp.solutions.drawing_utils.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1)
                f2 = mp.solutions.drawing_utils.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)

                hr1 = mp.solutions.drawing_utils.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4)
                hr2 = mp.solutions.drawing_utils.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)

                hl1 = mp.solutions.drawing_utils.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4)
                hl2 = mp.solutions.drawing_utils.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)

                p1 = mp.solutions.drawing_utils.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4)
                p2 = mp.solutions.drawing_utils.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)

                self.mp_drawing.draw_landmarks(image, results.face_landmarks, mp.solutions.holistic.FACEMESH_TESSELATION,
                                           f1, f2)

                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                           mp.solutions.holistic.HAND_CONNECTIONS, hr1, hr2)

                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                           mp.solutions.holistic.HAND_CONNECTIONS, hl1, hl2)

                self.mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                           mp.solutions.holistic.POSE_CONNECTIONS, p1, p2)

                cv2.imshow("holistic vision", image)
                cv2.waitKey(1)

                try:
                    self.image_pub.publish(self.bridge.cv2_to_imgmsg(image, "bgr8"))
                except CvBridgeError as e:
                    print(e)

        pipeline.stop()
        cv2.destroyAllWindows()



dros = detectorhros()
dros.run()
