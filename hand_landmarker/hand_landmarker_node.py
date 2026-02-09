#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time

from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge


class HandLandmarkerNode(Node):
    def __init__(self):
        super().__init__('hand_landmarker_node')

        # Parameters
        self.declare_parameter('model_path', '/root/models/hand_landmarker.task')
        self.declare_parameter('debug', True)
        self.declare_parameter('publish_rate', 15.0)  # Hz

        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.debug = self.get_parameter('debug').get_parameter_value().bool_value
        self.publish_rate = self.get_parameter('publish_rate').get_parameter_value().double_value

        # MediaPipe Tasks API setup
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(options)

        # ROS interfaces
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            10
        )
        self.publisher = self.create_publisher(PoseArray, '/hand_landmarks', 10)

        # Timing for 15 Hz
        self.last_draw_time = 0.0

        self.get_logger().info('Hand Landmarker Node started')

    def image_callback(self, msg: Image):
        # Convert ROS Image → OpenCV
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # Detect hands
        result = self.detector.detect(mp_image)

        # Publish PoseArray
        pose_array = PoseArray()
        pose_array.header = msg.header

        if result.hand_landmarks:
            for hand in result.hand_landmarks:
                for lm in hand:
                    pose = Pose()
                    pose.position.x = lm.x
                    pose.position.y = lm.y
                    pose.position.z = lm.z
                    pose.orientation.w = 1.0
                    pose_array.poses.append(pose)

        self.publisher.publish(pose_array)

        # Draw landmarks manually if debug=True at limited rate
        if self.debug and (time.time() - self.last_draw_time) >= 1.0 / self.publish_rate:
            annotated = frame.copy()
            h, w, _ = frame.shape
            for hand in result.hand_landmarks:
                # Draw landmarks
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 4, (0, 255, 0), -1)

                # Draw connections manually (Hand topology)
                connections = [
                    (0,1),(1,2),(2,3),(3,4),       # Thumb
                    (0,5),(5,6),(6,7),(7,8),       # Index
                    (0,9),(9,10),(10,11),(11,12),  # Middle
                    (0,13),(13,14),(14,15),(15,16),# Ring
                    (0,17),(17,18),(18,19),(19,20) # Pinky
                ]
                for start_idx, end_idx in connections:
                    start = hand[start_idx]
                    end = hand[end_idx]
                    x0, y0 = int(start.x * w), int(start.y * h)
                    x1, y1 = int(end.x * w), int(end.y * h)
                    cv2.line(annotated, (x0, y0), (x1, y1), (0, 0, 255), 2)

            cv2.imshow("Hand Landmarks", annotated)
            cv2.waitKey(1)
            self.last_draw_time = time.time()


def main():
    rclpy.init()
    node = HandLandmarkerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
