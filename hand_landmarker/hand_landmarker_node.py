#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from hand_msgs.msg import HandLandmarks
from geometry_msgs.msg import Pose

# Standard hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # Index finger + wrist to index MCP
    (5, 9), (9, 10), (10, 11), (11, 12),   # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16), # Ring finger
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky finger + wrist to pinky MCP
]



class HandLandmarkerNode(Node):
    def __init__(self):
        super().__init__('hand_landmarker_node')

        # Parameters
        self.declare_parameter('model_path', '/root/ros2_ws/src/hand_landmarker/models/hand_landmarker.task')
        self.declare_parameter('debug', False)
        self.declare_parameter('publish_rate', 30.0)
        self.declare_parameter('camera_topic', '/camera/color/image_raw')

        model_path = self.get_parameter('model_path').value
        self.debug = self.get_parameter('debug').value
        self.publish_rate = self.get_parameter('publish_rate').value
        self.camera_topic = self.get_parameter('camera_topic').value

        # MediaPipe setup
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        # ROS setup
        self.bridge = CvBridge()
        self.latest_image = None

        # Camera subscription (small queue to avoid lag buildup)
        self.sub_image = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            1
        )

        self.pub_landmarks = self.create_publisher(
            HandLandmarks,
            '/hand_landmarks',
            10
        )

        # Timer for throttled processing
        self.timer = self.create_timer(
            1.0 / self.publish_rate,
            self.process_image
        )

        self.get_logger().info("Hand Landmarker Node started")
        self.get_logger().info(f"Subscribing to: {self.camera_topic}")
        self.get_logger().info(f"Processing at: {self.publish_rate} Hz")

    # -------------------------
    # Image callback (lightweight)
    # -------------------------
    def image_callback(self, msg: Image):
        self.latest_image = msg  # just store newest frame

    # -------------------------
    # Timer callback (heavy work)
    # -------------------------
    def process_image(self):
        if self.latest_image is None:
            return

        msg = self.latest_image

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self.detector.detect(mp_image)

        annotated = cv_image.copy()
        h, w, _ = cv_image.shape

        for hand_idx, hand_landmarks in enumerate(result.hand_landmarks or []):

            if result.handedness and len(result.handedness) > hand_idx:
                hand_label = result.handedness[hand_idx][0].category_name
            else:
                hand_label = f'Hand_{hand_idx}'

            hl_msg = HandLandmarks()
            hl_msg.hand_id = hand_label

            for lm in hand_landmarks:
                pose = Pose()
                pose.position.x = lm.x
                pose.position.y = lm.y
                pose.position.z = lm.z
                pose.orientation.w = 1.0
                hl_msg.landmarks.append(pose)

                if self.debug:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(annotated, (cx, cy), 4, (0, 0, 255), -1)

            self.pub_landmarks.publish(hl_msg)

            if self.debug:
                line_color = (255, 0, 0) if hand_label.lower() == "right" else (0, 255, 0)
                for start_idx, end_idx in HAND_CONNECTIONS:
                    start_lm = hand_landmarks[start_idx]
                    end_lm = hand_landmarks[end_idx]
                    x0, y0 = int(start_lm.x * w), int(start_lm.y * h)
                    x1, y1 = int(end_lm.x * w), int(end_lm.y * h)
                    cv2.line(annotated, (x0, y0), (x1, y1), line_color, 2)

        if self.debug:
            cv2.imshow("Hand Landmarks", annotated)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarkerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
