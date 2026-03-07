#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from hand_msgs.msg import HandLandmarks, RH56DFTPFeedback
from collections import defaultdict

# Standard hand landmark connections
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (13, 17), (17, 18), (18, 19), (19, 20)
]

MAX_TOUCH = 800.0   # Upper limit for touch sensors

def get_color(value, mode="smooth"):
    # ⚠️ Normalize in a robust way
    v = np.clip(value / MAX_TOUCH, 0.0, 1.0)

    if mode == "threshold":
        if v < 0.25:
            return (0,255,0)
        elif v < 0.5:
            return (0,255,255)
        elif v < 0.75:
            return (0,165,255)
        else:
            return (0,0,255)
    elif mode == "smooth":
        # Hue 60 (green) → 0 (red)
        hue = 60 * (1 - v)
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
        return (int(bgr[0]), int(bgr[1]), int(bgr[2]))

class HandLandmarksVisualizer(Node):
    def __init__(self):
        super().__init__('hand_landmarks_visualizer')

        # Parameters
        self.declare_parameter("tactile_feedback", True)
        self.declare_parameter("color_mode", "smooth")  # "smooth" or "threshold"
        self.declare_parameter("ema_alpha", 0.3)
        self.declare_parameter("camera_topic", "/camera/color/image_raw")
        self.declare_parameter("feedback_topic", "/rh56dftp/feedback")
        self.declare_parameter("controlled_hand", "left")  # "left", "right" or "both"

        self.tactile_feedback = self.get_parameter("tactile_feedback").value
        self.color_mode = self.get_parameter("color_mode").value
        self.ema_alpha = self.get_parameter("ema_alpha").value
        self.camera_topic = self.get_parameter("camera_topic").value
        self.feedback_topic = self.get_parameter("feedback_topic").value
        self.controlled_hand = self.get_parameter("controlled_hand").value

        # ROS setup
        self.bridge = CvBridge()
        self.latest_landmarks = None
        self.latest_image = None
        self.filtered_feedback = defaultdict(float)

        self.create_subscription(
            HandLandmarks,
            "/hand_landmarks",
            self.landmarks_callback,
            10
        )

        self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            1
        )

        if self.tactile_feedback:
            self.create_subscription(
                RH56DFTPFeedback,
                self.feedback_topic,
                self.feedback_callback,
                10
            )

        self.timer = self.create_timer(0.03, self.render)
        self.get_logger().info("Hand Landmarks Visualizer started")

    # --------------------------
    # Callbacks
    # --------------------------
    def landmarks_callback(self, msg):
        self.latest_landmarks = msg

    def image_callback(self, msg):
        self.latest_image = msg

    def feedback_callback(self, msg):
        raw_data = {
            "pinky_tip": np.max(msg.pinky_tip_touch),
            "pinky_top": np.max(msg.pinky_top_touch),
            "pinky_palm": np.max(msg.pinky_palm_touch),
            "ring_tip": np.max(msg.ring_tip_touch),
            "ring_top": np.max(msg.ring_top_touch),
            "ring_palm": np.max(msg.ring_palm_touch),
            "middle_tip": np.max(msg.middle_tip_touch),
            "middle_top": np.max(msg.middle_top_touch),
            "middle_palm": np.max(msg.middle_palm_touch),
            "index_tip": np.max(msg.index_tip_touch),
            "index_top": np.max(msg.index_top_touch),
            "index_palm": np.max(msg.index_palm_touch),
            "thumb_tip": np.max(msg.thumb_tip_touch),
            "thumb_top": np.max(msg.thumb_top_touch),
            "thumb_middle": np.max(msg.thumb_middle_touch),
            "thumb_palm": np.max(msg.thumb_palm_touch),
            "palm": np.max(msg.palm_touch)
        }

        for key, value in raw_data.items():
            prev = self.filtered_feedback[key]
            self.filtered_feedback[key] = self.ema_alpha * value + (1 - self.ema_alpha) * prev

    # --------------------------
    # Rendering
    # --------------------------
    def render(self):
        # Base image
        if self.latest_image is not None:
            canvas = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
        else:
            canvas = np.zeros((600, 800, 3), dtype=np.uint8)

        if self.latest_landmarks is None:
            cv2.imshow("Hand Visualizer", canvas)
            cv2.waitKey(1)
            return

        landmarks = self.latest_landmarks.landmarks
        if len(landmarks) != 21:
            return

        h, w = canvas.shape[:2]
        pts = [(int(lm.position.x * w), int(lm.position.y * h)) for lm in landmarks]

        if not self.tactile_feedback:
            for start, end in HAND_CONNECTIONS:
                cv2.line(canvas, pts[start], pts[end], (255,0,0), 2)
            for p in pts:
                cv2.circle(canvas, p, 5, (0,0,255), -1)
        else:
            self.draw_with_feedback(canvas, pts)

        self.draw_legend(canvas)
        cv2.imshow("Hand Visualizer", canvas)
        cv2.waitKey(1)

    # Draw feedback colored hand (draw EVERYTHING)
    def draw_with_feedback(self, canvas, pts):

        f = self.filtered_feedback
        finger_map = {
            "pinky":  (17,18,19,20),
            "ring":   (13,14,15,16),
            "middle": (9,10,11,12),
            "index":  (5,6,7,8)
        }

        for name, (mcp, pip, dip, tip) in finger_map.items():
            cv2.line(canvas, pts[mcp], pts[pip], get_color(f[f"{name}_palm"], self.color_mode), 3)
            cv2.line(canvas, pts[pip], pts[dip], get_color(f[f"{name}_top"], self.color_mode), 3)
            cv2.line(canvas, pts[dip], pts[tip], get_color(f[f"{name}_top"], self.color_mode), 3)
            cv2.circle(canvas, pts[tip], 8, get_color(f[f"{name}_tip"], self.color_mode), -1)

        # Thumb
        cv2.line(canvas, pts[1], pts[2], get_color(f["thumb_palm"], self.color_mode), 3)
        cv2.line(canvas, pts[2], pts[3], get_color(f["thumb_middle"], self.color_mode), 3)
        cv2.line(canvas, pts[3], pts[4], get_color(f["thumb_top"], self.color_mode), 3)
        cv2.circle(canvas, pts[4], 8, get_color(f["thumb_tip"], self.color_mode), -1)

        # Palm loop
        palm_links = [(0,5),(5,9),(9,13),(13,17),(17,0)]
        for a,b in palm_links:
            cv2.line(canvas, pts[a], pts[b], get_color(f["palm"], self.color_mode), 2)
        cv2.line(canvas, pts[0], pts[1], (255, 0, 0), 2)
        cv2.circle(canvas, pts[0], 6, (255,255,255), -1)

    # Draw color legend
    def draw_legend(self, canvas):
        x0 = canvas.shape[1]-50
        y0 = 50
        height = 400
        for i in range(height):
            value = MAX_TOUCH*(1 - i/height)
            cv2.line(canvas, (x0, y0+i), (x0+40, y0+i), get_color(value, self.color_mode), 1)
        cv2.putText(canvas, str(MAX_TOUCH), (x0-45, y0+10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
        cv2.putText(canvas, "0", (x0-45, y0+height), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)


def main(args=None):
    rclpy.init(args=args)
    node = HandLandmarksVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()