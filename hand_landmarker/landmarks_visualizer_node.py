#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from cv_bridge import CvBridge
from collections import defaultdict
from sensor_msgs.msg import Image
from hand_msgs.msg import HandLandmarks, RH56DFTPFeedback

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20)
]

MAX_TOUCH = 800.0

def get_color(value, hand):
    v = np.clip(value / MAX_TOUCH, 0.0, 1.0)
    if hand == "left":
        if v < 0.5:
            r = int(2*v*255)
            g = 255
        else:
            r = 255
            g = int((1-2*(v-0.5))*255)
        b = 0
    else:  # right
        if v < 0.5:
            g = int(2*v*255)
            b = 255
        else:
            g = int((1-2*(v-0.5))*255)
            b = 255
        r = int(v*255)
    return (b,g,r)

class HandLandmarksVisualizer(Node):

    def __init__(self):
        super().__init__("hand_landmarks_visualizer")

        # Parameters
        self.declare_parameter("controlled_hand","left")
        self.declare_parameter("ema_alpha",0.3)
        self.declare_parameter("camera_topic","/camera/color/image_raw")
        self.declare_parameter("feedback_topic","/rh56dftp/feedback")

        self.controlled_hand = self.get_parameter("controlled_hand").value.lower()
        self.ema_alpha = self.get_parameter("ema_alpha").value
        camera_topic = self.get_parameter("camera_topic").value
        feedback_topic = self.get_parameter("feedback_topic").value

        self.bridge = CvBridge()
        self.latest_image = None

        # Store the last received landmarks/feedback for both hands
        self.landmarks = {"left": None, "right": None}
        self.feedback = {"left": defaultdict(float), "right": defaultdict(float)}

        # Subscriptions
        self.create_subscription(Image, camera_topic, self.image_callback, 1)
        self.create_subscription(HandLandmarks, "/hand_landmarks", self.landmarks_callback, 10)
        self.create_subscription(RH56DFTPFeedback, feedback_topic, self.feedback_callback, 10)

        self.timer = self.create_timer(0.03, self.render)
        self.get_logger().info("Hand Landmarks Visualizer started")

    # --------------------------
    # Callbacks
    # --------------------------
    def image_callback(self, msg):
        self.latest_image = msg

    def landmarks_callback(self, msg):
        hand_id = msg.hand_id.lower()
        if "left" in hand_id:
            hand = "left"
        elif "right" in hand_id:
            hand = "right"
        else:
            return

        # Store latest message for this hand
        self.landmarks[hand] = msg

    def feedback_callback(self, msg):
        hand_id = msg.hand_id.lower()
        if "left" in hand_id:
            hand = "left"
        elif "right" in hand_id:
            hand = "right"
        else:
            return

        raw = {
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

        fb = self.feedback[hand]
        for k, v in raw.items():
            fb[k] = self.ema_alpha*v + (1-self.ema_alpha)*fb[k]

    # --------------------------
    # Rendering helpers
    # --------------------------
    def compute_points(self, landmarks, canvas):
        h, w = canvas.shape[:2]
        return [(int(lm.position.x*w), int(lm.position.y*h)) for lm in landmarks]

    def draw_hand(self, canvas, pts, hand):
        fb = self.feedback[hand]
        def c(v): return get_color(v, hand)
        finger_map = {
            "pinky": (17,18,19,20),
            "ring": (13,14,15,16),
            "middle": (9,10,11,12),
            "index": (5,6,7,8)
        }

        for name,(mcp,pip,dip,tip) in finger_map.items():
            cv2.line(canvas, pts[mcp], pts[pip], c(fb[f"{name}_palm"]), 3)
            cv2.line(canvas, pts[pip], pts[dip], c(fb[f"{name}_top"]), 3)
            cv2.line(canvas, pts[dip], pts[tip], c(fb[f"{name}_top"]), 3)
            cv2.circle(canvas, pts[tip], 8, c(fb[f"{name}_tip"]), -1)

        cv2.line(canvas, pts[1], pts[2], c(fb["thumb_palm"]), 3)
        cv2.line(canvas, pts[2], pts[3], c(fb["thumb_middle"]), 3)
        cv2.line(canvas, pts[3], pts[4], c(fb["thumb_top"]), 3)
        cv2.circle(canvas, pts[4], 8, c(fb["thumb_tip"]), -1)

        palm_links = [(0,1),(0,5),(5,9),(9,13),(13,17),(17,0)]
        for a,b in palm_links:
            cv2.line(canvas, pts[a], pts[b], c(fb["palm"]), 2)
        cv2.circle(canvas, pts[0], 7, (255,255,255), -1)

    def draw_legend(self, canvas, x0, y0, hand):
        height = 300
        width = 20
        for i in range(height):
            value = MAX_TOUCH*(1-i/height)
            cv2.line(canvas, (x0,y0+i), (x0+width, y0+i), get_color(value, hand), 1)
        cv2.putText(canvas,str(MAX_TOUCH),(x0-10,y0-10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)
        cv2.putText(canvas,"0",(x0-10,y0+height+10),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,255,255),1)

    def draw_legends(self, canvas):
        h,w = canvas.shape[:2]
        self.draw_legend(canvas, 10, 50, "left")
        self.draw_legend(canvas, w-30, 50, "right")

    # --------------------------
    # Main render
    # --------------------------
    def render(self):
        if self.latest_image is None:
            canvas = np.zeros((720,1280,3), dtype=np.uint8)
        else:
            canvas = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding="bgr8")

        hands_to_draw = []
        if self.controlled_hand == "both":
            hands_to_draw = ["left","right"]
        else:
            hands_to_draw = [self.controlled_hand]

        for hand in hands_to_draw:
            msg = self.landmarks.get(hand)
            if msg is None or len(msg.landmarks) != 21:
                continue
            pts = self.compute_points(msg.landmarks, canvas)
            self.draw_hand(canvas, pts, hand)

        self.draw_legends(canvas)
        cv2.imshow("Hand Visualizer", canvas)
        cv2.waitKey(1)

def main():
    rclpy.init()
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