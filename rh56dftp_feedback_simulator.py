#!/usr/bin/env python3
import os
import shutil
import time
import numpy as np

import rclpy
from rclpy.serialization import serialize_message
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from hand_msgs.msg import RH56DFTPFeedback

from rosbag2_py import SequentialWriter, StorageOptions, ConverterOptions, TopicMetadata

def create_fake_feedback():
    """Crea un messaggio RH56DFTPFeedback con dati casuali realistici"""
    msg = RH56DFTPFeedback()
    now = time.time()
    sec = int(now)
    nsec = int((now - sec) * 1e9)
    msg.header = Header(stamp=Time(sec=sec, nanosec=nsec), frame_id='')

    # valori di esempio per posizione, angolo, forza, corrente ecc.
    msg.position = np.random.randint(0, 1000, 6, dtype=np.int16)
    msg.angle = np.random.randint(0, 1000, 6, dtype=np.int16)
    msg.force = np.random.randint(0, 1000, 6, dtype=np.int16)
    msg.current = np.random.randint(0, 1000, 6, dtype=np.int16)
    msg.error = np.random.randint(0, 2, 6, dtype=np.uint8)
    msg.status = np.random.randint(0, 2, 6, dtype=np.uint8)
    msg.temperature = np.random.randint(20, 50, 6, dtype=np.uint8)

    # Funzione helper per touch sensor
    def random_touch(size):
        return np.random.randint(0, 4096, size, dtype=np.int16)

    # Pinky
    msg.pinky_tip_touch = random_touch(9)
    msg.pinky_top_touch = random_touch(96)
    msg.pinky_palm_touch = random_touch(80)

    # Ring
    msg.ring_tip_touch = random_touch(9)
    msg.ring_top_touch = random_touch(96)
    msg.ring_palm_touch = random_touch(80)

    # Middle
    msg.middle_tip_touch = random_touch(9)
    msg.middle_top_touch = random_touch(96)
    msg.middle_palm_touch = random_touch(80)

    # Index
    msg.index_tip_touch = random_touch(9)
    msg.index_top_touch = random_touch(96)
    msg.index_palm_touch = random_touch(80)

    # Thumb
    msg.thumb_tip_touch = random_touch(9)
    msg.thumb_top_touch = random_touch(96)
    msg.thumb_middle_touch = random_touch(9)
    msg.thumb_palm_touch = random_touch(96)

    # Palm
    msg.palm_touch = random_touch(112)

    return msg

def main():
    # percorso della rosbag di output
    bag_path = os.path.expanduser('~/ros2_ws/fake_rh56_feedback_bag')

    # Remove old bag if exists
    if os.path.exists(bag_path):
        print(f"Removing existing bag folder: {bag_path}")
        shutil.rmtree(bag_path)

    # os.makedirs(bag_path)

    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions('', '')  # Humble / Iron accetta solo input/output serialization format

    writer = SequentialWriter()
    writer.open(storage_options, converter_options)

    topic_name = '/rh56_feedback'
    topic_type = 'hand_msgs/msg/RH56DFTPFeedback'
    metadata = TopicMetadata(name=topic_name, type=topic_type, serialization_format='cdr')
    writer.create_topic(metadata)

    print(f"Scrittura rosbag fittizia in {bag_path}...")

    # scrive 200 messaggi a 10 Hz
    for i in range(200):
        msg = create_fake_feedback()
        serialized = serialize_message(msg)
        timestamp_ns = int(msg.header.stamp.sec * 1e9 + msg.header.stamp.nanosec)
        writer.write(topic_name, serialized, timestamp_ns)
        time.sleep(0.1)

    print("Rosbag completata!")

if __name__ == "__main__":
    main()