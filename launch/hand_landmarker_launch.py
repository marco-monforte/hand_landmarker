from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    # ---------------------------
    # Declare launch arguments
    # ---------------------------
    model_path_arg = DeclareLaunchArgument(
        'model_path',
        default_value='/root/ros2_ws/src/hand_landmarker/models/hand_landmarker.task',
        description='Path to the hand landmarker model'
    )

    debug_arg = DeclareLaunchArgument(
        'debug',
        default_value='False',
        description='Enable debug mode for hand landmarker node'
    )

    publish_rate_arg = DeclareLaunchArgument(
        'publish_rate',
        default_value='60.0',
        description='Publish rate for the hand landmarker node'
    )

    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/color/image_raw',
        description='Camera topic to subscribe to'
    )

    controlled_hand_arg = DeclareLaunchArgument(
        'controlled_hand',
        default_value='left',
        description='Hand to visualize in the visualizer'
    )

    ema_alpha_arg = DeclareLaunchArgument(
        'ema_alpha',
        default_value='0.3',
        description='EMA alpha for visualizer smoothing'
    )

    landmark_timeout_arg = DeclareLaunchArgument(
        'landmark_timeout',
        default_value='0.1',
        description='Timeout for landmarks in visualizer'
    )

    feedback_topic_arg = DeclareLaunchArgument(
        'feedback_topic',
        default_value='/rh56dftp/feedback',
        description='Feedback topic from the hand device'
    )

    # ---------------------------
    # Define nodes
    # ---------------------------
    hand_landmarker_node = Node(
        package='hand_landmarker',
        executable='hand_landmarker_node.py',
        name='hand_landmarker_node',
        output='screen',
        parameters=[{
            'model_path': LaunchConfiguration('model_path'),
            'debug': LaunchConfiguration('debug'),
            'publish_rate': LaunchConfiguration('publish_rate'),
            'camera_topic': LaunchConfiguration('camera_topic')
        }]
    )

    hand_visualizer_node = Node(
        package='hand_landmarker',
        executable='hand_landmarks_visualizer',
        name='hand_landmarks_visualizer',
        output='screen',
        parameters=[{
            'controlled_hand': LaunchConfiguration('controlled_hand'),
            'ema_alpha': LaunchConfiguration('ema_alpha'),
            'landmark_timeout': LaunchConfiguration('landmark_timeout'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'feedback_topic': LaunchConfiguration('feedback_topic')
        }]
    )

    # ---------------------------
    # LaunchDescription
    # ---------------------------
    return LaunchDescription([
        model_path_arg,
        debug_arg,
        publish_rate_arg,
        camera_topic_arg,
        controlled_hand_arg,
        ema_alpha_arg,
        landmark_timeout_arg,
        feedback_topic_arg,
        hand_landmarker_node,
        hand_visualizer_node
    ])