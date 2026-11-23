"""Explicit real-car Levine2 launch. Equivalent to the existing car.launch.py.
Requires the real /map (levine2), PF, VESC and RealSense drivers to be running.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'), 'config', 'frenet.yaml')
    foxglove_topics = [
        r'/tf', r'/tf_static', r'/clock', r'/map', r'/scan', r'/odom',
        r'/pf/pose/odom', r'/drive', r'/ball/position', r'/ball/camera_depth',
        r'/ball_frenet_mpc_node/.*', r'/parameter_events', r'/rosout',
    ]
    return LaunchDescription([
        DeclareLaunchArgument('enable_motion', default_value='false',
                              description='Enable ONLY after sensor, safety, and map checks'),
        DeclareLaunchArgument('start_web_video', default_value='true',
                              description='Serve compressed detector video over HTTP'),
        DeclareLaunchArgument('web_video_port', default_value='8080'),
        DeclareLaunchArgument('start_foxglove_bridge', default_value='true',
                              description='Start a filtered bridge that excludes every camera image topic'),
        DeclareLaunchArgument('foxglove_port', default_value='8765'),
        # Use the system NumPy that matches OpenCV for the camera processes.
        Node(package='ball_dribble', executable='ball_detector_node', output='screen',
             additional_env={'PYTHONNOUSERSITE': '1'},
             parameters=[cfg]),
        Node(package='ball_dribble', executable='ball_video_web_node', output='screen',
             condition=IfCondition(LaunchConfiguration('start_web_video')),
             additional_env={'PYTHONNOUSERSITE': '1'},
             parameters=[cfg, {'port': ParameterValue(
                 LaunchConfiguration('web_video_port'), value_type=int)}]),
        Node(package='ball_dribble', executable='ball_frenet_mpc_node', output='screen',
             parameters=[cfg, {'enable_motion': ParameterValue(
                 LaunchConfiguration('enable_motion'), value_type=bool)}]),
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             output='screen', condition=IfCondition(LaunchConfiguration('start_foxglove_bridge')),
             parameters=[{
                 'port': ParameterValue(LaunchConfiguration('foxglove_port'), value_type=int),
                 'address': '0.0.0.0',
                 'topic_whitelist': foxglove_topics,
                 'best_effort_qos_topic_whitelist': [
                     r'/scan', r'/odom', r'/pf/pose/odom', r'/ball/.*'],
                 'max_qos_depth': 2,
             }]),
    ])
