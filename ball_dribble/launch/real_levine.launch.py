"""Explicit real-car Levine2 launch. Equivalent to the existing car.launch.py.
Requires the real /map (levine2), PF, VESC and RealSense drivers to be running.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'), 'config', 'frenet.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('enable_motion', default_value='false',
                              description='Enable ONLY after sensor, safety, and map checks'),
        Node(package='ball_dribble', executable='ball_detector_node', output='screen',
             parameters=[cfg]),
        Node(package='ball_dribble', executable='ball_frenet_mpc_node', output='screen',
             parameters=[cfg, {'enable_motion': ParameterValue(
                 LaunchConfiguration('enable_motion'), value_type=bool)}]),
    ])
