"""Levine real-car Frenet MPC + RealSense detector (Stanley is NOT launched)."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'),'config','frenet.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('enable_motion',default_value='false',
                              description='Keep false until calibration and safety checks pass'),
        Node(package='ball_dribble',executable='ball_detector_node',output='screen',parameters=[cfg]),
        Node(package='ball_dribble',executable='ball_frenet_mpc_node',output='screen',
             parameters=[cfg, {'enable_motion': ParameterValue(LaunchConfiguration('enable_motion'),value_type=bool)}]),
    ])
