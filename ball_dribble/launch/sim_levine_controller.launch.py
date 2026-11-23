"""Launch ONLY the low-speed Frenet MPC controller for Gym levine_blocked.

Start sim_levine_world.launch.py in a separate terminal first.
Loads sim_frenet.yaml, never the real-car levine2 settings or Stanley.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'), 'config', 'sim_frenet.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('enable_motion', default_value='false'),
        Node(package='ball_dribble', executable='ball_frenet_mpc_node', output='screen',
             parameters=[cfg, {'enable_motion': ParameterValue(
                 LaunchConfiguration('enable_motion'), value_type=bool)}]),
    ])
