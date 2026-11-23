"""Frenet MPC plus synthetic ball/depth for f1tenth_gym_ros Levine map.

Start gym_bridge_launch.py map_path:=levine_blocked in another terminal.
Do not start legacy sim.launch.py at the same time (it runs Stanley).
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
        DeclareLaunchArgument('enable_motion', default_value='false',
                              description='Start stationary and explicitly opt in to simulated driving'),
        Node(package='ball_dribble', executable='ball_sim_node', output='screen', parameters=[cfg]),
        Node(package='ball_dribble', executable='ball_frenet_mpc_node', output='screen',
             parameters=[cfg, {'enable_motion': ParameterValue(LaunchConfiguration('enable_motion'), value_type=bool)}]),
    ])
