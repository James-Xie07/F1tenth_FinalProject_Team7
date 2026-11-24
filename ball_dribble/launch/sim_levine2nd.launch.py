"""Whole sim on the real Levine 2nd-floor map: f1tenth_gym_ros + ball_sim_node + ball_push_node.

    ros2 launch ball_dribble sim_levine2nd.launch.py [open_foxglove:=false] [max_speed:=0.8] [lat_accel_limit:=0.25]
Needs the setup done once with scripts/setup_levine2nd.py (maps/, paths/ball_path_levine2nd.csv, config/*_levine2nd.yaml).
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = get_package_share_directory('ball_dribble')
    gym = get_package_share_directory('f1tenth_gym_ros')
    sim_cfg = os.path.join(share, 'config', 'sim_levine2nd.yaml')
    return LaunchDescription([
        DeclareLaunchArgument('open_foxglove', default_value='false', description='open Foxglove in the browser'),
        DeclareLaunchArgument('max_speed', default_value='0.8', description='0.8 m/s: 5 of 5 clean laps; 1.0 m/s failed 1 of 2; the real car config (ball.yaml) uses 0.5'),
        DeclareLaunchArgument('lat_accel_limit', default_value='0.0', description='lateral acceleration cap in turns [m/s^2], 0 = off; with it max_speed is the straight-line speed'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(gym, 'launch', 'gym_bridge_launch.py')),
            launch_arguments={'config': os.path.join(share, 'config', 'gym_levine2nd.yaml'),
                              'map_path': os.path.join(share, 'maps', 'levine_2nd'),
                              'open_foxglove': LaunchConfiguration('open_foxglove')}.items()),
        Node(package='ball_dribble', executable='ball_sim_node', output='screen', parameters=[sim_cfg]),
        Node(package='ball_dribble', executable='ball_viz_node', output='screen',
             parameters=[{'waypoint_file': 'ball_path_levine2nd.csv'}]),          # big ball + path line for Foxglove, display only
        Node(package='ball_dribble', executable='ball_push_node', output='screen',
             parameters=[sim_cfg, {'max_speed': ParameterValue(LaunchConfiguration('max_speed'), value_type=float),
                                    'lat_accel_limit': ParameterValue(LaunchConfiguration('lat_accel_limit'), value_type=float)}]),
    ])
