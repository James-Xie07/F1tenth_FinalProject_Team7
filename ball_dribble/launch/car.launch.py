"""Real car: detector + controller. Start the RealSense (align_depth.enable:=true), the particle filter and the VESC first."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'), 'config', 'ball.yaml')
    params = LaunchConfiguration('params_file')
    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=cfg, description='Detector and controller YAML'),
        DeclareLaunchArgument('reacquire_enabled', default_value='false', description='Enable the directional search arc'),
        Node(package='ball_dribble', executable='ball_detector_node', output='screen', parameters=[params]),
        Node(package='ball_dribble', executable='ball_push_node', output='screen', parameters=[
            params, {'reacquire_enabled': ParameterValue(LaunchConfiguration('reacquire_enabled'), value_type=bool)}]),
    ])
