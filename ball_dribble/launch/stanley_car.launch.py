"""Real car: detector + controller. Start the RealSense (align_depth.enable:=true), the particle filter and the VESC first."""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'), 'config', 'ball.yaml')
    return LaunchDescription([
        Node(package='ball_dribble', executable='ball_detector_node', output='screen', parameters=[cfg]),
        Node(package='ball_dribble', executable='ball_push_node', output='screen', parameters=[cfg]),
    ])
