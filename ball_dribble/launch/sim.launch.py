"""Ball simulator + controller. Start f1tenth_gym_ros first (its own launch file), then:  ros2 launch ball_dribble sim.launch.py"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('ball_dribble'), 'config', 'sim.yaml')
    return LaunchDescription([
        Node(package='ball_dribble', executable='ball_sim_node', output='screen', parameters=[cfg]),
        Node(package='ball_dribble', executable='ball_push_node', output='screen', parameters=[cfg]),
    ])
