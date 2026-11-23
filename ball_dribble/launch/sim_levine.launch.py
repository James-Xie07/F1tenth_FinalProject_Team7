"""Optional all-in-one Levine-blocked Gym + ball + Frenet MPC launch.

For independent start/stop, use sim_levine_world.launch.py and
sim_levine_controller.launch.py instead.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    launch_dir = os.path.join(get_package_share_directory('ball_dribble'), 'launch')
    return LaunchDescription([
        DeclareLaunchArgument('enable_motion', default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'sim_levine_world.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(launch_dir, 'sim_levine_controller.launch.py')),
            launch_arguments={'enable_motion': LaunchConfiguration('enable_motion')}.items()),
    ])
