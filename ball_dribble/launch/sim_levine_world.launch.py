"""Levine-blocked Gym vehicle + ball simulator; no controller is launched.

Prepare the Gym raceline once using scripts/prepare_gym_levine.py --install.
Start the MPC separately with sim_levine_controller.launch.py.
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def _verify_sim_track(context, *args, **kwargs):
    """Refuse to start with the old real-map CSV accidentally installed in Gym."""
    from pathlib import Path
    import importlib.util
    import numpy as np
    spec = importlib.util.find_spec('f1tenth_gym.envs.track.utils')
    if spec is None or spec.origin is None:
        raise RuntimeError('Source ~/sim_ws/install/setup.bash before launching the Gym world.')
    gym_repo = Path(spec.origin).resolve().parents[3]
    track = gym_repo / 'maps' / 'levine_blocked'
    path = track / 'levine_blocked_raceline.csv'
    png = track / 'levine_blocked.png'
    yaml = track / 'levine_blocked.yaml'
    pkg = Path(get_package_share_directory('ball_dribble'))
    reference = pkg / 'paths' / 'levine_ccw_raceline.csv'
    if not all(f.exists() for f in (path,png,yaml,reference)):
        raise RuntimeError(
            'Gym levine_blocked assets missing. Run: '
            'cd ~/roboracer_ws/src/F1tenth_FinalProject_Team7-stage1-dribble-FrenetMPC/ball_dribble '
            '&& python3 scripts/prepare_gym_levine.py --install')
    src = np.genfromtxt(reference, delimiter=',', names=True)
    try:
        installed = np.loadtxt(path, delimiter=';', skiprows=3)
    except Exception as exc:
        raise RuntimeError(f'Cannot parse Gym levine_blocked raceline: {exc}') from exc
    if installed.ndim != 2 or installed.shape[0] != len(src)+1 or installed.shape[1] != 7:
        raise RuntimeError('Gym raceline does not match the simulation CCW file. '
                           'Run scripts/prepare_gym_levine.py --install again.')
    if not np.allclose(installed[:-1,1:3], np.column_stack((src['x'],src['y'])), atol=1e-5):
        raise RuntimeError('Gym raceline currently has different waypoints than '
                           'levine_ccw_raceline.csv. Run scripts/prepare_gym_levine.py --install.')
    if not np.allclose(installed[0,1:3], installed[-1,1:3], atol=1e-6):
        raise RuntimeError('Gym raceline must be closed. Run scripts/prepare_gym_levine.py --install.')
    return []


def generate_launch_description():
    gym = os.path.join(get_package_share_directory('f1tenth_gym_ros'),
                       'launch', 'gym_bridge_launch.py')
    cfg = os.path.join(get_package_share_directory('ball_dribble'),
                       'config', 'sim_frenet.yaml')
    return LaunchDescription([
        OpaqueFunction(function=_verify_sim_track),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gym),
            launch_arguments={'map_path': 'levine_blocked'}.items()),
        Node(package='ball_dribble', executable='ball_sim_node', output='screen', parameters=[cfg]),
    ])
