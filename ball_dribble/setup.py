from glob import glob
from setuptools import find_packages, setup

package_name = 'ball_dribble'
setup(
    name=package_name, version='0.1.0', packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/paths', glob('paths/*')),
        ('share/' + package_name + '/maps', glob('maps/*')),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'ball_push_node = ball_dribble.ball_push_node:main',  # retained Stanley implementation
        'ball_frenet_mpc_node = ball_dribble.ball_frenet_mpc_node:main',
        'ball_detector_node = ball_dribble.ball_detector_node:main',
        'ball_video_web_node = ball_dribble.ball_video_web_node:main',
        'ball_sim_node = ball_dribble.ball_sim_node:main',
    ]},
)
