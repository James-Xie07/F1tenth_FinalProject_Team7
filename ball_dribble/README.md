# ball_dribble — stage 1: push a ball around the track (F1TENTH, ROS 2 Jazzy)

Layout
    ball_dribble/   ball_control.py (controller, ROS-free) physics.py sim.py track.py
                    ball_push_node.py ball_detector_node.py   (ROS nodes)
    scripts/        make_ball_path.py  calibrate_ball.py  experiments.py
    paths/          ball_path.csv (demo, race2 map)  race2_track.npz
    maps/ launch/ config/   (empty; filled in Parts B/C)
    test/           test_core.py (pytest, no ROS)   mock_ros_test.py (node wiring with stub ROS)

Offline check (no ROS):   python3 -m pytest test/   &&   python3 test/mock_ros_test.py
Build on the car/VM:      cd ~/roboracer_ws && colcon build --packages-select ball_dribble --symlink-install && source install/setup.bash
Run:                      ros2 run ball_dribble ball_push_node --ros-args -p waypoint_file:=ball_path.csv -p max_speed:=0.6

Work on a feature branch and open a PR; do not push to main.
Status: sim-only so far. Never run on ROS or hardware. See the parameters in ball_dribble/ball_control.py (Config).
