# ball_dribble — stage 1: push a ball around the track (F1TENTH, ROS 2 Jazzy)

Layout
    ball_dribble/   ball_control.py (controller, ROS-free) physics.py sim.py track.py
                    ball_push_node.py ball_detector_node.py   (ROS nodes)
    scripts/        make_ball_path.py  calibrate_ball.py  experiments.py
    paths/          ball_path.csv (demo, race2 map)  race2_track.npz
    maps/ launch/ config/   (empty; filled in Parts B/C)
    test/           test_core.py (pytest, no ROS)   mock_ros_test.py (node wiring with stub ROS)

Simulation (Part B)
    1. start f1tenth_gym_ros with the map you want (its own launch file)
    2. ros2 launch ball_dribble sim.launch.py        (ball_sim_node + ball_push_node, config/sim.yaml)
    3. in rviz add Marker topics /ball_sim_node/marker (true ball) and /ball_push_node/ball (estimate); "Publish Point" moves the ball
    The ball path (waypoint_file) must be built from the same map: python3 scripts/make_ball_path.py  (see its header)
Real car (Part C):  ros2 launch ball_dribble car.launch.py   (ball_detector_node + ball_push_node, config/ball.yaml)

The supplied real-car and simulation launch profiles select `push_controller: mpc`.
MPC follows the ball waypoint file using the active scoop geometry and Frank's
speed profile. It uses the main-branch tracking weights, a one-second prediction
window (20 steps at the default 50 ms callback period), and a 0.50 m/s PUSH cap
or the active profile's lower speed limit. The node uses the existing topics
and the same APPROACH, CHASE and SEARCH routines. The `ball_vnarrow.yaml` profile
still requires explicitly loading it in place of `ball.yaml`.

The MPC profile requires CVXPY with OSQP available in the ROS Python environment.
For a comparison run, use `push_controller: stanley` in the loaded YAML.

Offline check (no ROS):   python3 -m pytest test/   &&   python3 test/mock_ros_test.py
Build on the car/VM:      cd ~/roboracer_ws && colcon build --packages-select ball_dribble --symlink-install && source install/setup.bash
Run:                      ros2 run ball_dribble ball_push_node --ros-args -p push_controller:=mpc -p waypoint_file:=ball_path.csv -p max_speed:=0.5

Work on a feature branch and open a PR; do not push to main.
Status: sim-only so far. Never run on ROS or hardware. See the parameters in ball_dribble/ball_control.py (Config).
