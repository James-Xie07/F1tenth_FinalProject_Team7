# Ubuntu Levine Frenet MPC simulation

This simulation uses `f1tenth_gym_ros` for the vehicle, LiDAR and occupancy map, and `ball_sim_node` for the ball. It launches **Frenet MPC** and keeps Stanley installed but inactive. The simulated `/ball/camera_depth` is calibrated to read 0.29 m at the simulated scoop contact location. This is **not a real D435i depth simulation**.

## Install and build

```bash
source /opt/ros/jazzy/setup.bash
cd /home/zhenbang/roboracer_ws
python3 -c 'import numpy, scipy, cvxpy, osqp; print("Solver available")'
# If missing, use the packages available for your Ubuntu/ROS Python environment.
colcon build --packages-select ball_dribble --symlink-install
source install/setup.bash
```

Confirm `ros2 pkg executables ball_dribble` includes `ball_frenet_mpc_node`.

## Terminal 1: start the existing gym simulator

```bash
source /opt/ros/jazzy/setup.bash
source /home/zhenbang/sim_ws/install/setup.bash
ros2 launch f1tenth_gym_ros gym_bridge_launch.py map_path:=levine_blocked
```

The gym map must use **the same metric coordinates** as `levine_ball_path.csv`. The included package does not include the Levine occupancy-map image or YAML. Check that `/map`, `/scan` and `/ego_racecar/odom` exist, and that the odometry `header.frame_id` is `map`. If it is `odom` or another frame, do not just overwrite it: establish the transform into `map` first.

## Terminal 2: Frenet MPC + ball (stationary first)

```bash
source /opt/ros/jazzy/setup.bash
source /home/zhenbang/sim_ws/install/setup.bash
source /home/zhenbang/roboracer_ws/install/setup.bash
ros2 launch ball_dribble sim_frenet.launch.py
```

Confirm `/ball/position` and `/ball/camera_depth` produce messages. Start another fresh launch after checking the map path matches and there is exactly one `/drive` publisher:

```bash
ros2 launch ball_dribble sim_frenet.launch.py enable_motion:=true
```

Use `Ctrl+C` to stop the stationary launch before restarting with motion enabled. Never run both launches at once.

## Terminal 3: status and visualisation

```bash
ros2 topic echo /ball_frenet_mpc_node/mode
ros2 topic echo /ball/camera_depth
ros2 topic hz /ego_racecar/odom
ros2 topic info /drive -v
```

In RViz, add `Path` displays for:

* `/ball_frenet_mpc_node/reference_path`
* `/ball_frenet_mpc_node/predicted_path`

Set Fixed Frame = `map`, display `/ball_sim_node/marker`, and use the `Publish Point` tool on `/clicked_point` to move the ball sideways to test recovery. Note the ball marker has an enlarged visible size and does not represent the physical ball radius.

The recovery state machine will stop, then reverse at most 0.9 m and attempt approach when the simulated map and sensors permit. The simulator does **not** validate reverse LiDAR coverage or actual hardware braking. This simulation sets `require_rear_lidar: false` solely for testing. Real-car configuration still requires confirmed rear coverage.

If stuck in `HOLD`, inspect console warnings and issue:

```bash
ros2 param set /ball_frenet_mpc_node reset_recovery true
ros2 param set /ball_frenet_mpc_node reset_recovery false
```

## Common reasons for no movement

* `DISABLED` means use `enable_motion:=true` after the stationary check.
* `NO_ODOMETRY` means `/ego_racecar/odom` is missing, stale, or not in the `map` frame.
* `WAIT_BALL` means the simulator camera cannot see the ball or synthetic depth is outside the contact tolerance.
* `PUSH` but still stopped means a candidate trajectory was rejected, often because `/map` or `/scan` is missing or the Levine waypoint coordinates do not match the map.
* Do not run `sim.launch.py` for Frenet testing: it starts the legacy Stanley controller.
* This configuration assumes the Gym odometry origin is approximately 0.17 m ahead of the rear axle. If your Gym odometry is already at the rear axle, set `odom_offset_x: 0.0` under **both** `ball_sim_node` and `ball_frenet_mpc_node` in `config/sim_frenet.yaml`, then rebuild/relaunch.

## Why `No safe Frenet path`?

This diagnostic-only build reports a dictionary after that warning.
`rejected` reports counts among 7 candidates in check order: curvature first,
map wall second, then LiDAR obstacle. `lidar_obstacle_points` reports projected
laser returns that were not close to known map walls or the known ball position.
`current_map_body_gap_m` is the distance from the current estimated body outline
to the map obstacles. Positive means clear; negative suggests overlap.

Do **not** remove map safety checks or increase speed to force a plan. The simulated
ball is separate from Gym and is not inserted into Gym's `/scan`.
