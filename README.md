# F1TENTH Team 7 — Levine Ball Dribbling (Frenet MPC)

ROS 2 Jazzy project for F1TENTH ball dribbling. The same Frenet MPC code supports a real car on **levine2** and a Gym simulation on **levine_blocked**. An older Stanley controller is retained as a reference, but it is not used by the default Frenet launches.

The `ball_dribble/` folder is an `ament_python` ROS 2 package.

## Build

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
cd ~/roboracer_ws
colcon build --packages-select ball_dribble --symlink-install
source install/setup.bash
```

Use your existing ROS 2 dependencies, including NumPy, SciPy, CVXPY and OSQP. The Gym simulator lives in a **separate** workspace (`~/sim_ws`).

## Gym simulation on Levine (CCW)

The world uses the simulator's `levine_blocked` map and its matching CCW waypoints. The MPC uses `ball_dribble/paths/levine_ccw_raceline.csv`; these are distinct from the real car's `levine2` track and waypoints.

One-time setup for the Gym raceline:

```bash
cd ~/roboracer_ws/src/F1tenth_FinalProject_Team7-stage1-dribble-FrenetMPC/ball_dribble
python3 scripts/prepare_gym_levine.py --install
```

Start the simulator and the controller in **separate terminals**. Source the ROS 2, simulator and project workspaces in each terminal.

Simulator + simulated tennis ball, **without** a controller:

```bash
ros2 launch ball_dribble sim_levine_world.launch.py
```

Frenet MPC controller, **without** Gym:

```bash
ros2 launch ball_dribble sim_levine_controller.launch.py enable_motion:=true
```

Omit `enable_motion:=true` to start with motion disabled. Do not run the older all-in-one launch simultaneously.

## Real car

The real-car launch uses the `levine2` map coordinate system and `ball_dribble/paths/levine_ball_path.csv` (ball-center reference), with particle-filter localization, LiDAR, VESC and the RealSense camera running separately.

```bash
ros2 launch ball_dribble real_levine.launch.py
```

Motion is disabled by default. Enable only after checking sensor frames, collision safety, drive topic ownership, and the physical vehicle response. **Do not use simulator-only recovery settings on the real vehicle.**

## Diagnostics

```bash
ros2 topic echo /ball_frenet_mpc_node/mode
ros2 topic echo /ball_frenet_mpc_node/recovery_info
ros2 topic info /drive -v
```

See [ball_dribble/README_SPLIT_SIM_RECOVERY.md](ball_dribble/README_SPLIT_SIM_RECOVERY.md) and [ball_dribble/README_DUAL_MODE.md](ball_dribble/README_DUAL_MODE.md) for behavior, installation, and limitations.

## Status

Offline recovery, configuration, and launch tests were run. Full simulator and real-car validation of the latest changes remain in progress.
