# Levine Frenet MPC: separated simulation + recontact fix

## What was changed

A log showed `DIRECT_APPROACH -> PUSH` followed 0.139 s later by `PUSH -> STOP`, and then `STOP -> HOLD`. The original logic returned to PUSH after **one** camera sample within the 0.29 m contact tolerance. A single subsequent out-of-range depth sample triggered an immediate stop; if the ball was close enough that no direct forward path could be generated, HOLD was permanent.

The simulated `channel` scoop places the touching ball center at about **0.5081 m** from the rear axle (`make_scoop` + `rider_x`). The controller had used 0.540 m as its approach reference, which created a near-ball interval where camera depth reported lost contact but `direct_forward_curve` declined to move. The **simulation-only** value now matches 0.5081 m.

Simulator-specific recovery settings in `config/sim_frenet.yaml`:

- `contact_confirm_time: 0.15` requires a short continuous run of camera samples before switching to `PUSH`.
- `contact_loss_grace: 0.30` ignores isolated depth-only dropouts in `PUSH`. A lateral escape exceeding 0.10 m still stops immediately. Missing/stale ball data still stops.
- `retry_depth_hold: true` allows a stopped controller to try a newly feasible **forward** approach to a visible ball after a depth-only loss. It never retries a lateral escape or reverses blindly.
- `max_speed: 0.5`, `allow_direct_approach: true`, `ball_center_ahead: 0.5081`.

Real vehicle `config/frenet.yaml` and `real_levine.launch.py` remain unchanged. Real-car release and rear-sensing safety continue to be conservative. The Python entry point and Stanley implementation remain present.

## Prepare Gym map and waypoints once

The simulator must use the **simulator** Levine track `levine_blocked`, with `levine_ccw_raceline.csv` (CCW) converted to Gym's required `levine_blocked_raceline.csv`. It must never substitute the real `levine2` and `levine_ball_path.csv`.

```bash
cd ~/roboracer_ws/src/F1tenth_FinalProject_Team7-stage1-dribble-FrenetMPC/ball_dribble
python3 scripts/prepare_gym_levine.py --install
```

The preparation script backs up an existing installed Gym raceline, writes a three-row header, closes the loop, and uses the existing `levine_blocked.png/.yaml`. The **world launch checks** that the installed Gym raceline matches the 1,225 CCW simulation waypoints before starting.

## Build

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
cd ~/roboracer_ws
colcon build --packages-select ball_dribble --symlink-install
source install/setup.bash
```

## Run independently in two terminals

Terminal 1: **Gym + ball physics/synthetic camera only. No controller.**

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 launch ball_dribble sim_levine_world.launch.py
```

Terminal 2: **Frenet MPC only. No Gym and no ball simulator.**

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 launch ball_dribble sim_levine_controller.launch.py enable_motion:=true
```

For a stationary checkout, omit `enable_motion:=true`. You may stop/restart the controller without restarting Gym; do **not** start `sim_levine.launch.py` or `sim_frenet.launch.py` at the same time, since those older launches start overlapping nodes. The older all-in-one `sim_levine.launch.py` is retained for convenience.

Check:

```bash
ros2 topic echo /ball_frenet_mpc_node/mode
ros2 topic echo /ball_frenet_mpc_node/recovery_info
ros2 topic info /drive -v
```

In Foxglove, avoid publishing zero speed to `/drive` while the MPC is running. It creates another controller publisher. To temporarily disable the MPC itself, run `ros2 param set /ball_frenet_mpc_node enable_motion false` and later `... true`. Note: this resets the recovery state when re-enabled, so it is different from injecting a transient zero command.

## Real vehicle

Keep the independent real-car launch:

```bash
ros2 launch ball_dribble real_levine.launch.py
```

It requires the real `levine2` map, PF odometry, VESC odometry, LiDAR, and RealSense. Never use `sim_frenet.yaml` on the physical car: simulator mode bypasses rear-LiDAR verification for reversing.

## Verification and limitations

Offline state-machine, geometry, CSV and launch checks passed. Full ROS 2 / F1TENTH Gym end-to-end testing was not available in the editing environment. The simulator ball and contact detector use a simplified model; camera contact confirmation is still not equivalent to a force/physical contact switch. Forward motion remains dependent on fresh perception, collision checking, and a feasible MPC solution. Check recovery_info if it stops again.
