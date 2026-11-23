# Levine Frenet MPC: real-car levine2 + Ubuntu Gym levine_blocked

The SAME `ball_frenet_mpc_node` and MPC implementation operate in both environments.
Neither launch uses the retained Stanley implementation.

| | Physical F1TENTH | Ubuntu Gym |
|---|---|---|
| Map | physical `/map` from `levine2.yaml` | `levine_blocked` in Gym |
| Input CSV | `levine_ball_path.csv` | `levine_ccw_raceline.csv` |
| `reference_is_ball_path` | `true` (shift ball centre back 0.54 m) | `false` (CSV is already vehicle reference) |
| Pose | `/pf/pose/odom` plus `/odom` speed | `/ego_racecar/odom` |
| Ball | D435i ROS detector | ball physics/depth simulation |
| Safety | Rear LiDAR visibility required before reverse | Map-only rear validation, **simulation only** |
| Speed limit | 0.50 m/s | 0.50 m/s |

The real-car configuration `config/frenet.yaml`, original `launch/car.launch.py`,
and real-car `levine_ball_path.csv` are preserved. A named real-car alias
`real_levine.launch.py` is provided for clarity. The simulator's launch is
`sim_levine.launch.py`, which starts the simulator plus ball and MPC in one call.

## Installation (Ubuntu)

If replacing an earlier package, first back up the existing directory. Extract
this zip in `~/roboracer_ws/src/` and ensure the folder contains `ball_dribble/setup.py`.

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
cd ~/roboracer_ws
python3 -c 'import numpy, scipy, cvxpy, osqp; print("MPC dependencies OK")'
colcon build --packages-select ball_dribble --symlink-install
source install/setup.bash
```

### Prepare the Gym track ONCE

The installed `f1tenth_gym` version expects a named track directory and a
seven-column semicolon-separated raceline CSV with two comment rows, a header
on line 3, and a repeated closing point. Its previous `levine_blocked` CSV
was generated from physical-map waypoints and must NOT be reused in simulation.

From the package directory:

```bash
cd ~/roboracer_ws/src/F1tenth_FinalProject_Team7-stage1-dribble-FrenetMPC/ball_dribble
python3 scripts/prepare_gym_levine.py                         # dry-run
python3 scripts/prepare_gym_levine.py --install                # backs up old CSV
```

If you kept a different top-level folder name, change only the `cd` path.
The helper **does not alter real `levine2` files**. It creates symlinks to
`~/sim_ws/src/f1tenth_gym_ros/maps/levine_blocked.{png,yaml}` and a corrected
Gym `levine_blocked_raceline.csv` derived from the 1,225-point simulator CCW
vehicle path. The helper refuses to overwrite a map link pointing to an unknown
source; inspect the directory manually if that happens.

**IMPORTANT:** This supplied simulator raceline has not yet been verified against
*your exact* `levine_blocked.png` and YAML. You MUST check its position against
`/map` in Foxglove. Map name alone is insufficient evidence of alignment.

### Start simulation with ONE launch

Open a fresh terminal:

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
source ~/roboracer_ws/install/setup.bash
ros2 launch ball_dribble sim_levine.launch.py
```

Motion is disabled by default, even though Gym, the synthetic ball, and the
Frenet MPC node are running. Do not separately start another Gym or standalone
`sim_frenet.launch.py` at the same time, or you may create duplicate publishers.

In a second terminal, check alignment while the simulator is running:

```bash
source /opt/ros/jazzy/setup.bash
source ~/sim_ws/install/setup.bash
source ~/roboracer_ws/install/setup.bash
python3 ~/roboracer_ws/src/F1tenth_FinalProject_Team7-stage1-dribble-FrenetMPC/ball_dribble/scripts/check_levine_map_alignment.py \
  --csv ~/roboracer_ws/src/F1tenth_FinalProject_Team7-stage1-dribble-FrenetMPC/ball_dribble/paths/levine_ccw_raceline.csv
ros2 topic info /drive -v
ros2 topic echo /ego_racecar/odom --once
```

In Foxglove, enable `/map`, `/ball_frenet_mpc_node/reference_path`, and
`/ball_frenet_mpc_node/predicted_path`. For a vehicle *anywhere* along the CCW
raceline, localization should give small lateral and heading errors. The car
need not start at the FIRST waypoint. Set its initial pose on the **simulation**
track, not the previous `(10.125, -37.314)` physical-map coordinate.

When alignment is validated, stop the stationary launch using Ctrl+C, then:

```bash
ros2 launch ball_dribble sim_levine.launch.py enable_motion:=true
```

### Real F1TENTH launch (unchanged behavior)

Source your real-car ROS 2 workspace, start the real `levine2` map server,
Particle Filter, VESC, D435i color+aligned depth, LiDAR, and required transforms.
The launch does not start those external drivers for you.

```bash
ros2 launch ball_dribble real_levine.launch.py
# Only after safety checks:
ros2 launch ball_dribble real_levine.launch.py enable_motion:=true
```

The existing `car.launch.py` is equivalent to `real_levine.launch.py`.
`stanley_car.launch.py` is preserved for manual legacy experiments, and never
runs as part of the commands above.

### Current limitations

* The new Gym raceline's coordinate alignment still needs a check on your local map.
* The Python CVXPY/OSQP formulation previously reported an unstable sparse matrix
  update and a non-DPP warning. Simulation uses `solver_warm_start: false` as a
  temporary workaround. Real mode retains `solver_warm_start: true`, matching
  the prior configuration; this needs a verified solver fix and timing test
  before real-car operation.
* `/drive` previously showed a DURABILITY QoS incompatibility warning. Check
  `ros2 topic info /drive -v`; confirm Gym is actually receiving commands.
* `odom_offset_x=0.17` is an estimate for Gym; verify the frame origin.
* The simulated ball is separate from Gym. It is NOT inserted into LiDAR scans.
* Reverse safety in the simulator does not prove that the real car can safely
  reverse. The real configuration still requires verified rear coverage.
