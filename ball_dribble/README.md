# ball_dribble (ROS 2 Jazzy)

Stage-one F1TENTH ball dribbling project. The **real-car default controller is now Frenet MPC** for the Levine track. The previous Stanley implementation is retained and not started by default.

**Read [README_FRENET_MPC.md](README_FRENET_MPC.md) before real-car use.** It covers sensor requirements, 0.29 m contact depth, 100 mm release detection, reversing safety, installation, launch, tests and limitations.

## Main code

- `ball_dribble/frenet_mpc.py` — Frenet path generator + CVXPY/OSQP MPC (ROS-free)
- `ball_dribble/ball_frenet_mpc_node.py` — new ROS 2 real-car node
- `ball_dribble/ball_recovery.py` — stop, reverse, replan, approach, push supervisor
- `ball_dribble/ball_detector_node.py` — RealSense ball detector; publishes `/ball/position` and `/ball/camera_depth`
- `ball_dribble/ball_control.py` and `ball_dribble/ball_push_node.py` — original Stanley controller, preserved
- `ball_dribble/ball_sim_node.py` and `sim.py` — original simplified ball simulator; still Stanley-based

## Real car

```bash
cd ~/f1tenth_ws
colcon build --packages-select ball_dribble --symlink-install
source install/setup.bash
ros2 launch ball_dribble car.launch.py                    # motion disabled
# Only after safety checks/calibration:
ros2 launch ball_dribble car.launch.py enable_motion:=true
```

`car.launch.py` launches the detector + Frenet MPC; `stanley_car.launch.py` launches the original detector + Stanley controller if explicitly needed. **Never run both** since both publish `/drive`.

The original simulator workflow remains `ros2 launch ball_dribble sim.launch.py` (uses Stanley and requires f1tenth_gym_ros running separately).

Offline check: `python3 -m pytest -q test/`. The CVXPY smoke test is skipped if the solver is not installed.
