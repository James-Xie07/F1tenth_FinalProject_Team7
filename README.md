# F1TENTH Team 7 — Frank's Stanley controller

The `frank` branch contains Frank's ROS 2 Jazzy ball-dribbling package under
[`ball_dribble/`](ball_dribble/).

- The real-car launch is `ros2 launch ball_dribble car.launch.py`. It starts the
  detector and Stanley controller with `config/ball.yaml`. Start the vehicle
  drivers and particle-filter localization separately.
- The Levine second-floor simulation launch is
  `ros2 launch ball_dribble sim_levine2nd.launch.py`.
- `config/ball_vnarrow.yaml` provides an alternate narrow-V pusher profile.
  The default real-car launch loads `config/ball.yaml`.

See the [package README](ball_dribble/README.md) for the original project notes.
