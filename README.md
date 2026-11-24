# F1TENTH Team 7 — Frank's dribbling controller

The `frank` branch contains Frank's ROS 2 Jazzy ball-dribbling package under
[`ball_dribble/`](ball_dribble/).

- The real-car launch is `ros2 launch ball_dribble car.launch.py`. It starts the
  detector and controller with `config/ball.yaml`. The profiles use MPC while
  dribbling and Stanley for chase and approach. Start the vehicle drivers and
  particle-filter localization separately.
- The Levine second-floor simulation launch is
  `ros2 launch ball_dribble sim_levine2nd.launch.py`.
- `config/ball_vnarrow.yaml` provides an alternate narrow-V pusher profile.
  Select it with `params_file:=/absolute/path/to/ball_vnarrow.yaml`.

### Directional ball reacquisition

Try the optional search arc in the Levine simulation:

```bash
ros2 launch ball_dribble sim_levine2nd.launch.py reacquire_enabled:=true
ros2 topic echo /ball_push_node/mode
```

Place a visible ball off to one side, then move it just past the camera's field
of view on that side. After losing the ball, the controller first settles,
then may try one forward arc toward the last accepted observation. Its default
speed is 0.12 m/s with a 0.30 m travel budget. It checks the mapped clearance
of the car and scoop and only starts an arc that improves the predicted bearing
to the last observation. The mode reads `REACQUIRE` during the arc. A fresh
detection returns control to the normal state machine. Without a fresh
detection the attempt ends in `SEARCH` and will not repeat.

Both launches default to `reacquire_enabled:=false`. The same flag opts in on
the real-car launch. The arc uses the occupancy map for wall clearance and
does not check live LiDAR obstacles. It stays stopped without a usable map or
recent off-center observation. The `reacquire_camera_x/y` settings match
the horizontal camera offsets in the corresponding real-car profile.

### Ball missing from the detector

Check `/ball_detector_node/debug_image` while the ball is seated and as it moves
left and right. The overlay distinguishes a color candidate touching the image
boundary, a candidate with invalid depth, and no compact color candidate.
Watch `/ball/position` to check which frames have usable 3D detections.

If the ball is clipped by the left or right edge, try the optional
`allow_clipped_ball:=true` argument on `car.launch.py`. A clipped candidate
needs a visible arc from approximately half the ball and a depth consistent
with a tennis ball. Move or re-aim the camera if most of the ball is outside
the frame or the scoop blocks it. Keep `cam_y` and `reacquire_camera_y` aligned
with the measured lateral offset when changing the mount.

See the [package README](ball_dribble/README.md) for the original project notes.
