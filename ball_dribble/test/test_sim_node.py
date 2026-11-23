"""ball_sim_node with stub ROS: a car driving straight at 1 m/s must carry the ball in the scoop, and the node must publish
detections in the car frame once the ball is outside the camera blind zone."""
import sys, math, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1])); sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np
import mock_ros_test as m
from mock_ros_test import ns, Clock, odom
import ball_dribble.ball_sim_node as bsn


def open_walls():
    class W:
        def dist(self, x, y): return np.full(np.shape(x), 5.0)
        def grad(self, x, y): return np.zeros(np.shape(x)), np.zeros(np.shape(x))
    return W()


def test_ball_rides_in_scoop_and_is_seen_after_release():
    node = bsn.BallSimNode(); node.walls = open_walls()
    det = node.pubs['/ball/position'].out
    gym_off = 0.17                                           # the gym odom point is ahead of the rear axle
    x = 0.0; v = 1.0; dt = 0.005; t = 0.0; ball_x = []
    for k in range(int(3.0 / dt)):
        t = k * dt; Clock.t = t; x = v * t                   # rear axle at x on the x-axis, heading 0
        if k % 2 == 0:
            node.odom_callback(odom(t, x + gym_off, 0.0, 0.0, v))
        node.physics_callback()
        if k % 7 == 0: node.camera_callback()
        ball_x.append(node.ball[0] - x)
    # (the car starts at 1 m/s so the first touch kicks the ball ahead; it settles back into the scoop)
    # in the scoop the ball stays at the same place relative to the car (centre ~0.5 m ahead of the rear axle)
    assert abs(ball_x[-1] - ball_x[-100]) < 0.02, ball_x[-1]
    assert 0.4 < ball_x[-1] < 0.6, ball_x[-1]
    assert node.touch
    # release the ball far ahead (click): detections must appear, ~ 2.0 m ahead in the car frame
    node.clicked_callback(ns(point=ns(x=x + 2.0, y=0.0)))
    for k in range(60):
        Clock.t = t + 0.01 * (k + 1); node.camera_callback()
    assert det, 'no detections published'
    last = det[-1]
    assert last.header.frame_id == 'rear_axle' and 1.8 < last.point.x < 2.3 and abs(last.point.y) < 0.1, (last.point.x, last.point.y)


def test_synthetic_depth_at_scoop_contact(monkeypatch):
    """Frenet-sim synthetic optical reading is 0.29 m at the model's scoop seat."""
    original = m.Node.declare_parameter
    def set_synthetic_depth(self, key, value):
        original(self, key, True if key == 'publish_synthetic_depth' else value)
    monkeypatch.setattr(m.Node, 'declare_parameter', set_synthetic_depth)
    node = bsn.BallSimNode()
    node.walls = open_walls()
    node.odom_callback(odom(0.0, 0.17, 0.0, 0.0, 0.0))
    for i in range(12):
        Clock.t = i * 0.05
        node.camera_callback()
    out = node.pubs['/ball/camera_depth'].out
    assert out
    assert abs(out[-1].data - 0.29) < 0.035
