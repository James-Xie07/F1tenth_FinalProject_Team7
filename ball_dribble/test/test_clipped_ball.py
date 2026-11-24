"""Synthetic RGB-D checks for the opt-in side-edge detector."""
import math
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip('cv2')
import mock_ros_test  # installs the ROS message stubs and package path

with patch.dict('sys.modules', {
    'cv_bridge': SimpleNamespace(CvBridge=object),
    'message_filters': SimpleNamespace(ApproximateTimeSynchronizer=object, Subscriber=object),
    'sensor_msgs.msg': SimpleNamespace(CameraInfo=object, Image=object),
}):
    from ball_dribble.ball_detector_node import BallDetector, clipped_ball, find_ball, make_mask


INTRINSICS = (190.0, 190.0, 120.0, 80.0)
BALL_RADIUS = 0.0335
BALL_COLOR = (0, 240, 220)


def detect(rgb, depth):
    mask = make_mask(rgb, 20, 48, 70, 70)
    edges = []
    full = find_ball(mask, edge_contours=edges)
    clipped = clipped_ball(edges, mask, depth, INTRINSICS, BALL_RADIUS, 7.0, 0.18, 4.0)
    return full, clipped


def scene(cx):
    rgb = np.zeros((160, 240, 3), np.uint8)
    cv2.circle(rgb, (cx, 80), 20, BALL_COLOR, -1)
    fx, _, principal_x, _ = INTRINSICS
    depth = BALL_RADIUS * fx * math.hypot(1.0, (cx - principal_x) / fx) / 20.0 - BALL_RADIUS
    return rgb, np.full(rgb.shape[:2], depth, np.float32)


@pytest.mark.parametrize('cx', [0, 239])
def test_recovers_a_ball_with_its_center_at_the_edge(cx):
    full, result = detect(*scene(cx))
    assert full is None
    assert result is not None
    circle, z = result
    np.testing.assert_allclose(circle, [cx, 80, 20], atol=1.5)
    assert 0.3 < z < 0.4


def test_keeps_normal_detection_for_a_complete_ball():
    full, clipped = detect(*scene(120))
    assert full is not None
    assert clipped is None


def test_rejects_a_sliver_a_wall_and_inconsistent_depth():
    rgb, depth = scene(-8)
    assert detect(rgb, depth) == (None, None)
    rgb, depth = scene(0)
    assert detect(rgb, np.zeros_like(depth)) == (None, None)
    assert detect(rgb, depth * 2.0) == (None, None)
    rgb[:] = 0
    cv2.rectangle(rgb, (0, 60), (23, 100), BALL_COLOR, -1)
    assert detect(rgb, depth) == (None, None)


def test_node_only_publishes_a_clipped_detection_when_enabled():
    rgb, depth = scene(239)
    stamp = object()
    messages = []
    node = SimpleNamespace(
        K=INTRINSICS,
        p=dict(h_min=20, h_max=48, s_min=70, v_min=70, min_area=150.0, min_radius=7.0,
               min_fill=0.35, min_aspect=0.60, max_aspect=1.60, ball_radius=BALL_RADIUS,
               min_depth=0.18, max_depth=4.0, depth_scale=1.0, frame='rear_axle', allow_clipped_ball=False),
        pitch=0.0, mount=(0.39, 0.04, 0.10),
        bridge=SimpleNamespace(imgmsg_to_cv2=lambda msg, encoding: msg.data, cv2_to_imgmsg=lambda image, encoding: image),
        dbg=SimpleNamespace(get_subscription_count=lambda: 1, publish=lambda image: None),
        pub=SimpleNamespace(publish=messages.append),
    )
    color_msg = SimpleNamespace(header=SimpleNamespace(stamp=stamp), data=rgb)
    depth_msg = SimpleNamespace(data=depth)
    BallDetector.images_callback(node, color_msg, depth_msg)
    assert messages == []
    node.p['allow_clipped_ball'] = True
    BallDetector.images_callback(node, color_msg, depth_msg)
    msg, = messages
    assert msg.header.stamp is stamp
    assert msg.header.frame_id == 'rear_axle'
    assert msg.point.y < 0.0  # right side of the image is the car's right
