"""Regression tests for the observed reacquire-PUSH-STOP-HOLD failure.

No ROS required. Run with python -m pytest test/test_recontact_and_launch.py.
"""
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ball_dribble.ball_recovery import (RecoverySettings, RecoverySupervisor,
                                         direct_forward_curve)
from ball_dribble.frenet_mpc import MPCConfig
from ball_dribble.physics import BallParams, make_scoop, rider_x


class OpenWalls:
    def dist(self, x, y):
        return np.ones_like(np.asarray(x, dtype=float)) * 5.0


def test_sim_scoop_distance_is_consistent_with_synthetic_camera():
    x = rider_x(make_scoop('channel'), BallParams().radius)
    assert abs(x - 0.5081) < 1e-8
    # The old config used 0.540, causing a dead zone just past contact.
    mpc = MPCConfig(max_speed=0.5, ball_center_ahead=x)
    rec = RecoverySettings(allow_direct_approach=True)
    assert direct_forward_curve((0, 0, 0), (0.548, 0), OpenWalls(), mpc, rec) is not None


def test_contact_reacquisition_requires_stable_measurements():
    c = RecoverySettings(allow_direct_approach=True, contact_confirm_time=0.15,
                         contact_loss_grace=0.30, retry_depth_hold=True)
    sup = RecoverySupervisor(c)
    plan = np.array([[0, 0], [0.2, 0]])
    pose = (0, 0, 0)
    sup._enter('DIRECT_APPROACH', 0, 'test')
    assert sup.step(1.00, pose, .15, 0, .29, True, direct_plan=plan) == 'DIRECT_APPROACH'
    assert sup.step(1.10, pose, .15, 0, .36, True, direct_plan=plan) == 'DIRECT_APPROACH'
    assert sup.step(1.20, pose, .15, 0, .29, True, direct_plan=plan) == 'DIRECT_APPROACH'
    assert sup.step(1.40, pose, .15, 0, .29, True, direct_plan=plan) == 'PUSH'
    # Brief bounce at the scoop lip must not produce PUSH->STOP 0.1 s later.
    assert sup.step(1.50, pose, .15, 0, .34, True, direct_plan=plan) == 'PUSH'
    assert sup.step(1.62, pose, .15, 0, .29, True, direct_plan=plan) == 'PUSH'
    # Sustained missing contact still triggers a stop, even with grace enabled.
    assert sup.step(1.80, pose, .15, 0, .35, True, direct_plan=plan) == 'PUSH'
    assert sup.step(2.12, pose, .15, 0, .35, True, direct_plan=plan) == 'STOP'


def test_lateral_release_stops_immediately_even_with_grace():
    c = RecoverySettings(allow_direct_approach=True, contact_confirm_time=0.15,
                         contact_loss_grace=0.30, retry_depth_hold=True)
    sup = RecoverySupervisor(c)
    sup._enter('PUSH', 0, 'test')
    sup.step(0.00, (0, 0, 0), .2, 0, .29, True)
    assert sup.step(0.10, (0, 0, 0), .2, .101, .29, True) == 'STOP'
    assert sup.loss_kind == 'lateral'


def test_hold_can_retry_safe_depth_only_forward_approach():
    c = RecoverySettings(allow_direct_approach=True, retry_depth_hold=True)
    sup = RecoverySupervisor(c)
    sup._enter('PUSH', 0, 'test')
    assert sup.step(.1, (0, 0, 0), 0, 0, .39, True) == 'STOP'
    assert sup.step(.6, (0, 0, 0), 0, 0, .39, True, direct_plan=None) == 'HOLD'
    plan = np.array([[0, 0], [0.12, 0]])
    assert sup.step(.8, (0, 0, 0), 0, 0, .39, True, direct_plan=plan) == 'DIRECT_APPROACH'
    # Missing ball must never cause autonomous forward motion from HOLD.
    sup._enter('HOLD', 1, 'test')
    assert sup.step(1.1, (0, 0, 0), 0, None, None, False, direct_plan=plan) == 'HOLD'
    sup.loss_kind = 'lateral'
    assert sup.step(1.2, (0, 0, 0), 0, .11, .39, True, direct_plan=plan) == 'HOLD'


def test_recovery_defaults_preserve_real_car_safety():
    c = RecoverySettings()
    assert not c.retry_depth_hold
    assert not c.allow_direct_approach
    assert c.contact_loss_grace == 0
    assert c.contact_confirm_time == 0


def test_separate_sim_launches_do_not_start_duplicate_controller_or_gym():
    world = (ROOT / 'launch/sim_levine_world.launch.py').read_text()
    ctrl = (ROOT / 'launch/sim_levine_controller.launch.py').read_text()
    assert "'map_path': 'levine_blocked'" in world
    assert "executable='ball_sim_node'" in world
    assert 'ball_frenet_mpc_node' not in world
    assert "executable='ball_frenet_mpc_node'" in ctrl
    assert 'gym_bridge_launch.py' not in ctrl
    assert 'ball_sim_node' not in ctrl
    sim_cfg = (ROOT / 'config/sim_frenet.yaml').read_text()
    assert 'levine_ccw_raceline.csv' in sim_cfg
    assert 'reference_is_ball_path: false' in sim_cfg
    real_cfg = (ROOT / 'config/frenet.yaml').read_text()
    assert 'levine_ball_path.csv' in real_cfg
    assert 'reference_is_ball_path: true' in real_cfg


def test_real_launch_filters_camera_topics_and_starts_web_video():
    launch = (ROOT / 'launch/real_levine.launch.py').read_text()
    assert "executable='ball_video_web_node'" in launch
    assert "executable='foxglove_bridge'" in launch
    assert "'topic_whitelist': foxglove_topics" in launch
    assert '/camera/' not in launch
    assert '/ball_detector_node/debug_image' not in launch
    assert "r'/ball/position'" in launch
    assert "r'/ball_frenet_mpc_node/.*'" in launch


def test_web_video_is_packaged_and_does_not_change_contact_logic():
    setup = (ROOT / 'setup.py').read_text()
    video = (ROOT / 'ball_dribble/ball_video_web_node.py').read_text()
    recovery = (ROOT / 'ball_dribble/ball_recovery.py').read_text()
    assert 'ball_video_web_node = ball_dribble.ball_video_web_node:main' in setup
    assert "'/stream.mjpg'" in video
    assert "cv2.imencode('.jpg'" in video
    assert 'CALIBRATING_CONTACT' not in recovery
