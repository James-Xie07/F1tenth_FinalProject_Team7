"""Offline regression tests for switching only the pushing controller."""
import math
import sys
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'test')]
from ball_dribble.ball_control import BallPusher, Config
from ball_dribble.push_mpc import MPCConfig, TrackingMPC, push_reference


@unittest.skipUnless(find_spec('cvxpy') and find_spec('osqp'), 'requires CVXPY and OSQP')
class PushMPCTest(unittest.TestCase):
    def make_pusher(self, controller, **settings):
        return BallPusher(Config(
            path_file=str(ROOT / 'paths' / 'ball_path_levine2nd.csv'),
            push_controller=controller, max_speed=0.30, **settings,
        ))

    def start_pose(self, pusher, s=4.0):
        ref = push_reference(pusher.path, MPCConfig(), s, 0.0, 0.30)
        return float(ref[0, 0]), float(ref[1, 0]), float(ref[3, 0])

    def put_ball(self, pusher, xy, t=0.0):
        pusher.tracker._restart(t, np.asarray(xy), 0.015)
        pusher.tracker.has = True
        pusher.tracker.last_seen = t

    def test_solver_tracks_and_steers_toward_ball(self):
        cfg = MPCConfig()
        reference = np.zeros((4, cfg.horizon + 1))
        reference[0] = np.arange(cfg.horizon + 1) * cfg.dt * 0.30
        reference[2] = 0.30
        mpc = TrackingMPC(cfg)
        self.assertTrue(mpc.problem.is_dpp())
        _, left = mpc.solve([0.0, 0.0, 0.30, 0.0], reference, 0.0, 0.06)
        mpc.reset()
        _, right = mpc.solve([0.0, 0.0, 0.30, 0.0], reference, 0.0, -0.06)
        self.assertGreater(left, 0.0)
        self.assertLess(right, 0.0)
        self.assertLessEqual(max(left, -right), cfg.max_steer_rate * cfg.dt + 1e-3)

    def test_push_uses_configured_scoop_and_mpc(self):
        pusher = self.make_pusher('mpc', scoop_kind='vnarrow', scoop_apex_x=0.510)
        cfg = pusher.push_mpc.cfg
        self.assertEqual(cfg.dt * cfg.horizon, 1.0)
        self.assertEqual(cfg.max_speed, 0.30)
        self.assertAlmostEqual(pusher.path.x_b, 0.584, places=3)
        s = pusher.path.length - 0.03
        ball = pusher.path.at(s)[0]
        ref = push_reference(pusher.path, cfg, s, 0.0, 0.30)
        pose = self.start_pose(pusher, s)
        self.assertLess(np.max(np.abs(np.diff(ref[3]))), math.pi / 2)
        foot = ref[:2, 0] + pusher.path.x_b * np.array([math.cos(pose[2]), math.sin(pose[2])])
        np.testing.assert_allclose(foot, ball, atol=1e-9)
        self.put_ball(pusher, ball)
        with patch.object(pusher, '_stanley', side_effect=AssertionError('PUSH used Stanley')):
            speed, steer = pusher.update(0.0, pose, 0.0)
        self.assertEqual(pusher.mode, 'PUSH')
        self.assertLessEqual(abs(steer), cfg.max_steer_rate * cfg.dt + 1e-6)
        self.assertGreater(speed, 0.0)
        self.assertLessEqual(speed, cfg.max_accel * cfg.dt + 1e-6)

    def test_forward_recovery_matches_stanley(self):
        for mode, ball_s in (('SEARCH', None), ('APPROACH', 4.9), ('CHASE', 11.0)):
            with self.subTest(mode=mode):
                legacy, mpc = [self.make_pusher(controller) for controller in ('stanley', 'mpc')]
                mpc.mode = 'PUSH'
                mpc.push_mpc.last_plan = np.zeros((2, mpc.push_mpc.cfg.horizon))
                pose = self.start_pose(legacy)
                if ball_s is not None:
                    for pusher in (legacy, mpc):
                        self.put_ball(pusher, pusher.path.at(ball_s)[0])
                with patch.object(mpc.push_mpc, 'solve', side_effect=AssertionError('recovery called MPC')):
                    np.testing.assert_allclose(mpc.update(0.0, pose, 0.1), legacy.update(0.0, pose, 0.1))
                self.assertEqual(legacy.mode, mode)
                self.assertEqual(mpc.mode, mode)
                self.assertIsNone(mpc.push_mpc.last_plan)

    def test_node_keeps_existing_topics_and_stops_on_solver_error(self):
        import mock_ros_test as ros

        overrides = {
            'push_controller': 'mpc',
            'waypoint_file': str(ROOT / 'paths' / 'ball_path_levine2nd.csv'),
            'odom_is_laser_pose': False,
        }

        def declare(node, name, value):
            node._params[name] = overrides.get(name, value)

        with patch.object(ros.Node, 'declare_parameter', declare):
            node = ros.bpn.BallPushNode()
        self.assertEqual(set(node.subs), {'/pf/pose/odom', '/odom', '/ball/position', '/map'})
        self.assertEqual(set(node.pubs), {'/drive', '~/mode', '~/ball'})
        t = 1.0
        ros.Clock.t = t
        pusher = node.pusher
        pose = self.start_pose(pusher)
        node.odom_callback(ros.odom(t, *pose, 0.0))
        self.put_ball(pusher, pusher.path.at(4.0)[0], t)
        with patch.object(pusher.push_mpc, 'solve', side_effect=RuntimeError('mock solver failure')) as solve:
            node.control_callback()
        solve.assert_called_once()
        drive = node.pubs['/drive'].out[-1].drive
        self.assertEqual((drive.speed, drive.steering_angle), (0.0, 0.0))
        self.assertEqual(pusher.act.cmds[-1], (t, 0.0, 0.0))
