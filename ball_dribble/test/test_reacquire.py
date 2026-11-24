"""Directional search with the offline car and depth-camera models."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ball_dribble.ball_control import BallPusher, Config
from ball_dribble.mapwalls import MapWalls
from ball_dribble.physics import Camera, CameraParams, Car


def make_pusher(walls=None):
    return BallPusher(Config(
        path_file=str(ROOT / 'paths' / 'ball_path.csv'),
        reacquire_enabled=True, search_speed=0.0,
    ), walls=walls)


class OpenWalls:
    def dist(self, x, y):
        return np.full_like(x, 2.0)


@pytest.mark.parametrize('side', [-1, 1])
def test_limited_view_gets_the_ball_back(side):
    pusher = make_pusher(OpenWalls())
    camera = Camera(CameraParams(fov=0.50, p_detect=1.0, bearing_sigma=0.0, range_sigma0=0.0, range_sigma2=0.0))
    car = Car((0.0, 0.0, 0.0))
    ball = np.array([1.50, side * 0.85])
    first_pose = (0.0, 0.0, side * 0.15)
    pusher.update(0.0, first_pose, 0.0)
    first = camera.sense(first_pose, ball, pusher.cfg.ball_radius, None)
    assert first is not None
    pusher.observe(0.0, first)
    assert camera.sense(car.pose, ball, pusher.cfg.ball_radius, None) is None

    for t in np.arange(1.25, 4.0, 0.05):
        speed, steer = pusher.update(t, car.pose, car.v, car.yaw_rate)
        assert pusher.mode == 'REACQUIRE'
        assert 0.0 < speed <= pusher.cfg.reacquire_speed
        assert steer * side > 0.0
        car.command(t, speed, steer)
        car.step(t, 0.05)
        z = camera.sense(car.pose, ball, pusher.cfg.ball_radius, None)
        if z is not None:
            pusher.observe(t, z)
            pusher.update(t + 0.05, car.pose, car.v, car.yaw_rate)
            assert pusher.mode in ('APPROACH', 'CHASE')
            assert np.hypot(car.x, car.y) <= pusher.cfg.reacquire_distance
            return
    pytest.fail('the search arc did not bring the ball back into view')


def test_stops_without_observation_or_clearance():
    class WallOnLeft:
        def dist(self, x, y):
            return np.where(y < 0.23, 1.0, 0.0)

    for walls in (None, WallOnLeft()):
        pusher = make_pusher(walls)
        assert pusher.update(0.0, (0.0, 0.0, 0.0), 0.0) == (0.0, 0.0)
        pusher.observe(0.0, (1.50, 0.85))
        assert pusher.update(1.25, (0.0, 0.0, 0.0), 0.0) == (0.0, 0.0)
        assert pusher.mode == 'SEARCH'
        pusher.walls = OpenWalls()
        assert pusher.update(1.30, (0.0, 0.0, 0.0), 0.0) == (0.0, 0.0)


def test_search_brakes_and_does_not_restart_without_new_image():
    pusher = make_pusher(OpenWalls())
    car = Car((0.0, 0.0, 0.0))
    pusher.update(0.0, car.pose, 0.0)
    pusher.observe(0.0, (1.50, 0.85))
    modes = []
    for t in np.arange(1.25, 5.0, 0.05):
        speed, steer = pusher.update(t, car.pose, car.v, car.yaw_rate)
        modes.append(pusher.mode)
        car.command(t, speed, steer)
        car.step(t, 0.05)
    assert modes[0] == 'REACQUIRE'
    stopped = modes.index('SEARCH')
    assert set(modes[stopped:]) == {'SEARCH'}
    assert 0.0 < np.hypot(car.x, car.y) <= pusher.cfg.reacquire_distance
    assert pusher.v_cmd == 0.0


def test_outside_occupancy_grid_has_no_clearance():
    origin = SimpleNamespace(
        position=SimpleNamespace(x=0.0, y=0.0),
        orientation=SimpleNamespace(w=1.0, x=0.0, y=0.0, z=0.0),
    )
    grid = SimpleNamespace(info=SimpleNamespace(width=20, height=20, resolution=0.05, origin=origin),
                           data=[0] * 400)
    walls = MapWalls(grid)
    np.testing.assert_allclose(walls.dist([-0.05, 0.05, 0.50, 1.01], [0.5] * 4), [0.0, 0.05, 0.50, 0.0])
