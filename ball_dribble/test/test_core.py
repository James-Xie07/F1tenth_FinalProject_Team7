"""Offline tests (no ROS needed): `python3 -m pytest test/`"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from ball_dribble import sim
from ball_dribble.ball_control import Config


def test_nominal_lap_keeps_ball():
    cfg = Config(path_file=str(pathlib.Path(__file__).resolve().parents[1] / 'paths' / 'ball_path.csv'))
    r = sim.run(cfg, seed=1)
    assert r['lap_time'] == r['lap_time'] and r['lap_time'] < 35      # finished (not NaN) in a sensible time
    assert r['car_hits'] == 0 and r['contact'] > 0.8
