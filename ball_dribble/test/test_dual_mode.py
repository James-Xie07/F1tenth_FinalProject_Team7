"""Pure-Python tests of real and simulator Levine path modes."""
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ball_dribble.frenet_mpc import MPCConfig, LevinePath, FrenetPlanner

ROOT = Path(__file__).resolve().parents[1]


def test_real_path_is_unchanged_ball_path_semantics():
    cfg = MPCConfig(max_speed=0.5)
    path = LevinePath(ROOT/'paths'/'levine_ball_path.csv', cfg, reference_is_ball_path=True)
    data = np.genfromtxt(ROOT/'paths'/'levine_ball_path.csv', names=True, delimiter=',')
    delta = np.linalg.norm(path.xy[0] - np.array([data['x'][0], data['y'][0]]))
    assert 0.50 < delta < 0.58
    assert len(path.xy) == 1237


def test_sim_path_is_vehicle_reference_without_offset():
    cfg = MPCConfig(max_speed=0.5)
    path = LevinePath(ROOT/'paths'/'levine_ccw_raceline.csv', cfg, reference_is_ball_path=False)
    data = np.genfromtxt(ROOT/'paths'/'levine_ccw_raceline.csv', names=True, delimiter=',')
    assert len(path.xy) == 1225
    assert np.allclose(path.xy, np.column_stack([data['x'], data['y']]))
    assert 60 < path.length < 65
    # Spot check zero offset and correct heading at multiple points.
    planner = FrenetPlanner(path, cfg)
    for i in (0, 60, 180, 440, 760, 980):
        s, d, yaw = path.project(path.xy[i])
        assert abs(d) < 1e-4
        assert np.isfinite([s, yaw]).all()


def test_sim_and_real_paths_are_separate_coordinate_frames():
    cfg = MPCConfig()
    real = LevinePath(ROOT/'paths'/'levine_ball_path.csv', cfg, True)
    sim = LevinePath(ROOT/'paths'/'levine_ccw_raceline.csv', cfg, False)
    # Avoid accidental future wiring of the real ball path into the simulator.
    assert abs(float(real.xy[:,1].mean() - sim.xy[:,1].mean())) > 15.0
