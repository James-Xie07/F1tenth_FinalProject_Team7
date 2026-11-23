"""Optional solver smoke test, executed on the Jetson when CVXPY/OSQP are installed."""
import pathlib
import sys
import numpy as np
import pytest

pytest.importorskip('cvxpy')
pytest.importorskip('osqp')
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ball_dribble.frenet_mpc import MPCConfig,LevinePath,FrenetPlanner,TrackingMPC


def test_mpc_qp_on_levine():
    root=pathlib.Path(__file__).resolve().parents[1]
    cfg=MPCConfig()
    path=LevinePath(root/'paths/levine_ball_path.csv',cfg)
    state=np.array([*path.xy[400],0.0,path.yaw[400]])
    result=FrenetPlanner(path,cfg).plan(state)
    assert result is not None
    ref=result[0]
    vel,steer,pred=TrackingMPC(cfg).solve(state,ref,0.0,use_ball=True,ball_lateral=0.02)
    assert 0 <= vel <= .5
    assert abs(steer) <= cfg.max_steer+1e-6
    assert pred.shape == (4,cfg.horizon+1)
    assert np.isfinite(pred).all()
