"""Offline tests for the Frenet planner and contact-based recovery logic.

CVXPY/OSQP solver must be installed on the Jetson for the live node. These
geometry and state-machine tests intentionally do not need ROS or a solver.
"""
import math
import pathlib
import sys
import numpy as np

sys.path.insert(0,str(pathlib.Path(__file__).resolve().parents[1]))
from ball_dribble.frenet_mpc import MPCConfig,LevinePath,FrenetPlanner,quintic_offset
from ball_dribble.ball_recovery import (RecoverySupervisor,RecoverySettings,approach_curve,
                                        rear_scan_clear,rear_map_clear,direct_forward_curve)

ROOT=pathlib.Path(__file__).resolve().parents[1]

class OpenWalls:
    def dist(self,x,y):return np.ones_like(np.asarray(x,dtype=float))*5.0


def test_levine_waypoints_and_speed():
    cfg=MPCConfig(max_speed=0.5)
    p=LevinePath(ROOT/'paths'/'levine_ball_path.csv',cfg)
    assert len(p.xy)==1237
    assert 60<p.length<65
    planner=FrenetPlanner(p,cfg)
    for idx in [0,150,400,650,900,1100]:
        state=np.array([*p.xy[idx],0.20,p.yaw[idx]])
        result=planner.plan(state)
        assert result is not None, ('No Frenet plan at waypoint',idx)
        ref,poly,target=result
        assert ref.shape==(4,11)
        assert ref[2].max()<=.5+1e-9
        assert np.isfinite(ref).all()


def test_contact_release_stops_before_reverse():
    sup=RecoverySupervisor()
    assert sup.step(0,(0,0,0),0,0.0,.29,True)=='PUSH'
    assert sup.step(.10,(0,0,0),.15,.11,.29,True)=='STOP'
    assert sup.step(.30,(0,0,0),.04,.11,.29,True)=='STOP'
    assert sup.step(.65,(0,0,0),0,.11,.29,True,reverse_safe=True)=='REVERSE'
    assert sup.step(.75,(-.10,0,0),-.1,.11,.29,True,True,object())=='REVERSE'
    assert sup.step(.90,(-.24,0,0),-.1,.11,.29,True,True,object())=='REVERSE_STOP'
    assert sup.step(1.10,(-.25,0,0),-.02,.11,.29,True,True,object())=='REVERSE_STOP'
    assert sup.step(1.35,(-.25,0,0),0,.11,.29,True,True,object())=='APPROACH'
    assert sup.step(1.45,(-.25,0,0),0,0,.29,True,True,object())=='PUSH'


def test_missing_rear_visibility_prevents_reverse():
    sup=RecoverySupervisor()
    sup.step(0,(0,0,0),0,0,.29,True)
    sup.step(.10,(0,0,0),0,.15,.34,True)
    assert sup.step(.65,(0,0,0),0,.15,.34,True,False)=='HOLD'
    assert sup.step(1.0,(0,0,0),0,.15,.34,True,True)=='HOLD'


def test_rear_lidar_coverage_required():
    cfg=RecoverySettings()
    front=np.linspace(-math.radians(135),math.radians(135),1081)
    assert not rear_scan_clear((front,np.ones_like(front)*2),cfg)
    full=np.linspace(-math.pi,math.pi,1440)
    assert rear_scan_clear((full,np.ones_like(full)*2),cfg)
    blocked=np.ones_like(full)*2
    blocked[(np.abs(full)>math.radians(165))]=0.6
    assert not rear_scan_clear((full,blocked),cfg)


def test_replan_behind_ball_after_enough_reverse():
    cfg=MPCConfig()
    path=LevinePath(ROOT/'paths'/'levine_ball_path.csv',cfg)
    t=path.yaw[0]; tangent=np.array([math.cos(t),math.sin(t)])
    normal=np.array([-math.sin(t),math.cos(t)])
    # Ball slid 110 mm left of the scoop, causing recovery.
    ball=path.xy[0]+.54*tangent+.11*normal
    early=path.xy[0]-.20*tangent
    later=path.xy[0]-.90*tangent
    assert approach_curve((*early,t),ball,path,OpenWalls(),cfg) is None
    assert approach_curve((*later,t),ball,path,OpenWalls(),cfg) is not None


def test_reverse_requires_map():
    cfg=MPCConfig(); rcfg=RecoverySettings()
    assert not rear_map_clear((0,0,0),None,cfg,rcfg)
    assert rear_map_clear((0,0,0),OpenWalls(),cfg,rcfg)


def test_depth_only_release_can_approach_directly_in_sim():
    cfg=RecoverySettings(allow_direct_approach=True)
    sup=RecoverySupervisor(cfg)
    short=np.array([[0.0,0.0],[0.3,0.0]])
    assert sup.step(0.0,(0,0,0),0.0,0.0,.29,True,direct_plan=short)=='PUSH'
    assert sup.step(.1,(0,0,0),.15,0.0,.38,True,direct_plan=short)=='STOP'
    assert sup.step(.3,(0,0,0),.04,0.0,.38,True,direct_plan=short)=='STOP'
    assert sup.step(.6,(0,0,0),0.0,0.0,.38,True,direct_plan=short)=='DIRECT_APPROACH'
    # One missed path causes a brief safe stop, not a permanent HOLD.
    assert sup.step(.7,(0,0,0),0.0,0.0,.38,True,direct_plan=None)=='DIRECT_APPROACH'
    assert sup.step(.9,(0,0,0),0.0,0.0,.38,True,direct_plan=short)=='DIRECT_APPROACH'
    assert sup.step(1.1,(0,0,0),.05,0.0,.29,True,direct_plan=short)=='PUSH'

def test_depth_only_release_default_stays_conservative():
    sup=RecoverySupervisor()
    sup.step(0.0,(0,0,0),0,0,.29,True)
    assert sup.step(.1,(0,0,0),0,0,.38,True)=='STOP'
    assert sup.step(.7,(0,0,0),0,0,.38,True,reverse_safe=True)=='REVERSE'

def test_direct_curve_checks_center_and_forward_distance():
    mpc=MPCConfig(max_speed=.5)
    recovery=RecoverySettings(allow_direct_approach=True)
    straight=direct_forward_curve((0,0,0),(.8,0.0),OpenWalls(),mpc,recovery)
    assert straight is not None
    assert np.allclose(straight[0],[0,0])
    assert abs(straight[-1,0]-(.8-mpc.ball_center_ahead)) < 1e-5
    assert direct_forward_curve((0,0,0),(.8,.12),OpenWalls(),mpc,recovery) is None
    assert direct_forward_curve((0,0,0),(.51,0),OpenWalls(),mpc,recovery) is None

def test_approach_grace_before_hold():
    sup=RecoverySupervisor(RecoverySettings(approach_plan_grace=.5))
    sup._enter('APPROACH',0,'test')
    assert sup.step(.1,(0,0,0),.1,.10,.40,True,approach_plan=None)=='APPROACH'
    assert sup.step(.3,(0,0,0),.1,.10,.40,True,approach_plan=None)=='APPROACH'
    assert sup.step(.61,(0,0,0),.1,.10,.40,True,approach_plan=None)=='HOLD'
