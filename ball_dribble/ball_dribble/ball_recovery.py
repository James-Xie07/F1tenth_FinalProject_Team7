"""ROS-free safe stop / reverse / replan supervisor for a front-mounted ball scoop.

Reverse is deliberately conservative. It requires a loaded occupancy map, a fresh
rear-facing laser sector, and a fresh detected ball. With the typical forward-facing
270-degree F1TENTH LiDAR, the rear blind spot prevents automatic reversal.
Provide rear sensing before enabling autonomous recovery on the physical car.
"""
from dataclasses import dataclass
import math
import numpy as np
from .frenet_mpc import body_clearance, obstacle_clearance, wrap


@dataclass
class RecoverySettings:
    contact_depth: float = 0.29          # optical-axis surface depth from D435i, metres
    depth_tolerance: float = 0.035
    lateral_release: float = 0.100      # absolute ball y in rear_axle frame (100 mm)
    lateral_recontact: float = 0.075
    sensor_timeout: float = 0.40
    stop_hold: float = 0.40
    stationary_speed: float = 0.035
    reverse_speed: float = 0.12
    min_reverse: float = 0.20
    max_reverse: float = 0.90
    reverse_step_check: float = 0.22
    rear_lidar_margin: float = 0.75
    rear_sector_deg: float = 20.0
    rear_scan_timeout: float = 0.35
    min_rear_beams: int = 3
    safe_clearance: float = 0.07
    approach_speed: float = 0.20
    touch_speed: float = 0.12
    # Enable only for simulator testing until forward obstacle sensing and
    # physical approach dynamics have been verified on the real vehicle.
    allow_direct_approach: bool = False
    approach_plan_grace: float = 0.70  # seconds to stop and retry a lost path
    direct_ball_max_x: float = 1.25    # forward camera/recovery operating range
    # Simulation can use temporal confirmation to avoid false recontact followed
    # immediately by a single noisy depth reading. Real defaults stay strict.
    contact_confirm_time: float = 0.0
    contact_loss_grace: float = 0.0
    retry_depth_hold: bool = False      # only permit safe retries of forward recovery


def car_point_to_world(pose, local_xy):
    x, y, yaw = pose
    px, py = local_xy
    return np.array([x + px*math.cos(yaw) - py*math.sin(yaw),
                     y + px*math.sin(yaw) + py*math.cos(yaw)])


def hermite_curve(p0, yaw0, p1, yaw1, n=51):
    p0 = np.asarray(p0,float); p1 = np.asarray(p1,float)
    length = float(np.linalg.norm(p1-p0))
    h0 = np.array([math.cos(yaw0), math.sin(yaw0)]) * length * 0.8
    h1 = np.array([math.cos(yaw1), math.sin(yaw1)]) * length * 0.8
    t = np.linspace(0,1,n)
    return ((2*t**3 - 3*t**2 + 1)[:,None]*p0 + (t**3 - 2*t**2 + t)[:,None]*h0
            + (-2*t**3 + 3*t**2)[:,None]*p1 + (t**3 - t**2)[:,None]*h1)


def approach_curve(pose, ball_world, path, walls, mpc_cfg, clearance=0.07, obstacle_points=None):
    """Find a collision-checked forward approach from the rear axle to behind the ball.

    Small changes of final heading are allowed. This matters in the narrow Levine
    track: an exact raceline tangent is often unattainable in a short recovery.
    """
    if walls is None or ball_world is None:
        return None
    pos = np.asarray(pose[:2],float)
    ball = np.asarray(ball_world,float)
    s, _, _ = path.project(ball)
    _, heading = path.at(s)
    yaw_path = float(heading[0])
    max_kappa = math.tan(mpc_cfg.max_steer)/mpc_cfg.wheelbase
    best = None
    # Aim for path tangent, but accept the current vehicle direction if necessary.
    candidate_angles = [yaw_path, pose[2], yaw_path-0.2, yaw_path+0.2,
                        pose[2]-0.15, pose[2]+0.15, yaw_path-0.4, yaw_path+0.4]
    for yaw_goal in candidate_angles:
        u = np.array([math.cos(yaw_goal), math.sin(yaw_goal)])
        staging = ball - (mpc_cfg.ball_center_ahead + 0.11)*u
        near_stage = np.linalg.norm(staging-pos) < 0.22 and abs(wrap(pose[2]-yaw_goal)) < 0.4
        target = (ball - mpc_cfg.ball_center_ahead*u) if near_stage else staging
        forward = float(np.dot(target-pos,np.array([math.cos(pose[2]),math.sin(pose[2])])))
        if forward < 0.035 or np.linalg.norm(target-pos) < 0.035:
            continue
        curve = hermite_curve(pos, pose[2], target, yaw_goal)
        delta = np.diff(curve,axis=0)
        step = np.linalg.norm(delta,axis=1)
        if np.min(step) < 1e-5:
            continue
        psi = np.unwrap(np.arctan2(delta[:,1],delta[:,0]))
        curvature = np.diff(psi)/np.maximum(0.5*(step[1:]+step[:-1]),1e-6)
        if len(curvature) and np.any(np.abs(curvature) > max_kappa*1.05):
            continue
        yaw_curve = np.r_[psi,psi[-1]]
        if body_clearance(walls,curve,yaw_curve,mpc_cfg) < clearance:
            continue
        if obstacle_clearance(obstacle_points,curve,yaw_curve,mpc_cfg) < mpc_cfg.obstacle_margin:
            continue
        cost = 2.0*abs(wrap(yaw_goal-yaw_path)) + 0.3*float(np.sum(step))
        if best is None or cost < best[0]:
            best = (cost,curve)
    return None if best is None else best[1]



def direct_forward_curve(pose, ball_xy, walls, mpc_cfg, settings, obstacle_points=None):
    """Collision-check a short straight approach to a centred ball ahead.

    This case is for depth-only loss (the vehicle stopped but the ball rolled
    forward). It does not apply when the ball escapes sideways: that still
    follows STOP -> REVERSE -> APPROACH. All geometry is in rear-axle frame.
    """
    if walls is None or ball_xy is None:
        return None
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    if not np.isfinite([bx, by]).all():
        return None
    if abs(by) > settings.lateral_recontact:
        return None
    if not (mpc_cfg.ball_center_ahead + 0.02 < bx <= settings.direct_ball_max_x):
        return None
    # Move the rear axle toward the ball, stopping as soon as the intended
    # scoop-to-ball distance is reached. A steering correction is deliberately
    # excluded for this narrow centred-ball case.
    distance = min(bx - mpc_cfg.ball_center_ahead, 0.55)
    if distance <= 0.02:
        return None
    p = np.asarray(pose[:2], float)
    direction = np.array([math.cos(pose[2]), math.sin(pose[2])])
    curve = p[None, :] + np.linspace(0.0, distance, 21)[:, None] * direction[None, :]
    heading = np.full(len(curve), float(pose[2]))
    if body_clearance(walls, curve, heading, mpc_cfg) < settings.safe_clearance:
        return None
    if obstacle_clearance(obstacle_points, curve, heading, mpc_cfg) < mpc_cfg.obstacle_margin:
        return None
    return curve

def rear_map_clear(pose, walls, cfg, settings):
    if walls is None:
        return False
    x,y,yaw = pose
    steps = np.arange(0,settings.reverse_step_check+1e-6,0.04)
    xy = np.column_stack((x - steps*math.cos(yaw), y - steps*math.sin(yaw)))
    heading = np.full(len(xy),yaw)
    return body_clearance(walls, xy, heading, cfg) >= settings.safe_clearance


def rear_scan_clear(scan, settings, lidar_yaw=0.0):
    """Require actual LiDAR coverage of the rear. Missing/outdated beams are unsafe."""
    if scan is None:
        return False
    angles, ranges = scan
    angles = np.asarray(angles,float) + lidar_yaw
    ranges = np.asarray(ranges,float)
    near_rear = np.abs(np.abs(np.arctan2(np.sin(angles),np.cos(angles)))-math.pi) < math.radians(settings.rear_sector_deg)
    coverage = ranges[near_rear]
    if coverage.size < settings.min_rear_beams:
        return False
    # Inf means valid no-return only when the sensor has a published maximum range.
    if not np.all(np.isfinite(coverage)):
        return False
    return bool(np.all(coverage > settings.rear_lidar_margin))


class RecoverySupervisor:
    def __init__(self, settings=None):
        self.cfg = settings or RecoverySettings()
        self.mode = 'WAIT_BALL'
        self.changed_at = 0.0
        self.reverse_start = None
        self.reason = 'waiting for contact'
        self.loss_kind = None
        self.plan_missing_since = None
        self.contact_since = None
        self.last_good_contact_at = None

    def _enter(self, mode, now, reason):
        if self.mode != mode:
            self.mode = mode
            self.changed_at = now
            self.plan_missing_since = None
        self.reason = reason

    def step(self, now, pose, speed, ball_y, optical_depth, detection_fresh,
             reverse_safe=False, approach_plan=None, direct_plan=None):
        """Return mode; caller must translate STOP/HOLD/WAIT into 0 speed."""
        c = self.cfg
        seen = detection_fresh and ball_y is not None and optical_depth is not None
        depth_matches = bool(seen and abs(optical_depth-c.contact_depth) <= c.depth_tolerance)
        contact = bool(depth_matches and abs(ball_y) <= c.lateral_recontact)
        if contact:
            if self.contact_since is None:
                self.contact_since = now
            self.last_good_contact_at = now
        else:
            self.contact_since = None
        confirmed_contact = bool(contact and self.contact_since is not None
                                 and now - self.contact_since + 1e-9 >= c.contact_confirm_time)
        released = bool(seen and abs(ball_y) > c.lateral_release)
        if self.mode == 'WAIT_BALL':
            if released:
                self.loss_kind = 'lateral'
                self._enter('STOP',now,'ball lateral displacement > 100 mm')
            elif confirmed_contact:
                self._enter('PUSH',now,'camera contact confirmed')
            elif c.allow_direct_approach and seen and direct_plan is not None:
                self._enter('DIRECT_APPROACH',now,'centred ball ahead, approach directly')
        elif self.mode == 'PUSH':
            if released:
                self.loss_kind = 'lateral'
                self._enter('STOP',now,'ball lateral displacement > 100 mm')
            elif not seen:
                self._enter('HOLD',now,'ball reading missing/stale')
            elif not depth_matches:
                # A brief noisy frame after re-contact must not trigger a full
                # stop/recovery cycle. Large lateral escape still stops above.
                recent_contact = (self.last_good_contact_at is not None and
                                  now - self.last_good_contact_at < c.contact_loss_grace)
                if not recent_contact:
                    self.loss_kind = 'depth'
                    self._enter('STOP',now,'camera depth no longer confirms contact')
        elif self.mode == 'STOP':
            if now-self.changed_at >= c.stop_hold and abs(speed) <= c.stationary_speed:
                if self.loss_kind == 'depth' and c.allow_direct_approach:
                    if confirmed_contact:
                        self.loss_kind = None
                        self._enter('PUSH',now,'contact confirmed again while stopped')
                    elif seen and direct_plan is not None:
                        self._enter('DIRECT_APPROACH',now,'ball rolled ahead; no reverse required')
                    else:
                        self._enter('HOLD',now,'no collision-free direct forward approach')
                elif seen and reverse_safe:
                    self.reverse_start = np.asarray(pose[:2],float).copy()
                    self._enter('REVERSE',now,'clear rear; reversing slowly')
                elif not seen:
                    self._enter('HOLD',now,'ball not visible: cannot reverse')
                else:
                    self._enter('HOLD',now,'rear/map clearance not verified')
        elif self.mode == 'REVERSE':
            if not seen or not reverse_safe:
                self._enter('HOLD',now,'rear or ball data unavailable')
            else:
                distance = np.linalg.norm(np.asarray(pose[:2],float)-self.reverse_start)
                if distance >= c.min_reverse and approach_plan is not None:
                    self._enter('REVERSE_STOP',now,'valid path found; brake before moving forward')
                elif distance >= c.max_reverse:
                    self._enter('HOLD',now,'maximum reverse distance without a feasible plan')
        elif self.mode == 'REVERSE_STOP':
            if not seen:
                self._enter('HOLD',now,'ball lost after reversing')
            elif now-self.changed_at >= c.stop_hold and abs(speed) <= c.stationary_speed:
                if approach_plan is not None:
                    self._enter('APPROACH',now,'stopped and ready for forward approach')
                else:
                    self._enter('HOLD',now,'recovery plan invalid after stopping')
        elif self.mode in ('APPROACH', 'DIRECT_APPROACH'):
            if confirmed_contact:
                self.loss_kind = None
                self._enter('PUSH',now,'camera confirmed reacquisition')
            elif released and self.mode == 'DIRECT_APPROACH':
                self.loss_kind = 'lateral'
                self._enter('STOP',now,'ball moved sideways during direct approach')
            elif not seen:
                self._enter('HOLD',now,'ball lost during approach')
            else:
                current_plan = direct_plan if self.mode == 'DIRECT_APPROACH' else approach_plan
                if current_plan is None:
                    if self.plan_missing_since is None:
                        self.plan_missing_since = now
                    elif now-self.plan_missing_since >= c.approach_plan_grace:
                        self._enter('HOLD',now,'approach trajectory unavailable after retry')
                else:
                    self.plan_missing_since = None
        elif self.mode == 'HOLD':
            # Only auto-retry the *forward*, visible, depth-only recovery when
            # the simulator explicitly enables it. Never auto-resume a lateral
            # escape, stale perception, or blind reverse.
            if (c.retry_depth_hold and c.allow_direct_approach and
                    self.loss_kind == 'depth' and seen and not released and
                    abs(speed) <= c.stationary_speed):
                if confirmed_contact:
                    self.loss_kind = None
                    self._enter('PUSH',now,'stable contact confirmed while waiting')
                elif direct_plan is not None:
                    self._enter('DIRECT_APPROACH',now,'new safe forward path available')
        return self.mode

    def reset(self, now):
        self._enter('WAIT_BALL',now,'manual recovery reset')
        self.reverse_start = None
        self.loss_kind = None
        self.plan_missing_since = None
        self.contact_since = None
        self.last_good_contact_at = None
