"""Ball-pushing controller (ROS-free core).

The car pushes a small ball around the track with a V-shaped scoop on its front.  Three pieces:

  PathRef     the path the BALL should follow (ball_path.csv) plus a speed profile that never brakes harder
              than a pushed ball can follow (a free ball only slows at its rolling decel, so a harder-braking car
              drives away from it).
  BallTracker a Kalman filter on the ball in the WORLD frame.  Its motion model is rolling friction plus contact with
              the scoop, so while the ball sits in the depth camera's blind zone (inside the scoop) the estimate keeps
              riding with the car instead of vanishing.  Handles camera latency by rewinding to the image time.
  BallPusher  the state machine.  PUSH: Stanley steering with the scoop point as control point, speed from the profile.
              APPROACH: line-Stanley onto the line "behind the ball, pointing along the path", speed matched to the
              ball.  CHASE: follow the path to a ball that rolled away (a ball behind the car is reached by going round
              the loop).  SEARCH: follow the path slowly until the ball is seen again.

Frames: car frame = rear axle origin, x forward, y left (same convention as the Frenet MPC node).
"""
import collections
import math
from dataclasses import dataclass, fields

import numpy as np
from scipy.spatial import cKDTree

from .physics import (BODY_FRONT, BallParams, ball_step, car_to_world, car_velocity_field, make_scoop, rider_x, world_to_car, wrap)


@dataclass
class Config:
    # ---- path and speed profile ----
    path_file: str = 'ball_path.csv'
    max_speed: float = 2.5             # [m/s]
    v_min: float = 0.6
    speed_up: float = 1.0              # speed ramp while pushing [m/s^2]
    hold_margin: float = 0.75          # fraction of the ball's rolling decel the pusher may use (see PathRef._profile)
    # ---- car ----
    wheelbase: float = 0.46
    max_steer: float = 0.4189
    scoop_kind: str = 'channel'        # 'channel' (recommended), 'v', 'claw', 'flat'
    scoop_apex_x: float = BODY_FRONT + 0.02   # back of the scoop (where a pushed ball touches it) ahead of the rear axle [m]; ball centre = this + ball_radius
    # ---- steering ----
    k_cross: float = 2.0               # Stanley cross-track gain (push)
    k_line: float = 2.5                # Stanley gain onto the approach line
    k_soft: float = 0.4
    steer_rate_limit: float = 5.0      # [rad/s]
    lead: float = 0.18                 # command-to-motion delay the pose is extrapolated over [s] (cmd_delay + servo lag)
    cmd_delay: float = 0.10            # drive command -> actuator [s]            (identify on the car)
    steer_tau: float = 0.08            # steering servo lag [s]                    (identify on the car)
    speed_tau: float = 0.25            # speed-loop lag [s]                        (identify on the car)
    # ---- approach / chase / search ----
    aim_dist: float = 0.6              # point on the path this far ahead of the ball gives the push direction [m]
    approach_speed: float = 0.7
    touch_speed: float = 0.15          # speed over the ball's own speed at the moment of contact (a soft scoop can take more)
    search_speed: float = 0.8          # speed while the ball is not seen
    approach_decel: float = 1.5
    chase_speed: float = 1.6           # speed while driving to a ball that is not in the scoop (car-only limits apply)
    chase_lat_accel: float = 1.5
    chase_brake: float = 1.0
    final_range: float = 1.6           # along the path, the last stretch before the ball that uses the aligned approach [m]
    stage_dist: float = 0.60           # staging point: this far behind the ball, pointing along the push direction [m]
    stage_clear: float = 0.20          # the approach curve must keep this distance from walls [m]
    stage_kappa: float = 1.0           # ... and not bend sharper than this (front-point curvature) [1/m]
    stage_speed: float = 0.7
    # ---- ball model used by the estimator ----
    ball_radius: float = 0.033
    ball_roll_decel: float = 0.35      # MEASURE THIS: roll the ball on the real floor and fit v(t); everything below scales with it
    meas_sigma: float = 0.015
    accel_sigma: float = 0.5
    gate: float = 25.0
    lost_time: float = 1.2             # not seen for this long and not in contact -> lost [s]

    def __post_init__(self):
        if not (0.2 <= self.hold_margin <= 1.0) or self.max_speed <= 0 or self.ball_roll_decel <= 0:
            raise ValueError('hold_margin must be in 0.2..1, speeds and ball_roll_decel must be positive')


def advance_pose(pose, v, steer, dt, wheelbase, steps=4):
    x, y, yaw = pose
    h = dt / steps
    for _ in range(steps):
        yaw_mid = yaw + 0.5 * h * v * math.tan(steer) / wheelbase
        x += h * v * math.cos(yaw_mid)
        y += h * v * math.sin(yaw_mid)
        yaw += h * v * math.tan(steer) / wheelbase
    return (x, y, yaw)


class ActuatorModel:
    """Remembers the commands that were sent and predicts the car pose over the dead time between pose stamp and the moment
    a new command acts. This is what stops the steering from chattering: the prediction uses the commands that are still
    in flight, not the newest one."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.cmds = collections.deque(maxlen=200)   # (t, v_cmd, steer_cmd)

    def add(self, t, v_cmd, steer_cmd):
        self.cmds.append((t, v_cmd, steer_cmd))

    def _cmd_at(self, t):
        """Command that is active at time t (a command sent at t_c acts from t_c + cmd_delay)."""
        tc = t - self.cfg.cmd_delay
        out = (0.0, 0.0)
        for (t0, vc, sc) in self.cmds:
            if t0 <= tc:
                out = (vc, sc)
            else:
                break
        return out

    def predict(self, pose, v, t_pose, t_target, last_cmd):
        cfg = self.cfg
        # steering servo state at t_pose from the command history, then roll the car forward to t_target
        delta, t = 0.0, t_pose - 0.6
        h = 0.01
        while t < t_pose - 1e-9:
            delta += (self._cmd_at(t)[1] - delta) / cfg.steer_tau * h
            t += h
        x, y, yaw = pose
        t = t_pose
        while t < t_target - 1e-9:
            hh = min(h, t_target - t)
            vc, sc = self._cmd_at(t)
            if t > (self.cmds[-1][0] + cfg.cmd_delay if self.cmds else t):
                vc, sc = last_cmd
            delta += (float(np.clip(sc, -cfg.max_steer, cfg.max_steer)) - delta) / cfg.steer_tau * hh
            v += float(np.clip((vc - v) / cfg.speed_tau, -2.0, 4.0)) * hh
            ym = yaw + 0.5 * hh * v * math.tan(delta) / cfg.wheelbase
            x += hh * v * math.cos(ym)
            y += hh * v * math.sin(ym)
            yaw += hh * v * math.tan(delta) / cfg.wheelbase
            t += hh
        return (x, y, yaw), v


def hermite(p0, t0, p1, t1, n=26):
    """Cubic Hermite curve from p0 (tangent t0) to p1 (tangent t1); returns points and curvature."""
    u = np.linspace(0.0, 1.0, n)[:, None]
    h00, h10, h01, h11 = 2 * u ** 3 - 3 * u ** 2 + 1, u ** 3 - 2 * u ** 2 + u, -2 * u ** 3 + 3 * u ** 2, u ** 3 - u ** 2
    P = h00 * p0 + h10 * t0 + h01 * p1 + h11 * t1
    d1 = (6 * u ** 2 - 6 * u) * p0 + (3 * u ** 2 - 4 * u + 1) * t0 + (-6 * u ** 2 + 6 * u) * p1 + (3 * u ** 2 - 2 * u) * t1
    d2 = (12 * u - 6) * p0 + (6 * u - 4) * t0 + (-12 * u + 6) * p1 + (6 * u - 2) * t1
    kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.maximum((d1 ** 2).sum(1) ** 1.5, 1e-9)
    return P, kappa[:], d1


def poly_stanley(point, yaw, v, poly, k, k_soft):
    """Stanley steering for a short polyline. Returns (steer, e_left, index of the closest vertex)."""
    d = np.hypot(poly[:, 0] - point[0], poly[:, 1] - point[1])
    i = int(np.argmin(d))
    i = min(i, len(poly) - 2)
    seg = poly[i + 1] - poly[i]
    L = max(float(np.hypot(*seg)), 1e-9)
    t = float(np.clip(np.dot(point - poly[i], seg) / (L * L), 0, 1))
    q = poly[i] + t * seg
    e = float(seg[0] * (point[1] - q[1]) - seg[1] * (point[0] - q[0])) / L
    theta = wrap(math.atan2(seg[1], seg[0]) - yaw)
    return theta - math.atan2(k * e, max(abs(v), 0.3) + k_soft), e, i


# =============================================================================================
# Path
# =============================================================================================
class PathRef:
    def __init__(self, cfg):
        d = np.genfromtxt(cfg.path_file, delimiter=',', skip_header=1)
        self.xy = d[:, :2]
        n = len(self.xy)
        nxt = np.roll(self.xy, -1, 0)
        self.seg = np.hypot(*(nxt - self.xy).T)
        self.s = np.concatenate([[0.0], np.cumsum(self.seg)[:-1]])
        self.length = float(self.seg.sum())
        self.yaw = d[:, 2]
        k = d[:, 3]
        kern = np.ones(9) / 9
        self.kappa = np.convolve(np.r_[k[-4:], k, k[:4]], kern, mode='valid')
        self.tree = cKDTree(self.xy)
        self.cfg = cfg
        self.x_b = rider_x(make_scoop(cfg.scoop_kind, apex_x=cfg.scoop_apex_x), cfg.ball_radius)    # ball centre ahead of the rear axle
        self.v_ref = self._profile(cfg)
        self.v_free = self._free_profile(cfg)

    def budget(self):
        """Deceleration budget [m/s^2] a pushed ball can follow: only its own rolling resistance slows it."""
        return self.cfg.hold_margin * self.cfg.ball_roll_decel

    def rear_curvature(self, kappa):
        """Curvature of the rear-axle path when the ball (ahead by x_b) follows a path of curvature kappa."""
        return kappa / np.sqrt(np.maximum(1.0 - (kappa * self.x_b) ** 2, 0.25))

    def _profile(self, cfg):
        """Speed along the path such that the ball stays in the scoop.

        In a turn with yaw rate w, the ball (x_b ahead of the rear axle) has to be pulled BACKWARD relative to the car's
        forward direction by w^2 * x_b, and the only thing that slows a ball is its rolling resistance.  So at every point
              w^2 * x_b + (deceleration of the car) <= hold_margin * ball_roll_decel,      w = v * kappa_rear.
        The pass below applies exactly this, backward for braking and forward for acceleration."""
        mu = self.budget()
        kr = np.abs(self.rear_curvature(self.kappa))
        n = len(kr)
        v = np.full(n, cfg.max_speed)
        v = np.minimum(v, np.sqrt(mu / self.x_b) / np.maximum(kr, 1e-6))          # steady turn: w^2 x_b <= mu
        v = np.maximum(v, cfg.v_min)
        for _ in range(3):
            for i in range(n - 1, -2, -1):
                j, i = (i + 1) % n, i % n
                ds = self.seg[i]
                v[i] = min(v[i], math.sqrt((v[j] ** 2 + 2 * ds * mu) / (1 + 2 * ds * kr[i] ** 2 * self.x_b)))
            for i in range(0, n + 1):
                j, i = (i - 1) % n, i % n
                v[i] = min(v[i], math.sqrt(v[j] ** 2 + 2 * cfg.speed_up * self.seg[j]))
        return v

    def _free_profile(self, cfg):
        """Car-only speed (no ball in the scoop): used to drive to a ball that has rolled away."""
        kr = np.abs(self.rear_curvature(self.kappa))
        v = np.clip(np.sqrt(cfg.chase_lat_accel / np.maximum(kr, 1e-6)), cfg.v_min, cfg.chase_speed)
        n = len(v)
        for _ in range(3):
            for i in range(n - 1, -2, -1):
                v[i % n] = min(v[i % n], math.sqrt(v[(i + 1) % n] ** 2 + 2 * cfg.chase_brake * self.seg[i % n]))
            for i in range(0, n + 1):
                v[i % n] = min(v[i % n], math.sqrt(v[(i - 1) % n] ** 2 + 2 * 1.0 * self.seg[(i - 1) % n]))
        return v

    def free_speed(self, s):
        return float(np.interp(s % self.length, self.s, self.v_free, period=self.length))

    def at(self, s):
        s = s % self.length
        i = int(np.searchsorted(self.s, s, side='right') - 1)
        j = (i + 1) % len(self.xy)
        t = (s - self.s[i]) / max(self.seg[i], 1e-9)
        xy = self.xy[i] + t * (self.xy[j] - self.xy[i])
        yaw = math.atan2((1 - t) * math.sin(self.yaw[i]) + t * math.sin(self.yaw[j]),
                         (1 - t) * math.cos(self.yaw[i]) + t * math.cos(self.yaw[j]))
        return xy, yaw, (1 - t) * self.kappa[i] + t * self.kappa[j], (1 - t) * self.v_ref[i] + t * self.v_ref[j]

    def project(self, p):
        """-> (s, e_left, yaw, kappa, v_ref) of the closest path point; e_left > 0 when p is left of the path."""
        _, i0 = self.tree.query(p)
        best = None
        n = len(self.xy)
        for i in (i0 - 1, i0):
            i %= n
            j = (i + 1) % n
            ab = self.xy[j] - self.xy[i]
            t = float(np.clip(np.dot(p - self.xy[i], ab) / max(np.dot(ab, ab), 1e-12), 0, 1))
            q = self.xy[i] + t * ab
            d = float(np.hypot(*(p - q)))
            if best is None or d < best[0]:
                best = (d, i, j, t, q, ab)
        _, i, j, t, q, ab = best
        e = float(ab[0] * (p[1] - q[1]) - ab[1] * (p[0] - q[0])) / max(float(np.hypot(*ab)), 1e-9)
        s = self.s[i] + t * self.seg[i]
        yaw = math.atan2(ab[1], ab[0])
        return s, e, yaw, (1 - t) * self.kappa[i] + t * self.kappa[j], (1 - t) * self.v_ref[i] + t * self.v_ref[j]


# =============================================================================================
# Ball estimator
# =============================================================================================
class BallTracker:
    def __init__(self, cfg):
        self.cfg = cfg
        self.prm = BallParams(radius=cfg.ball_radius, roll_decel=cfg.ball_roll_decel)
        self.scoop = make_scoop(cfg.scoop_kind, apex_x=cfg.scoop_apex_x)
        self.hist = collections.deque(maxlen=800)
        self.reset()

    def reset(self):
        self.has = False
        self.t = 0.0
        self.p = np.zeros(2)
        self.v = np.zeros(2)
        self.P = np.diag([1.0, 1.0, 1.0, 1.0])
        self.snaps = collections.deque(maxlen=80)
        self.last_seen = -1e9
        self.touch_t = -1e9
        self.rejected = 0

    # ---- car pose history ----
    def add_pose(self, t, pose, v, w):
        self.hist.append((t, pose[0], pose[1], pose[2], v, w))

    def pose_at(self, t):
        h = self.hist
        if not h:
            return (0.0, 0.0, 0.0, 0.0, 0.0)
        if t >= h[-1][0]:
            t0, x, y, yaw, v, w = h[-1]
            dt = t - t0
            ym = yaw + 0.5 * w * dt
            return (x + v * math.cos(ym) * dt, y + v * math.sin(ym) * dt, yaw + w * dt, v, w)
        if t <= h[0][0]:
            return h[0][1:]
        lo, hi = 0, len(h) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if h[mid][0] <= t:
                lo = mid
            else:
                hi = mid
        a, b = h[lo], h[hi]
        f = (t - a[0]) / max(b[0] - a[0], 1e-9)
        yaw = a[3] + f * wrap(b[3] - a[3])
        return (a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2]), yaw, a[4] + f * (b[4] - a[4]), a[5] + f * (b[5] - a[5]))

    # ---- filter ----
    def _propagate(self, t_to):
        q = self.cfg.accel_sigma ** 2
        while self.t < t_to - 1e-9:
            dt = min(0.01, t_to - self.t)
            ta = self.t + dt
            px, py, pyaw, pv, pw = self.pose_at(ta)
            scoop_w = car_to_world((px, py, pyaw), self.scoop)
            vel_at = car_velocity_field((px, py, pyaw), pv, pw)
            self.p, self.v, touching = ball_step(self.p, self.v, dt, self.prm, scoop_w, vel_at, None)
            F = np.eye(4)
            F[0, 2] = F[1, 3] = dt
            Q = np.diag([1e-6, 1e-6, q * dt, q * dt])
            self.P = F @ self.P @ F.T + Q
            if touching:
                self.touch_t = ta
                self.P[2, 2] += 0.03 ** 2 * dt * 20
                self.P[3, 3] += 0.03 ** 2 * dt * 20
            self.t = ta
        self.snaps.append((self.t, self.p.copy(), self.v.copy(), self.P.copy()))

    def predict_to(self, t):
        if self.has and t > self.t:
            self._propagate(t)

    def update(self, t_meas, z_car):
        """z_car: ball centre in the car frame (x forward, y left) as seen at time t_meas."""
        z_car = np.asarray(z_car, float)
        pose = self.pose_at(t_meas)
        zw = car_to_world(pose[:3], z_car)[0]
        r = float(np.hypot(*z_car))
        sig = self.cfg.meas_sigma + 0.01 * r * r
        R = np.eye(2) * sig ** 2
        if not self.has:
            self.p, self.v, self.t = zw.copy(), np.zeros(2), t_meas
            self.P = np.diag([sig ** 2, sig ** 2, 0.5 ** 2, 0.5 ** 2])
            self.last_seen, self.has = t_meas, True
            self.snaps.clear()
            return
        t_now = self.t
        if t_meas < self.t - 1e-6:                       # old image: rewind to the nearest earlier snapshot, then replay
            snap = None
            for s in self.snaps:
                if s[0] <= t_meas:
                    snap = s
            if snap is None:
                snap = self.snaps[0] if self.snaps else (self.t, self.p, self.v, self.P)
            self.t, self.p, self.v, self.P = snap[0], snap[1].copy(), snap[2].copy(), snap[3].copy()
        self._propagate(max(t_meas, self.t))
        S = self.P[:2, :2] + R
        innov = zw - self.p
        maha = float(innov @ np.linalg.solve(S, innov))
        gone = (t_meas - self.last_seen) > self.cfg.lost_time
        if maha > self.cfg.gate and not gone:
            self.rejected += 1
        else:
            if gone:                                    # re-acquire: trust the image, widen the velocity uncertainty
                self.p, self.v = zw.copy(), np.zeros(2)
                self.P = np.diag([sig ** 2, sig ** 2, 0.5 ** 2, 0.5 ** 2])
            else:
                K = self.P[:, :2] @ np.linalg.inv(S)
                x = np.r_[self.p, self.v] + K @ innov
                self.p, self.v = x[:2], x[2:]
                self.P = (np.eye(4) - K @ np.eye(4)[:2]) @ self.P
            self.last_seen = max(self.last_seen, t_meas)
        self._propagate(t_now)

    # ---- outputs ----
    def contact(self, t):
        return self.has and (t - self.touch_t) < 0.25

    def pos_std(self):
        return float(math.sqrt(max(self.P[0, 0], self.P[1, 1])))


# =============================================================================================
# Controller
# =============================================================================================
class BallPusher:
    def __init__(self, cfg=None, walls=None):
        """walls: optional object with dist(x, y) = distance to the nearest wall [m] (from the map); enables the staged approach."""
        self.cfg = cfg or Config()
        self.walls = walls
        self.path = PathRef(self.cfg)
        self.tracker = BallTracker(self.cfg)
        self.scoop = make_scoop(self.cfg.scoop_kind, apex_x=self.cfg.scoop_apex_x)
        self.apex = float(self.scoop[:, 0].min())
        self.rider = rider_x(self.scoop, self.cfg.ball_radius)                   # where a pushed ball's centre sits
        self.v_cmd = 0.0
        self.steer_cmd = 0.0
        self.mode = 'SEARCH'
        self.last_t = None
        self.act = ActuatorModel(self.cfg)
        self._psi_prev = None
        self.info = {}

    def observe(self, t_meas, z_car):
        self.tracker.update(t_meas, z_car)

    # ---- helpers ----
    def _stanley(self, point, yaw, v, s_proj=None):
        s, e, psi, kap, vref = self.path.project(point) if s_proj is None else s_proj
        theta = wrap(psi - yaw)
        e = max(-0.35, min(0.35, e))
        return theta - math.atan2(self.cfg.k_cross * e, max(abs(v), 0.3) + self.cfg.k_soft), (s, e, psi, kap, vref)

    def _plan_stage(self, F, yaw, B, u0):
        """Curve from the scoop apex F (heading yaw) to a staging point behind the ball B, arriving along a push direction u.

        u is the wanted direction u0 or, when that is blocked (ball resting against a wall: there is no room behind it), the
        nearest direction that is possible: the ball is then pushed along the wall until it is free.  Tries several
        directions, staging distances and tangent lengths; keeps the cheapest curve that clears the walls and is drivable.
        Returns (points, staging point, u) or None."""
        cfg = self.cfg
        best = None
        h0 = np.array([math.cos(yaw), math.sin(yaw)])
        psi0 = math.atan2(u0[1], u0[0])
        for dpsi in (0.0, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2, 1.6, -1.6):
            psi = psi0 + dpsi
            u = np.array([math.cos(psi), math.sin(psi)])
            if self.walls is not None:                                      # the ball needs room to roll along u
                ahead = B[None, :] + np.outer([0.15, 0.4, 0.8], u)
                if float(np.min(self.walls.dist(ahead[:, 0], ahead[:, 1]))) < cfg.ball_radius + 0.04:
                    continue
            prefer = 0.0 if self._psi_prev is None else 0.6 * abs(wrap(psi - self._psi_prev))
            for g in (cfg.stage_dist, cfg.stage_dist + 0.3, cfg.stage_dist + 0.7):
                G = B - g * u
                dist = float(np.hypot(*(G - F)))
                if dist < 0.05:
                    continue
                if self.walls is not None and float(self.walls.dist(G[0], G[1])) < cfg.stage_clear:
                    continue
                for k0, k1 in ((0.8, 0.8), (1.3, 0.8), (0.5, 0.6), (1.6, 1.2)):
                    P, kap, _ = hermite(F, h0 * dist * k0, G, u * dist * k1)
                    if np.abs(kap).max() > cfg.stage_kappa:
                        continue
                    if self.walls is not None and float(np.min(self.walls.dist(P[:, 0], P[:, 1]))) < cfg.stage_clear:
                        continue
                    L = float(np.sum(np.hypot(*np.diff(P, axis=0).T)))
                    cost = L + 2.0 * float(np.abs(kap).max()) + 0.5 * g + 1.5 * abs(dpsi) + prefer
                    if best is None or cost < best[0]:
                        best = (cost, P, G, u, psi)
        if best is None:
            self._psi_prev = None
            return None
        self._psi_prev = best[4]
        return best[1], best[2], best[3]

    def _slew(self, target, dt, up, down):
        if target > self.v_cmd:
            self.v_cmd = min(target, self.v_cmd + up * dt)
        else:
            self.v_cmd = max(target, self.v_cmd - down * dt)

    def update(self, t, pose, v, yaw_rate=0.0, t_pose=None):
        """t: now; pose/v/yaw_rate: latest car state, stamped t_pose (<= t). Returns (speed command, steering command)."""
        cfg, tr = self.cfg, self.tracker
        t_pose = t if t_pose is None else t_pose
        dt = 0.05 if self.last_t is None else max(1e-3, t - self.last_t)
        self.last_t = t
        tr.add_pose(t_pose, pose, v, yaw_rate)
        tr.predict_to(t)
        pc, _ = self.act.predict(pose, v, t_pose, t + cfg.lead, (self.v_cmd, self.steer_cmd))   # pose when this command acts
        in_contact = tr.contact(t)
        F = car_to_world(pc, np.array([[self.apex, 0.0]]))[0]                     # scoop apex in the world
        rider = car_to_world(pc, np.array([[self.rider, 0.0]]))[0]
        # a ball predicted inside the scoop region counts as held, even when the camera cannot see it (blind zone)
        near = False
        if tr.has:
            bc = world_to_car(pc, tr.p)[0]
            near = (self.apex - 0.04 < bc[0] < self.apex + 0.25) and abs(bc[1]) < 0.20
        held = in_contact or near
        lost = (not tr.has) or ((t - tr.last_seen) > cfg.lost_time and not held)
        steer = 0.0
        v_target, down = 0.0, cfg.approach_decel
        mode = 'SEARCH' if lost else ('PUSH' if held else 'APPROACH')

        if mode == 'PUSH':
            steer, (s, e, psi, kap, vref) = self._stanley(rider, pc[2], v)
            ahead = self.path.at(s + max(0.0, v) * cfg.lead + 0.2)[3]
            kr = float(self.path.rear_curvature(kap))
            down = max(0.03, self.path.budget() - (max(v, 0.0) * kr) ** 2 * self.path.x_b)    # what the ball can follow right now
            v_target = min(vref, ahead) / (1.0 + 2.5 * max(0.0, abs(e) - 0.05))     # ball off the path: go gently until it is back
            up = cfg.speed_up
        elif mode == 'APPROACH':
            w = tr.v
            sp = float(np.hypot(*w))
            T = cfg.lead
            disp = min(sp * T - 0.5 * cfg.ball_roll_decel * T * T, sp * sp / (2 * cfg.ball_roll_decel)) if sp > 1e-3 else 0.0
            B = tr.p + (w / sp * disp if sp > 1e-3 else 0.0)
            s_b, e_b, psi_b, kap_b, vref_b = self.path.project(B)
            s_c = self.path.project(F)[0]
            d = (s_b - s_c) % self.path.length
            if d > self.path.length / 2:
                d -= self.path.length                                              # signed distance to the ball along the path
            near_ball = float(np.hypot(*(B - F))) < 2.2 and abs(e_b) > 0.25            # ball resting away from the path: plan to it directly
            if (-0.1 < d < cfg.final_range) or near_ball:
                # --- final approach: line "behind the ball, pointing along the path", Stanley onto that line ---
                A = self.path.at(s_b + cfg.aim_dist)[0]
                u = A - B
                if abs(e_b) > 0.25:                                                # ball off the path: push it back toward it
                    u = u + 0.8 * (self.path.at(s_b)[0] - B)
                u = u / max(float(np.hypot(*u)), 1e-6)
                plan = self._plan_stage(F, pc[2], B, u)
                if plan is not None:
                    u = plan[2]                                                    # push direction actually possible here
                rel = F - B
                a_along = float(rel @ u)
                e_line = float(u[0] * rel[1] - u[1] * rel[0])
                theta = wrap(math.atan2(u[1], u[0]) - pc[2])
                gap = -a_along - cfg.ball_radius
                aligned = abs(e_line) < 0.18 and abs(theta) < 0.5 and a_along < 0.0
                if aligned and gap < 1.0:
                    plan = None
                if plan is not None:
                    P, G, _ = plan
                    steer, _, _ = poly_stanley(F, pc[2], v, P, cfg.k_line, cfg.k_soft)
                    v_target = min(cfg.stage_speed, self.path.free_speed(s_c) + 0.3)
                    gap = 1.0                                                      # not touching yet
                else:
                    steer = theta - math.atan2(cfg.k_line * e_line, max(abs(v), 0.3) + cfg.k_soft)
                    v_ball = max(0.0, float(w @ u))
                    closing = float(np.clip(0.8 * gap + cfg.touch_speed, cfg.touch_speed, cfg.approach_speed))
                    v_target = v_ball + closing
                    if abs(theta) > 0.5 or abs(e_line) > 0.25:
                        v_target = min(v_target, 0.5)
                    if gap < 0.25 and abs(e_line) > 0.06:
                        v_target = min(v_target, v_ball + cfg.touch_speed)
                    v_target = min(v_target, cfg.max_speed, self.path.free_speed(s_c))
                mode = 'APPROACH'
            else:
                # --- chase: follow the path (forward, always: a ball that ended up behind is reached by going around the loop) ---
                steer, (s, e, psi, kap, vref) = self._stanley(F, pc[2], v)
                v_ball_s = max(0.0, float(w @ np.array([math.cos(psi_b), math.sin(psi_b)])))
                v_target = min(self.path.free_speed(s_c + max(v, 0) * cfg.lead), cfg.chase_speed)
                if 0 < d < cfg.final_range + 2.0:                                  # about to start the final approach: slow down
                    v_target = min(v_target, cfg.approach_speed + v_ball_s + 0.6 * (d - cfg.final_range))
                mode = 'CHASE'
            up = cfg.speed_up * 1.5
            if mode == 'APPROACH' and gap < 0.2:
                down = self.path.budget()
            elif mode == 'CHASE':
                down = cfg.chase_brake
        else:  # SEARCH
            steer, (s, e, psi, kap, vref) = self._stanley(F, pc[2], v)
            v_target = cfg.search_speed if tr.has else 0.0
            up = 0.8

        self._slew(max(v_target, 0.0), dt, up if mode != 'PUSH' else cfg.speed_up, down)
        steer = float(np.clip(steer, -cfg.max_steer, cfg.max_steer))
        step = cfg.steer_rate_limit * dt
        self.steer_cmd = float(np.clip(steer, self.steer_cmd - step, self.steer_cmd + step))
        self.act.add(t, self.v_cmd, self.steer_cmd)
        self.mode = mode
        self.info = dict(mode=mode, contact=in_contact, ball=tr.p.copy(), ball_std=tr.pos_std(), age=t - tr.last_seen)
        return self.v_cmd, self.steer_cmd


# =============================================================================================
# ROS parameter helper (every Config field is a ROS parameter of the same name)
# =============================================================================================
def config_fields():
    return {f.name: f.default for f in fields(Config)}
