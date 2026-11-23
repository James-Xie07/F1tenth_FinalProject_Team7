"""Frenet sampling + linearized kinematic-bicycle MPC for Levine ball pushing.

ROS-free module. Real-car CSV describes the ball-centre reference path and
is shifted backward to obtain the rear-axle path. Gym CSV is the vehicle
raceline and must not be shifted. Select via reference_is_ball_path.
This is an independent, low-speed adaptation of the team's Frenet MPC formulation
(z=[x,y,v,yaw], u=[acceleration,steering], CVXPY/OSQP), not a verbatim copy.
The Stanley controller remains untouched in ball_control.py.
"""
from dataclasses import dataclass
import math

try:
    import cvxpy as cp
except ImportError:  # Allows path/recovery geometry tests before installing solver
    cp = None
import numpy as np
from scipy.spatial import cKDTree


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


@dataclass
class MPCConfig:
    dt: float = 0.10
    horizon: int = 10
    solver_warm_start: bool = True
    wheelbase: float = 0.46  # effective wheelbase of the team's real-car Frenet controller
    max_speed: float = 0.50
    max_steer: float = 0.4189
    max_steer_rate: float = 2.0
    max_accel: float = 0.45
    max_brake: float = 0.55
    q_xy: float = 14.0
    q_yaw: float = 10.0
    q_speed: float = 6.0
    q_ball_lateral: float = 28.0
    r_accel: float = 0.10
    r_steer: float = 0.6
    rd_steer: float = 5.0
    # Road is narrow: keep offsets conservative and verify using /map.
    lateral_max: float = 0.12
    lateral_candidates: int = 7
    converge_length: float = 1.2
    planning_distance: float = 1.9
    car_length: float = 0.70  # includes front pushing mechanism (conservative)
    car_half_width: float = 0.155
    car_center_x: float = 0.19
    wall_margin: float = 0.035
    obstacle_margin: float = 0.11
    # CSV describes ball-centre path; the car's rear axle follows behind the ball.
    ball_center_ahead: float = 0.540

    def __post_init__(self):
        if not 0 < self.max_speed <= 0.5:
            raise ValueError('max_speed cannot exceed 0.5 m/s')
        if self.wheelbase <= 0 or self.horizon < 2 or self.dt <= 0:
            raise ValueError('invalid MPC geometry/horizon')


class LevinePath:
    def __init__(self, filename, cfg, reference_is_ball_path=True):
        raw = np.genfromtxt(filename, delimiter=',', names=True)
        if raw.size < 6 or not all(k in raw.dtype.names for k in ('x', 'y')):
            raise ValueError('Levine CSV needs columns x,y and at least six rows')
        reference = np.column_stack([raw['x'], raw['y']]).astype(float)
        if np.linalg.norm(reference[0] - reference[-1]) < 1e-5:
            reference = reference[:-1]
        if not np.all(np.isfinite(reference)):
            raise ValueError('non-finite waypoints')
        if reference_is_ball_path:
            # Real car: the ball reference is in front of the rear axle.
            tang = np.roll(reference, -1, axis=0) - np.roll(reference, 1, axis=0)
            tang /= np.maximum(np.linalg.norm(tang, axis=1, keepdims=True), 1e-8)
            rear = reference - cfg.ball_center_ahead * tang
        else:
            # Gym raceline already describes the vehicle reference path.
            rear = reference.copy()
        spacing = np.linalg.norm(np.roll(rear, -1, axis=0) - rear, axis=1)
        keep = spacing > 1e-5
        rear = rear[keep]
        if len(rear) < 6:
            raise ValueError('degenerate rear-axle path')
        self.xy = rear
        self.seg = np.linalg.norm(np.roll(rear, -1, axis=0) - rear, axis=1)
        self.s = np.r_[0.0, np.cumsum(self.seg[:-1])]
        self.length = float(np.sum(self.seg))
        self.tree = cKDTree(rear)
        t = np.roll(rear, -1, axis=0) - np.roll(rear, 1, axis=0)
        self.yaw = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
        self.cfg = cfg

    def at(self, s):
        s = np.atleast_1d(np.asarray(s, dtype=float)) % self.length
        i = np.searchsorted(self.s, s, side='right') - 1
        i = np.maximum(i, 0)
        j = (i + 1) % len(self.xy)
        f = (s - self.s[i]) / np.maximum(self.seg[i], 1e-9)
        xy = self.xy[i] * (1 - f[:, None]) + self.xy[j] * f[:, None]
        delta = np.arctan2(np.sin(self.yaw[j] - self.yaw[i]), np.cos(self.yaw[j] - self.yaw[i]))
        yaw = self.yaw[i] + f * delta
        return xy, yaw

    def project(self, pos):
        pos = np.asarray(pos, dtype=float)
        _, nearest = self.tree.query(pos)
        best = None
        for i in ((int(nearest) - 1) % len(self.xy), int(nearest)):
            j = (i + 1) % len(self.xy)
            a, v = self.xy[i], self.xy[j] - self.xy[i]
            f = float(np.clip(np.dot(pos - a, v) / max(np.dot(v, v), 1e-10), 0, 1))
            p = a + f * v
            dist = float(np.linalg.norm(pos - p))
            if best is None or dist < best[0]:
                yaw = math.atan2(v[1], v[0])
                lateral = -math.sin(yaw) * (pos[0] - p[0]) + math.cos(yaw) * (pos[1] - p[1])
                best = (dist, self.s[i] + f * self.seg[i], lateral, yaw)
        return best[1:]


def quintic_offset(d0, slope, target, sigma, L):
    L = max(float(L), 0.2)
    m = np.array([[L**3, L**4, L**5],
                  [3*L**2, 4*L**3, 5*L**4],
                  [6*L, 12*L**2, 20*L**3]], float)
    a, b, c = np.linalg.solve(m, [target - d0 - slope*L, -slope, 0.0])
    q = np.minimum(np.asarray(sigma, float), L)
    return d0 + slope*q + a*q**3 + b*q**4 + c*q**5


def body_clearance(walls, xy, yaw, cfg):
    if walls is None:
        return np.inf
    # Cover the rectangle using 5 longitudinal strips and a circumscribed disc.
    seg = cfg.car_length / 5.0
    radius = math.hypot(cfg.car_half_width, seg / 2.0)
    offsets = cfg.car_center_x + (np.arange(5) - 2) * seg
    px = xy[:, 0, None] + offsets * np.cos(yaw[:, None])
    py = xy[:, 1, None] + offsets * np.sin(yaw[:, None])
    return float(np.min(walls.dist(px, py))) - radius


def obstacle_clearance(points, xy, yaw, cfg):
    """Minimum body-to-LiDAR-obstacle clearance, using circumscribed discs."""
    if points is None or len(points) == 0:
        return np.inf
    points = np.asarray(points,float)
    seg = cfg.car_length / 5.0
    radius = math.hypot(cfg.car_half_width,seg/2)
    offsets = cfg.car_center_x + (np.arange(5)-2)*seg
    px = xy[:,0,None]+offsets*np.cos(yaw[:,None])
    py = xy[:,1,None]+offsets*np.sin(yaw[:,None])
    near,_ = cKDTree(points).query(np.column_stack((px.ravel(),py.ravel())))
    return float(near.min()-radius)


class FrenetPlanner:
    def __init__(self, path, cfg):
        self.path, self.cfg = path, cfg
        self.prev_offset = 0.0
        self.last_diagnostics = {}

    def plan(self, state, walls=None, speed=None, obstacle_points=None):
        cfg = self.cfg
        s0, d0, psi0 = self.path.project(state[:2])
        err = wrap(float(state[3]) - psi0)
        slope = float(np.clip(math.tan(np.clip(err, -0.65, 0.65)), -0.7, 0.7))
        dist = np.arange(0.0, cfg.planning_distance + 1e-6, 0.05)
        base_xy, base_yaw = self.path.at(s0 + dist)
        normal = np.column_stack([-np.sin(base_yaw), np.cos(base_yaw)])
        targets = np.linspace(-cfg.lateral_max, cfg.lateral_max, cfg.lateral_candidates)
        candidates = []
        # Sequential rejection counts diagnose why all 7 candidate paths fail.
        # These counters change only logging, never planning or safety checks.
        rejected = dict(curvature=0, map_wall=0, lidar_obstacle=0)
        best_wall = -float('inf')
        best_lidar = -float('inf')
        for offset in targets:
            d = quintic_offset(d0, slope, offset, dist, cfg.converge_length)
            xy = base_xy + d[:, None] * normal
            deriv = np.gradient(xy, axis=0)
            yaw = np.unwrap(np.arctan2(deriv[:, 1], deriv[:, 0]))
            step = np.maximum(np.linalg.norm(np.diff(xy, axis=0), axis=1), 1e-5)
            curvature = np.diff(yaw) / step
            # Frenet path must be feasible with the actual steering limit.
            if len(curvature) and np.max(np.abs(curvature[1:-1])) > 1.15 * math.tan(cfg.max_steer)/cfg.wheelbase:
                rejected['curvature'] += 1
                continue
            clearance = body_clearance(walls, xy, yaw, cfg)
            best_wall = max(best_wall, clearance)
            if clearance < cfg.wall_margin:
                rejected['map_wall'] += 1
                continue
            lidar_clearance = obstacle_clearance(obstacle_points, xy, yaw, cfg)
            best_lidar = max(best_lidar, lidar_clearance)
            if lidar_clearance < cfg.obstacle_margin:
                rejected['lidar_obstacle'] += 1
                continue
            cost = 3.0 * np.mean(d*d) + 0.7 * (offset - self.prev_offset)**2
            cost += 0.08 / max(clearance, 0.1) if np.isfinite(clearance) else 0.0
            candidates.append((cost, float(offset), xy, yaw))
        self.last_diagnostics = dict(
            candidate_count=len(targets), accepted=len(candidates),
            rejected=rejected,
            nearest_s_m=round(float(s0), 3),
            lateral_error_m=round(float(d0), 3),
            heading_error_deg=round(math.degrees(err), 1),
            best_map_body_gap_m=(round(float(best_wall), 3)
                                 if np.isfinite(best_wall) else 'not_checked'),
            best_lidar_body_gap_m=(round(float(best_lidar), 3)
                                   if np.isfinite(best_lidar) else 'not_checked'),
            lidar_obstacle_points=(0 if obstacle_points is None else len(obstacle_points)),
            # The wall gap from the car's CURRENT pose helps distinguish a
            # blocked start from a collision predicted farther down the path.
            current_map_body_gap_m=(round(body_clearance(
                walls, np.asarray([state[:2]]), np.asarray([state[3]]), cfg), 3)
                if walls is not None else 'no_map'),
        )
        if not candidates:
            return None
        _, offset, chosen, headings = min(candidates, key=lambda c: c[0])
        self.prev_offset = offset
        arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(chosen, axis=0), axis=1))]
        goal_speed = float(np.clip(cfg.max_speed if speed is None else speed, 0, cfg.max_speed))
        # Do not force a minimum forward speed; it must be able to stop.
        travelled = np.arange(cfg.horizon + 1) * cfg.dt * goal_speed
        travelled = np.minimum(travelled, arc[-1])
        reference = np.empty((4, cfg.horizon + 1))
        reference[0] = np.interp(travelled, arc, chosen[:, 0])
        reference[1] = np.interp(travelled, arc, chosen[:, 1])
        yaw_ref = headings + 2*math.pi*round((state[3]-headings[0])/(2*math.pi))
        reference[3] = np.interp(travelled, arc, yaw_ref)
        reference[2] = goal_speed
        return reference, chosen, offset


def bicycle_step(x, u, dt, wheelbase):
    px, py, v, psi = map(float, x)
    a, delta = map(float, u)
    vm = v + .5 * dt * a
    mid = psi + .5 * dt * vm * math.tan(delta) / wheelbase
    return np.array([px + dt*vm*math.cos(mid), py + dt*vm*math.sin(mid),
                     v + dt*a, psi + dt*vm*math.tan(delta)/wheelbase])


def linearize(x, u, dt, wheelbase):
    # Numerical Jacobian of the same midpoint model, to avoid a mismatched dynamics model.
    x = np.asarray(x, float); u = np.asarray(u, float)
    base = bicycle_step(x, u, dt, wheelbase)
    A, B = np.zeros((4,4)), np.zeros((4,2))
    h = 1e-5
    for i in range(4):
        xp = x.copy(); xp[i] += h
        xm = x.copy(); xm[i] -= h
        A[:,i] = (bicycle_step(xp, u, dt, wheelbase) - bicycle_step(xm, u, dt, wheelbase)) / (2*h)
    for i in range(2):
        up = u.copy(); up[i] += h
        um = u.copy(); um[i] -= h
        B[:,i] = (bicycle_step(x, up, dt, wheelbase) - bicycle_step(x, um, dt, wheelbase)) / (2*h)
    return A, B, base - A@x - B@u


class TrackingMPC:
    def __init__(self, cfg):
        if cp is None:
            raise ImportError('Frenet MPC requires cvxpy and osqp: install python3-cvxpy python3-osqp')
        self.c = cfg
        N = cfg.horizon
        self.X = cp.Variable((4, N+1))
        self.U = cp.Variable((2, N))
        self.x0 = cp.Parameter(4)
        self.ref = cp.Parameter((4, N+1))
        self.A = [cp.Parameter((4,4)) for _ in range(N)]
        self.B = [cp.Parameter((4,2)) for _ in range(N)]
        self.C = [cp.Parameter(4) for _ in range(N)]
        self.last_steer = cp.Parameter()
        self.ball_n = cp.Parameter((2,N+1))
        self.ball_goal = cp.Parameter(N+1)
        self.ball_weight = cp.Parameter(nonneg=True)
        constraints = [self.X[:,0] == self.x0,
                       self.X[2,:] >= 0.0, self.X[2,:] <= cfg.max_speed,
                       self.U[0,:] >= -cfg.max_brake, self.U[0,:] <= cfg.max_accel,
                       self.U[1,:] >= -cfg.max_steer, self.U[1,:] <= cfg.max_steer,
                       self.U[1,0] - self.last_steer <= cfg.max_steer_rate * cfg.dt,
                       self.last_steer - self.U[1,0] <= cfg.max_steer_rate * cfg.dt]
        obj = 0
        for k in range(N+1):
            e = self.X[:,k] - self.ref[:,k]
            obj += cfg.q_xy*cp.sum_squares(e[:2]) + cfg.q_yaw*cp.square(e[3]) + cfg.q_speed*cp.square(e[2])
            lateral = cp.multiply(self.ball_n[0,k],self.X[0,k]) + cp.multiply(self.ball_n[1,k],self.X[1,k])
            obj += self.ball_weight * cp.square(lateral - self.ball_goal[k])
            if k < N:
                constraints.append(self.X[:,k+1] == self.A[k] @ self.X[:,k] + self.B[k] @ self.U[:,k] + self.C[k])
                obj += cfg.r_accel*cp.square(self.U[0,k]) + cfg.r_steer*cp.square(self.U[1,k])
                if k:
                    constraints += [cp.abs(self.U[1,k] - self.U[1,k-1]) <= cfg.max_steer_rate*cfg.dt]
                    obj += cfg.rd_steer*cp.square(self.U[1,k] - self.U[1,k-1])
        self.problem = cp.Problem(cp.Minimize(obj), constraints)
        self.last_plan = None

    def solve(self, state, reference, steer_previous, ball_lateral=0.0, use_ball=False):
        c = self.c
        state = np.asarray(state, float).copy()
        if not np.all(np.isfinite(state)):
            raise ValueError('invalid state')
        # Avoid infeasible QP after small VESC speed overshoots.
        state[2] = float(np.clip(state[2], 0, c.max_speed))
        reference = np.asarray(reference, float)
        if reference.shape != (4, c.horizon+1):
            raise ValueError('reference shape mismatch')
        self.x0.value = state
        self.ref.value = reference
        self.last_steer.value = float(np.clip(steer_previous,-c.max_steer,c.max_steer))
        self.ball_weight.value = c.q_ball_lateral if use_ball else 0.0
        headings = reference[3]
        normal = np.vstack((-np.sin(headings), np.cos(headings)))
        self.ball_n.value = normal
        # The desired rear-axle side position is adjusted by the measured ball lateral error.
        # Use a bounded correction, so outliers cannot cause a large steering command.
        dy = float(np.clip(ball_lateral, -0.06, 0.06)) if use_ball else 0.0
        self.ball_goal.value = np.sum(normal * reference[:2], axis=0) + dy
        if self.last_plan is None:
            accel = np.zeros(c.horizon)
            steer = np.full(c.horizon, self.last_steer.value)
        else:
            accel = np.r_[self.last_plan[0,1:], self.last_plan[0,-1]]
            steer = np.r_[self.last_plan[1,1:], self.last_plan[1,-1]]
        nominal = state.copy()
        for k in range(c.horizon):
            lin = nominal.copy()
            lin[2] = max(lin[2], 0.12)
            A,B,C = linearize(lin, (accel[k],steer[k]), c.dt, c.wheelbase)
            self.A[k].value, self.B[k].value, self.C[k].value = A, B, C
            nominal = bicycle_step(nominal, (accel[k],steer[k]), c.dt,c.wheelbase)
        # Temporary robustness fix for the numeric-Jacobian MPC.  The sparse
        # constraint pattern can change between linearizations.  OSQP cannot
        # update a cached workspace when the number of nonzeros changes.
        # Force a fresh solver setup every cycle.  This is slower, so profile
        # solve time before enabling a real vehicle; replace with a fixed
        # sparsity QP for production use.
        self.problem.solve(solver=cp.OSQP, warm_start=c.solver_warm_start, verbose=False,
                           eps_abs=1e-3, eps_rel=1e-3, max_iter=2500)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.U.value is None:
            self.last_plan = None
            raise RuntimeError('Frenet MPC failed: %s' % self.problem.status)
        if not np.all(np.isfinite(self.U.value)):
            raise RuntimeError('MPC nonfinite solution')
        self.last_plan = self.U.value.copy()
        v_next = float(np.clip(state[2] + self.U.value[0,0]*c.dt, 0, c.max_speed))
        steer_cmd = float(np.clip(self.U.value[1,0], -c.max_steer,c.max_steer))
        return v_next, steer_cmd, self.X.value.copy()

    def reset(self):
        self.last_plan = None


def local_reference(points, state, cfg, speed):
    """Convert a collision-checked Hermite recovery curve to an MPC reference."""
    points = np.asarray(points, float)
    if len(points) < 3:
        return None
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points,axis=0),axis=1))]
    if arc[-1] < 1e-4:
        return None
    diff = np.gradient(points, axis=0)
    yaw = np.unwrap(np.arctan2(diff[:,1],diff[:,0]))
    yaw += 2*math.pi*round((state[3]-yaw[0])/(2*math.pi))
    preview = np.minimum(np.arange(cfg.horizon+1) * cfg.dt*speed,arc[-1])
    r = np.zeros((4,cfg.horizon+1))
    r[0] = np.interp(preview, arc, points[:,0]); r[1] = np.interp(preview,arc,points[:,1]);
    r[3] = np.interp(preview,arc,yaw)
    r[2] = float(np.clip(speed,0,cfg.max_speed))
    return r
