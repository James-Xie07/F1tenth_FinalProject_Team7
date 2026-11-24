"""Physics shared by the simulator and by the controller's ball estimator.

Frames: world (x, y), car frame = origin at the REAR AXLE, x forward, y left (same as the Frenet node).
A "scoop" is a polyline attached to the front of the car; the ball is a rolling disc that is pushed by it.
"""
import math
from dataclasses import dataclass
import numpy as np


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------------------------------------
# Car geometry (same numbers as the Frenet node's Config)
# ---------------------------------------------------------------------------------------------
BODY_FRONT = 0.1651 + 0.58 / 2          # front edge of the body, ahead of the rear axle [m]
BODY_REAR = 0.1651 - 0.58 / 2
HALF_WIDTH = 0.155


EXTRA_SCOOPS = {}      # name -> f(apex_x) -> polyline; lets experiments try other shapes without editing make_scoop


def make_scoop(kind='v', apex_x=BODY_FRONT + 0.02, depth=0.14, half_width=0.15, claw_in=0.05):
    """Polyline of points in the car frame (x forward, y left).

    'v'    : two straight arms, apex on the centre line, opening forward (self-centring pusher).
    'claw' : the V, but the arm tips curve back inward, so the ball is also held when the car slows down.
    'flat' : a plain bumper (for comparison; the ball slides off sideways).
    'arc'  : the real printed pusher: a shallow concave arc, R = 160 mm (see below).
    """
    if kind in EXTRA_SCOOPS:
        return EXTRA_SCOOPS[kind](apex_x)
    if kind == 'v':
        return np.array([[apex_x + depth, half_width], [apex_x, 0.0], [apex_x + depth, -half_width]])
    if kind == 'claw':
        return np.array([[apex_x + depth - 0.02, half_width - claw_in], [apex_x + depth + 0.03, half_width],
                         [apex_x + 0.03, 0.06], [apex_x, 0.0], [apex_x + 0.03, -0.06],
                         [apex_x + depth + 0.03, -half_width], [apex_x + depth - 0.02, -(half_width - claw_in)]])
    if kind == 'channel':
        # funnel mouth -> narrow channel (parallel walls, 0.10 m apart for a 0.066 m ball) -> flat back wall.
        # Parallel walls push the ball sideways only, so it is steered around corners instead of being squeezed out forward.
        a, ch, fl = apex_x, 0.05, 0.17
        return np.array([[a + 0.20, fl], [a + 0.07, ch], [a, ch], [a, -ch], [a + 0.07, -ch], [a + 0.20, -fl]])
    if kind == 'flat':
        return np.array([[apex_x, half_width], [apex_x, -half_width]])
    if kind == 'vnarrow':
        # the printed narrow-V pusher (ball_pusher_V_narrow_v1.stl): two straight walls, inner faces from the tip (apex_x) to +-0.10 m at
        # 0.20 m ahead (half angle 26.6 deg).  The ball centre sits 0.075 m ahead of the tip (rider_x handles 3-point V shapes).
        return np.array([[apex_x + 0.20, 0.10], [apex_x, 0.0], [apex_x + 0.20, -0.10]])
    if kind == 'arc':
        # The printed pusher (ball_pusher_clamp_v1.stl, 240 x 145 x 40 mm): a block whose front edge is a SHALLOW CONCAVE ARC,
        # radius 160.6 mm, 34.9 mm deep over +-100 mm, then flat out to +-120 mm.  No side walls: the ball is only held by the slope
        # of the arc (a ball 33.5 mm in radius sits at the bottom, apex_x = arc bottom).
        R, half, ext = 0.1606, 0.100, 0.120
        lat = np.linspace(half, -half, 41)
        fwd = apex_x + (R - np.sqrt(R * R - lat * lat))
        top = apex_x + (R - math.sqrt(R * R - half * half))
        return np.vstack([[top, ext], np.column_stack([fwd, lat]), [top, -ext]])
    raise ValueError(kind)


def rider_x(scoop, radius):
    """Where the centre of a ball sits when it is pushed by this scoop (car frame, x forward)."""
    apex = float(scoop[:, 0].min())
    if len(scoop) == 3:                                   # V: ball touches both arms
        half, depth = abs(scoop[0, 1]), scoop[0, 0] - apex
        return apex + radius / max(math.sin(math.atan2(half, max(depth, 1e-6))), 0.3)
    return apex + radius                                  # flat / channel / claw back wall


def car_to_world(pose, pts):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.atleast_2d(pts)
    return np.column_stack([x + c * pts[:, 0] - s * pts[:, 1], y + s * pts[:, 0] + c * pts[:, 1]])


def world_to_car(pose, pts):
    x, y, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.atleast_2d(pts)
    dx, dy = pts[:, 0] - x, pts[:, 1] - y
    return np.column_stack([c * dx + s * dy, -s * dx + c * dy])


# ---------------------------------------------------------------------------------------------
# Ball
# ---------------------------------------------------------------------------------------------
@dataclass
class BallParams:
    radius: float = 0.033          # tennis-ball size
    roll_decel: float = 0.35       # constant rolling-resistance deceleration [m/s^2] (must be measured on the real floor)
    e_scoop: float = 0.35          # restitution against the scoop (only above `e_speed`)
    e_wall: float = 0.45           # restitution against walls
    e_speed: float = 0.25          # slower impacts are inelastic, so a pushed ball stays in contact
    stop_speed: float = 0.015      # below this the ball is at rest


def _closest_on_segment(p, a, b):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-12), 0.0, 1.0)
    return a + t * ab


def ball_step(p, v, dt, prm, scoop_world=None, car_vel_at=None, track=None):
    """Advance the ball by dt. scoop_world: (K,2) polyline in world coordinates; car_vel_at(point)->velocity of the
    car body at that point. Returns (p, v, touching)."""
    touching = False
    sp = math.hypot(v[0], v[1])
    if sp > prm.stop_speed:
        dec = min(prm.roll_decel * dt, sp)
        v = v * (1.0 - dec / sp)
    else:
        v = np.zeros(2)
    p = p + v * dt
    if scoop_world is not None:
        for i in range(len(scoop_world) - 1):
            c = _closest_on_segment(p, scoop_world[i], scoop_world[i + 1])
            d = p - c
            dist = math.hypot(d[0], d[1])
            if dist < prm.radius:
                n = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                vc = car_vel_at(c)
                vrel = float(np.dot(v - vc, n))
                p = p + n * (prm.radius - dist)
                if vrel < 0.0:
                    e = prm.e_scoop if -vrel > prm.e_speed else 0.0
                    v = v - (1.0 + e) * vrel * n
        for i in range(len(scoop_world) - 1):          # "touching" = within 4 mm of the scoop after resolving
            c = _closest_on_segment(p, scoop_world[i], scoop_world[i + 1])
            if math.hypot(*(p - c)) < prm.radius + 0.004:
                touching = True
    if track is not None:
        w = float(track.dist(p[0], p[1]))
        if w < prm.radius:
            gx, gy = track.grad(p[0], p[1])
            g = math.hypot(gx, gy)
            if g > 1e-6:
                n = np.array([gx, gy]) / g
                p = p + n * (prm.radius - w)
                vn = float(np.dot(v, n))
                if vn < 0.0:
                    e = prm.e_wall if -vn > prm.e_speed else 0.0
                    v = v - (1.0 + e) * vn * n
    return p, v, touching


def car_velocity_field(pose, v, yaw_rate):
    """Velocity of the car body at a world point, for a car moving with speed v and turning at yaw_rate about the rear axle."""
    x, y, yaw = pose
    vx, vy = v * math.cos(yaw), v * math.sin(yaw)

    def at(pt):
        rx, ry = pt[0] - x, pt[1] - y
        return np.array([vx - yaw_rate * ry, vy + yaw_rate * rx])
    return at


# ---------------------------------------------------------------------------------------------
# Car footprint against the walls
# ---------------------------------------------------------------------------------------------
def footprint_points(scoop, step=0.04):
    """Sample points on the body outline plus the scoop, in the car frame."""
    corners = np.array([[BODY_REAR, -HALF_WIDTH], [BODY_FRONT, -HALF_WIDTH], [BODY_FRONT, HALF_WIDTH], [BODY_REAR, HALF_WIDTH]])
    loops = [np.vstack([corners, corners[:1]]), scoop]
    pts = []
    for poly in loops:
        for a, b in zip(poly[:-1], poly[1:]):
            n = max(2, int(np.hypot(*(b - a)) / step))
            pts.append(np.linspace(a, b, n, endpoint=False))
    return np.vstack(pts)


def wall_clearance(track, pose, foot):
    """Smallest distance between the car outline and a wall (negative = inside a wall)."""
    w = track.dist(*car_to_world(pose, foot).T)
    return float(w.min())


# ---------------------------------------------------------------------------------------------
# Car
# ---------------------------------------------------------------------------------------------
@dataclass
class CarParams:
    wheelbase: float = 0.46        # effective wheelbase (steering response measured on the real car)
    max_steer: float = 0.4189
    steer_tau: float = 0.08        # servo lag [s]
    steer_rate: float = 4.0        # servo rate limit [rad/s]
    speed_tau: float = 0.25        # VESC speed-loop lag [s]
    max_accel: float = 4.0
    max_brake: float = 2.0         # the real car brakes weakly (measured median ~0.9 m/s^2)
    cmd_latency: float = 0.10      # drive command -> actuator [s]


class Car:
    def __init__(self, pose, prm=None):
        self.prm = prm or CarParams()
        self.x, self.y, self.yaw = pose
        self.v = 0.0
        self.delta = 0.0
        self.yaw_rate = 0.0
        self.queue = []            # (t_apply, v_cmd, steer_cmd)
        self.v_cmd = 0.0
        self.s_cmd = 0.0

    @property
    def pose(self):
        return (self.x, self.y, self.yaw)

    def command(self, t, v_cmd, steer_cmd):
        self.queue.append((t + self.prm.cmd_latency, v_cmd, steer_cmd))

    def step(self, t, dt):
        while self.queue and self.queue[0][0] <= t:
            _, self.v_cmd, self.s_cmd = self.queue.pop(0)
        p = self.prm
        target = float(np.clip(self.s_cmd, -p.max_steer, p.max_steer))
        d_delta = np.clip((target - self.delta) / p.steer_tau, -p.steer_rate, p.steer_rate) * dt
        self.delta += d_delta
        a = np.clip((self.v_cmd - self.v) / p.speed_tau, -p.max_brake, p.max_accel)
        v_new = self.v + a * dt
        if (self.v_cmd >= 0) != (v_new >= 0) and abs(self.v_cmd) < 1e-3:
            v_new = 0.0
        self.yaw_rate = v_new * math.tan(self.delta) / p.wheelbase
        self.yaw += self.yaw_rate * dt
        self.x += v_new * math.cos(self.yaw) * dt
        self.y += v_new * math.sin(self.yaw) * dt
        self.v = v_new


# ---------------------------------------------------------------------------------------------
# Depth camera (RealSense D435-like) looking forward
# ---------------------------------------------------------------------------------------------
@dataclass
class CameraParams:
    x: float = 0.15                # mounting position ahead of the rear axle [m] (0.30 puts the scoop inside the blind zone)
    fov: float = math.radians(43.5)   # half field of view (87 deg horizontal)
    min_range: float = 0.28        # depth blind zone
    max_range: float = 4.0
    p_detect: float = 0.97
    bearing_sigma: float = 0.008   # [rad]
    range_sigma0: float = 0.004
    range_sigma2: float = 0.004    # sigma_r = s0 + s2 * r^2
    latency: float = 0.07          # image -> detection available [s]
    rate: float = 30.0


class Camera:
    def __init__(self, prm=None, seed=0):
        self.prm = prm or CameraParams()
        self.rng = np.random.default_rng(seed)

    def sense(self, pose, ball, radius, track):
        """Returns (x, y) of the ball centre in the CAR frame as the detector would report it, or None."""
        p = self.prm
        cx, cy, yaw = pose
        cam = np.array([cx + p.x * math.cos(yaw), cy + p.x * math.sin(yaw)])
        d = ball - cam
        rng_ = math.hypot(*d)
        bearing = wrap(math.atan2(d[1], d[0]) - yaw)
        if rng_ < p.min_range or rng_ > p.max_range or abs(bearing) > p.fov:
            return None
        if self.rng.random() > p.p_detect:
            return None
        n = max(3, int(rng_ / 0.05))                       # line of sight against walls
        line = cam + np.outer(np.linspace(0, 1, n), d)
        if track is not None and track.dist(line[:, 0], line[:, 1]).min() < 0.01:
            return None
        r = rng_ + self.rng.normal(0, p.range_sigma0 + p.range_sigma2 * rng_ ** 2)
        b = bearing + self.rng.normal(0, p.bearing_sigma)
        return (p.x + r * math.cos(b), r * math.sin(b))
