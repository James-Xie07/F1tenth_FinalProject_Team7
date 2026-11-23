import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # run from a source checkout
"""Closed-loop test of ball_push_node.py WITHOUT ROS: stub rclpy/message modules, feed the node's callbacks from the simulator's
car/ball/camera models, and read back what it publishes on /drive.  Checks the wiring (frames, stamps, odom conversion), not ROS itself."""
import math, sys, types
import numpy as np

# ---------------------------------------------------------------- stub ROS modules
class Msg:
    def __init__(self, **kw): self.__dict__.update(kw)
def ns(**kw): return types.SimpleNamespace(**kw)
class Clock:
    t = 0.0
    def now(self):
        c = Clock
        class T:
            nanoseconds = int(c.t * 1e9)
            def to_msg(self_): return ns(sec=int(c.t), nanosec=int((c.t % 1) * 1e9))
        return T()
class Log:
    def info(self, *a, **k): pass
    warning = error = info
class Node:
    def __init__(self, name): self._params, self.subs, self.pubs = {}, {}, {}
    def declare_parameter(self, n, v): self._params[n] = v
    def get_parameter(self, n): return ns(value=self._params[n])
    def create_publisher(self, t, topic, q):
        p = ns(out=[], publish=None); p.publish = p.out.append; self.pubs[topic] = p; return p
    def create_subscription(self, t, topic, cb, q, callback_group=None): self.subs[topic] = cb
    def create_timer(self, *a, **k): pass
    def get_clock(self): return Clock()
    def get_logger(self): return Log()
def mod(name, **attrs):
    m = types.ModuleType(name); m.__dict__.update(attrs); sys.modules[name] = m; return m
class Enum:  # QoS enums
    KEEP_LAST = RELIABLE = BEST_EFFORT = TRANSIENT_LOCAL = 0
class Marker:
    SPHERE, ADD = 2, 0
    def __init__(self):
        self.header = ns(frame_id='', stamp=None); self.pose = ns(position=ns(x=0, y=0, z=0), orientation=ns(w=1, x=0, y=0, z=0))
        self.scale = ns(x=0, y=0, z=0); self.color = ns(a=0, r=0, g=0, b=0)
class Drive:
    def __init__(self): self.header = ns(stamp=None, frame_id=''); self.drive = ns(speed=0, steering_angle=0, acceleration=0)
class _T:
    def __init__(self, nanoseconds=0): self.ns = nanoseconds
    def to_msg(self): return ns(sec=int(self.ns // 10**9), nanosec=int(self.ns % 10**9))
mod('rclpy', init=lambda args=None: None, try_shutdown=lambda: None)
mod('rclpy.time', Time=_T); sys.modules['rclpy'].time = sys.modules['rclpy.time']
mod('rclpy.node', Node=Node); mod('rclpy.callback_groups', MutuallyExclusiveCallbackGroup=lambda: None)
mod('rclpy.qos', QoSProfile=lambda **k: None, HistoryPolicy=Enum, ReliabilityPolicy=Enum, DurabilityPolicy=Enum)
mod('rclpy.executors', MultiThreadedExecutor=object)
mod('ackermann_msgs'); mod('ackermann_msgs.msg', AckermannDriveStamped=Drive)
class PointStamped:
    def __init__(self): self.header = ns(stamp=None, frame_id=''); self.point = ns(x=0, y=0, z=0)
mod('geometry_msgs'); mod('geometry_msgs.msg', PointStamped=PointStamped)
mod('nav_msgs'); mod('nav_msgs.msg', Odometry=object, OccupancyGrid=object)
mod('std_msgs'); mod('std_msgs.msg', String=lambda data='': ns(data=data), Float32=lambda data=0.: ns(data=data))
mod('visualization_msgs'); mod('visualization_msgs.msg', Marker=Marker)

import ball_dribble.ball_push_node as bpn
from ball_dribble.physics import Car, CarParams, Camera, BallParams, ball_step, car_to_world, car_velocity_field, make_scoop, wrap
from ball_dribble import sim

def odom(t, x, y, yaw, v, frame='map'):
    q = ns(x=0.0, y=0.0, z=math.sin(yaw / 2), w=math.cos(yaw / 2))
    return ns(header=ns(frame_id=frame, stamp=ns(sec=int(t), nanosec=int((t % 1) * 1e9))),
              pose=ns(pose=ns(position=ns(x=x, y=y), orientation=q)), twist=ns(twist=ns(linear=ns(x=v), angular=ns(z=0.0))))

def main(T=14.0, dt=0.002):
    node = bpn.BallPushNode()
    node.pusher.walls = sim.get_track()
    path = node.pusher.path
    xy0, yaw0, _, _ = path.at(4.0)
    car = Car((xy0[0], xy0[1], yaw0), CarParams())
    ball_p, ball_v = np.array(path.at(4.9)[0], float), np.zeros(2)
    scoop = make_scoop('channel'); prm = BallParams(); cam = Camera(seed=1); track = sim.get_track()
    pose_hist = []; det = []; next_odom = next_cam = next_ctl = 0.0
    drive_out = node.pubs['/drive'].out
    modes = {}; touch = 0.0
    for k in range(int(T / dt)):
        t = k * dt; Clock.t = t
        if t >= next_odom:                                           # PF pose of the LiDAR, 25 ms old, with noise
            next_odom += 0.02
            i = max(0, len(pose_hist) - 13); _, x, y, yaw, v = pose_hist[i] if pose_hist else (0, *car.pose, 0.0)
            lx, ly = x + 0.27 * math.cos(yaw), y + 0.27 * math.sin(yaw)
            node.subs['/pf/pose/odom'](odom(max(0, t - 0.026), lx, ly, yaw, v))
            node.subs['/odom'](odom(t, 0, 0, 0, car.v))
        if t >= next_cam:
            next_cam += 1 / 30
            z = cam.sense(car.pose, ball_p, prm.radius, track)
            if z is not None: det.append((t + 0.07, t, z))
        while det and det[0][0] <= t:
            _, ti, z = det.pop(0)
            node.subs['/ball/position'](ns(header=ns(frame_id='rear_axle', stamp=ns(sec=int(ti), nanosec=int((ti % 1) * 1e9))), point=ns(x=z[0], y=z[1], z=0.03)))
        if t >= next_ctl:
            next_ctl += 0.05
            n0 = len(drive_out); node.control_callback()
            if len(drive_out) > n0:
                d = drive_out[-1].drive; car.command(t, d.speed, d.steering_angle)
            modes[node.pusher.mode] = modes.get(node.pusher.mode, 0) + 1
        car.step(t, dt); pose_hist.append((t, *car.pose, car.v)); pose_hist[:] = pose_hist[-300:]
        ball_p, ball_v, tch = ball_step(ball_p, ball_v, dt, prm, car_to_world(car.pose, scoop), car_velocity_field(car.pose, car.v, car.yaw_rate), track)
        touch += tch * dt
    s_b = path.project(ball_p)[0]
    print('after %.0f s: modes %s | ball touching %.0f%% | ball at path position s=%.1f m (started 4.9) | car speed %.2f | last drive speed %.2f steer %+.2f'
          % (T, modes, 100 * touch / T, s_b, car.v, drive_out[-1].drive.speed, drive_out[-1].drive.steering_angle))
    ok = modes.get('PUSH', 0) > 0.6 * sum(modes.values()) and (s_b - 4.9) % path.length > 12
    print('WIRING TEST', 'PASSED' if ok else 'FAILED')
    return ok

if __name__ == '__main__':
    sys.exit(0 if main() else 1)
