#!/usr/bin/env python3
"""ROS 2 (Jazzy) node: push a ball around the track with the scoop on the front of the car.

All control logic lives in ball_control.py (ROS-free, tested in the simulator in sim.py). This file only connects it to ROS:

  subscribes  /pf/pose/odom              particle-filter pose of the LiDAR (converted to the rear axle, as in the Frenet node)
              /odom                      VESC wheel odometry (speed)
              /ball/position             geometry_msgs/PointStamped, ball centre in the CAR frame (x forward, y left, metres,
                                         origin = rear axle).  Produced by ball_detector_node.py; the header stamp must be
                                         the camera image time (the tracker corrects for the camera delay with it).
              /map                       occupancy grid -> wall distance field (used to plan the approach to a ball that rolled away)
  publishes   /drive                     ackermann_msgs/AckermannDriveStamped
              ~/mode                     std_msgs/String   PUSH / APPROACH / CHASE / SEARCH
              ~/ball                     visualization_msgs/Marker   estimated ball (green = held, yellow = rolling, red = lost)

Every field of Config in ball_control.py is also a ROS parameter of the same name:
  python3 ball_push_node.py --ros-args -p waypoint_file:=/path/ball_path.csv -p ball_roll_decel:=0.35 -p max_speed:=2.0
"""
import math
import sys
import time
from dataclasses import fields
from pathlib import Path
from threading import RLock

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from ball_dribble.ball_control import BallPusher, Config   # noqa: E402
from ball_dribble.mapwalls import MapWalls                 # noqa: E402
from ball_dribble.physics import wrap                      # noqa: E402


def find_data_file(name):
    """Absolute path as given; else relative to the current directory; else the installed package's paths/ folder."""
    f = Path(name)
    if f.is_absolute() or f.exists():
        return f
    try:
        from ament_index_python.packages import get_package_share_directory
        g = Path(get_package_share_directory('ball_dribble')) / 'paths' / name
        if g.exists():
            return g
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / 'paths' / name      # running from a source checkout


class BallPushNode(Node):
    def __init__(self):
        super().__init__('ball_push_node')
        defaults = {f.name: f.default for f in fields(Config)}
        ros_defaults = dict(waypoint_file='ball_path.csv', odom_is_laser_pose=True, global_frame='map',
                            odom_topic='/pf/pose/odom', speed_topic='/odom', drive_topic='/drive',
                            ball_topic='/ball/position', map_topic='/map', ball_frame='rear_axle',
                            lidar_offset_x=0.27, lidar_offset_y=0.0, lidar_yaw=0.0,
                            control_period=0.05, odom_timeout=0.5, vesc_timeout=0.3, enable=True)
        for name, value in {**defaults, **ros_defaults}.items():
            self.declare_parameter(name, value)
        p = {name: self.get_parameter(name).value for name in {**defaults, **ros_defaults}}
        filename = find_data_file(p['waypoint_file'])
        kw = {name: type(defaults[name])(p[name]) for name in defaults}
        kw['path_file'] = str(filename)
        self.cfg = Config(**kw)
        self.pusher = BallPusher(self.cfg, control_period=float(p['control_period']))
        self.p = p
        self.enabled = bool(p['enable'])
        self.lock = RLock()
        self.pose = None            # (x, y, yaw) of the rear axle
        self.pose_stamp = None
        self.pose_wall = None
        self.yaw_rate = 0.0
        self.speed = 0.0
        self.vesc_speed, self.vesc_wall = None, None
        self.prev_yaw = None

        self.drive_pub = self.create_publisher(AckermannDriveStamped, p['drive_topic'], 1)
        self.mode_pub = self.create_publisher(String, '~/mode', 1)
        self.marker_pub = self.create_publisher(Marker, '~/ball', 1)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=20, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sensor_group = MutuallyExclusiveCallbackGroup()
        self.control_group = MutuallyExclusiveCallbackGroup()
        self.create_subscription(Odometry, p['odom_topic'], self.odom_callback, qos, callback_group=self.sensor_group)
        if p['speed_topic']:
            self.create_subscription(Odometry, p['speed_topic'], self.vesc_callback, qos, callback_group=self.sensor_group)
        self.create_subscription(PointStamped, p['ball_topic'], self.ball_callback, qos, callback_group=self.sensor_group)
        map_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, p['map_topic'], self.map_callback, map_qos, callback_group=self.sensor_group)
        self.create_timer(float(p['control_period']), self.control_callback, callback_group=self.control_group)
        self.get_logger().info(
            f'ball_push_node ready: path {self.pusher.path.length:.2f} m from {filename}, scoop={self.cfg.scoop_kind}, '
            f'ball_roll_decel={self.cfg.ball_roll_decel:.2f} m/s^2 (MEASURE IT), max_speed={self.cfg.max_speed:.1f} m/s, '
            f'push={self.cfg.push_controller}, enable={self.enabled}')

    # ------------------------------------------------------------------ time helpers
    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def stamp_to_s(stamp):
        return stamp.sec + stamp.nanosec * 1e-9

    # ------------------------------------------------------------------ sensors
    def map_callback(self, msg):
        try:
            self.pusher.walls = MapWalls(msg)
            self.get_logger().info('map received: approach planning with wall clearance is on')
        except Exception as exc:                                    # keep driving without the planner
            self.get_logger().warning(f'could not build the wall field: {exc}')

    def odom_callback(self, msg):
        frame = str(msg.header.frame_id).lstrip('/')
        if frame and frame != self.p['global_frame']:
            with self.lock:
                self.pose = None
            self.get_logger().error('odometry frame differs from global_frame', throttle_duration_sec=2.0)
            return
        q = msg.pose.pose.orientation
        n = math.sqrt(q.x ** 2 + q.y ** 2 + q.z ** 2 + q.w ** 2)
        if not math.isfinite(n) or n < 1e-8:
            return
        qx, qy, qz, qw = q.x / n, q.y / n, q.z / n, q.w / n
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self.p['odom_is_laser_pose']:                           # the PF gives the LiDAR pose; the model uses the rear axle
            yaw -= float(self.p['lidar_yaw'])
            x -= math.cos(yaw) * float(self.p['lidar_offset_x']) - math.sin(yaw) * float(self.p['lidar_offset_y'])
            y -= math.sin(yaw) * float(self.p['lidar_offset_x']) + math.cos(yaw) * float(self.p['lidar_offset_y'])
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            return
        stamp = self.stamp_to_s(msg.header.stamp)
        with self.lock:
            if self.pose_stamp is not None and stamp > self.pose_stamp + 1e-4 and self.prev_yaw is not None:
                rate = wrap(yaw - self.prev_yaw) / (stamp - self.pose_stamp)
                if abs(rate) < 8.0:
                    self.yaw_rate = 0.5 * self.yaw_rate + 0.5 * rate
            self.prev_yaw = yaw
            self.pose = (x, y, yaw)
            self.pose_stamp = stamp
            self.pose_wall = time.monotonic()
            self.speed = msg.twist.twist.linear.x

    def vesc_callback(self, msg):
        with self.lock:
            self.vesc_speed = float(msg.twist.twist.linear.x)
            self.vesc_wall = time.monotonic()

    def ball_callback(self, msg):
        if msg.header.frame_id and msg.header.frame_id != self.p['ball_frame']:
            self.get_logger().warning(f'ball point is in frame "{msg.header.frame_id}", expected "{self.p["ball_frame"]}"',
                                      throttle_duration_sec=5.0)
            return
        x, y = msg.point.x, msg.point.y
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        t = min(self.stamp_to_s(msg.header.stamp), self.now())     # never in the future
        with self.lock:
            self.pusher.observe(t, (x, y))

    # ------------------------------------------------------------------ output
    def publish_drive(self, speed, steer):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'ego_racecar/base_link'
        limit = np.float32(self.cfg.max_steer)
        if float(limit) > self.cfg.max_steer:
            limit = np.nextafter(limit, np.float32(0.0))
        msg.drive.speed = float(np.clip(speed, 0.0, self.cfg.max_speed))
        msg.drive.steering_angle = float(np.float32(np.clip(steer, -float(limit), float(limit))))
        self.drive_pub.publish(msg)

    def stop(self):
        self.publish_drive(0.0, 0.0)

    def control_callback(self):
        if not self.enabled:
            self.stop()
            return
        with self.lock:
            pose, stamp, wall = self.pose, self.pose_stamp, self.pose_wall
            speed = self.speed
            if self.vesc_speed is not None and self.vesc_wall is not None and time.monotonic() - self.vesc_wall <= float(self.p['vesc_timeout']):
                speed = self.vesc_speed
            yaw_rate = self.yaw_rate
        if pose is None or wall is None or time.monotonic() - wall > float(self.p['odom_timeout']):
            self.stop()
            self.get_logger().warning('no fresh pose: stopped', throttle_duration_sec=2.0)
            return
        try:
            with self.lock:
                v_cmd, steer = self.pusher.update(self.now(), pose, speed, yaw_rate, t_pose=min(stamp, self.now()))
        except Exception as exc:                                    # never leave the last command running
            self.stop()
            self.get_logger().error(f'controller error, stopped: {exc!r}', throttle_duration_sec=1.0)
            return
        self.publish_drive(v_cmd, steer)
        info = self.pusher.info
        self.mode_pub.publish(String(data=info['mode']))
        self.publish_marker(info)

    def publish_marker(self, info):
        m = Marker()
        m.header.frame_id = self.p['global_frame']
        m.header.stamp = self.get_clock().now().to_msg()
        m.ns, m.id, m.type, m.action = 'ball', 0, Marker.SPHERE, Marker.ADD
        b = info['ball']
        m.pose.position.x, m.pose.position.y, m.pose.position.z = float(b[0]), float(b[1]), float(self.cfg.ball_radius)
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        held, lost = info['contact'], info['mode'] == 'SEARCH'
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = (1.0, 0.0, 0.0) if lost else ((0.0, 1.0, 0.0) if held else (1.0, 0.9, 0.0))
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = BallPushNode()
    from rclpy.executors import MultiThreadedExecutor
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
