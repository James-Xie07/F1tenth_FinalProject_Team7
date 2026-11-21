#!/usr/bin/env python3
"""ROS 2 node: a simulated ball (and D435-like camera) for f1tenth_gym_ros, which has no ball of its own.

  subscribes  /ego_racecar/odom        car pose from the gym (see odom_offset_x below)
              /map                     walls the ball bounces off
              /clicked_point           rviz "Publish Point": put the ball there (at rest)
  publishes   /ball/position           geometry_msgs/PointStamped, ball centre in the car frame (frame "rear_axle"),
                                       what ball_detector_node would publish (blind zone, field of view, noise, delay, walls block sight)
              ~/truth                  PointStamped, true ball position in the map frame
              ~/marker                 Marker, the ball in rviz (green = touching the scoop)

The ball is a rolling disc with constant deceleration; the car carries the same channel scoop as the controller
assumes (ball_dribble/physics.py). The gym itself does not know about the ball or the scoop: only the ball sees them.

Parameters: ball_roll_decel (truth, make it different from the controller's value to test robustness), scoop_kind,
odom_offset_x (how far ahead of the rear axle the gym odom point is; gym_ros reports roughly the centre of mass, ~0.17 m),
start_ahead (spawn the ball this far ahead of the rear axle on the first odom), camera_* values.
"""
import math
from collections import deque

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import Odometry, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker

from ball_dribble.mapwalls import MapWalls
from ball_dribble.physics import (BallParams, Camera, CameraParams, ball_step, car_to_world, car_velocity_field,
                                  make_scoop, rider_x, wrap)


class BallSimNode(Node):
    def __init__(self):
        super().__init__('ball_sim_node')
        params = dict(odom_topic='/ego_racecar/odom', map_topic='/map', ball_topic='/ball/position', global_frame='map',
                      odom_offset_x=0.17, scoop_kind='channel', ball_roll_decel=0.35, ball_radius=0.033,
                      start_ahead=0.0, physics_rate=200.0, camera_rate=30.0, camera_x=0.15, camera_min_range=0.28,
                      camera_latency=0.07, camera_p_detect=0.97, seed=0)
        for k, v in params.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in params}
        self.prm = BallParams(radius=float(self.p['ball_radius']), roll_decel=float(self.p['ball_roll_decel']))
        self.scoop = make_scoop(str(self.p['scoop_kind']))
        self.camera = Camera(CameraParams(x=float(self.p['camera_x']), min_range=float(self.p['camera_min_range']),
                                          latency=float(self.p['camera_latency']), p_detect=float(self.p['camera_p_detect']),
                                          rate=float(self.p['camera_rate'])), seed=int(self.p['seed']))
        self.walls = None
        self.odom = None          # (t_rx, x, y, yaw, v, yaw_rate) with (x, y) already at the rear axle
        self.ball = None          # world position
        self.vel = np.zeros(2)
        self.touch = False
        self.pending = deque()    # (release_time, capture_time, (x, y))
        self.last_step = None
        self.pub = self.create_publisher(PointStamped, self.p['ball_topic'], 1)
        self.truth_pub = self.create_publisher(PointStamped, '~/truth', 1)
        self.marker_pub = self.create_publisher(Marker, '~/marker', 1)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Odometry, self.p['odom_topic'], self.odom_callback, qos)
        map_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, self.p['map_topic'], self.map_callback, map_qos)
        self.create_subscription(PointStamped, '/clicked_point', self.clicked_callback, 10)
        self.create_timer(1.0 / float(self.p['physics_rate']), self.physics_callback)
        self.create_timer(1.0 / float(self.p['camera_rate']), self.camera_callback)
        self.get_logger().info('ball_sim_node ready: waiting for the map and the first odom')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def map_callback(self, msg):
        self.walls = MapWalls(msg)
        self.get_logger().info('map received: ball walls are on')

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        off = float(self.p['odom_offset_x'])
        x = msg.pose.pose.position.x - off * math.cos(yaw)
        y = msg.pose.pose.position.y - off * math.sin(yaw)
        self.odom = (self.now(), x, y, yaw, float(msg.twist.twist.linear.x), float(msg.twist.twist.angular.z))
        if self.ball is None:                                   # first odom: put the ball in front of the scoop
            xb = rider_x(self.scoop, self.prm.radius) + float(self.p['start_ahead'])
            self.ball = np.array([x + xb * math.cos(yaw), y + xb * math.sin(yaw)])
            self.vel = np.zeros(2)

    def clicked_callback(self, msg):
        self.ball = np.array([msg.point.x, msg.point.y], float)
        self.vel = np.zeros(2)
        self.pending.clear()
        self.get_logger().info(f'ball placed at ({msg.point.x:.2f}, {msg.point.y:.2f})')

    def car_pose_now(self):
        """Last odom pose advanced by dead reckoning to the current time (odom arrives slower than the physics)."""
        t0, x, y, yaw, v, w = self.odom
        dt = min(max(self.now() - t0, 0.0), 0.1)
        return (x + v * math.cos(yaw) * dt, y + v * math.sin(yaw) * dt, yaw + w * dt), v, w

    def physics_callback(self):
        if self.odom is None or self.ball is None:
            return
        t = self.now()
        dt = 1.0 / float(self.p['physics_rate']) if self.last_step is None else min(max(t - self.last_step, 1e-4), 0.05)
        self.last_step = t
        pose, v, w = self.car_pose_now()
        self.ball, self.vel, touching = ball_step(self.ball, self.vel, dt, self.prm, car_to_world(pose, self.scoop),
                                                  car_velocity_field(pose, v, w), self.walls)
        self.touch = bool(touching)

    def camera_callback(self):
        if self.odom is None or self.ball is None:
            return
        t = self.now()
        pose, _, _ = self.car_pose_now()
        z = self.camera.sense(pose, self.ball, self.prm.radius, self.walls)
        if z is not None:
            self.pending.append((t + self.camera.prm.latency, t, z))
        while self.pending and self.pending[0][0] <= t:
            _, tc, z = self.pending.popleft()
            msg = PointStamped()
            msg.header.stamp = rclpy.time.Time(nanoseconds=int(tc * 1e9)).to_msg()
            msg.header.frame_id = 'rear_axle'
            msg.point.x, msg.point.y, msg.point.z = float(z[0]), float(z[1]), float(self.prm.radius)
            self.pub.publish(msg)
        tr = PointStamped()
        tr.header.stamp = self.get_clock().now().to_msg()
        tr.header.frame_id = self.p['global_frame']
        tr.point.x, tr.point.y, tr.point.z = float(self.ball[0]), float(self.ball[1]), float(self.prm.radius)
        self.truth_pub.publish(tr)
        m = Marker()
        m.header.frame_id, m.header.stamp = self.p['global_frame'], tr.header.stamp
        m.ns, m.id, m.type, m.action = 'ball_truth', 0, Marker.SPHERE, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = tr.point.x, tr.point.y, tr.point.z
        m.pose.orientation.w = 1.0
        m.scale.x = m.scale.y = m.scale.z = 0.2
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = (0.0, 1.0, 0.0) if self.touch else (1.0, 0.5, 0.0)
        self.marker_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = BallSimNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
