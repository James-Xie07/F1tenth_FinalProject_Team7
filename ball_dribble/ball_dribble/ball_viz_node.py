#!/usr/bin/env python3
"""ROS 2 node, visualisation only (Foxglove / rviz): the real ball is only 6.6 cm, far too small to see on a 45 m map.

  subscribes  /ball_sim_node/marker   the simulated ball (colour: green = touching the scoop, orange = rolling)
  publishes   ~/ball                  Marker, the same ball drawn `ball_diameter` wide, same colour
              ~/path                  Marker LINE_STRIP, the ball path (waypoint_file), re-sent every 2 s for late subscribers
"""
import os
from pathlib import Path

import numpy as np
import rclpy
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from visualization_msgs.msg import Marker


def find_path_file(name):
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
    return Path(__file__).resolve().parents[1] / 'paths' / name


class BallViz(Node):
    def __init__(self):
        super().__init__('ball_viz_node')
        self.declare_parameter('waypoint_file', 'ball_path_levine2nd.csv')
        self.declare_parameter('ball_diameter', 0.22)
        self.declare_parameter('ball_topic', '/ball_sim_node/marker')
        self.declare_parameter('global_frame', 'map')
        self.d = float(self.get_parameter('ball_diameter').value)
        self.frame = str(self.get_parameter('global_frame').value)
        latched = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.ball_pub = self.create_publisher(Marker, '~/ball', 1)
        self.path_pub = self.create_publisher(Marker, '~/path', latched)
        self.create_subscription(Marker, str(self.get_parameter('ball_topic').value), self.ball_callback, 1)
        xy = np.genfromtxt(str(find_path_file(str(self.get_parameter('waypoint_file').value))), delimiter=',', skip_header=1)[:, :2]
        m = Marker()
        m.header.frame_id = self.frame
        m.ns, m.id, m.type, m.action = 'ball_path', 1, Marker.LINE_STRIP, Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.05
        m.color.r, m.color.g, m.color.b, m.color.a = 0.0, 0.8, 1.0, 1.0
        pts = np.vstack([xy[::2], xy[:1]])                                      # closed loop
        m.points = [Point(x=float(x), y=float(y), z=0.02) for x, y in pts]
        self.path_marker = m
        self.create_timer(2.0, self.send_path)
        self.send_path()
        self.get_logger().info(f'ball_viz_node ready: ball drawn {self.d:.2f} m wide, path with {len(m.points)} points')

    def send_path(self):
        self.path_marker.header.stamp = self.get_clock().now().to_msg()
        self.path_pub.publish(self.path_marker)

    def ball_callback(self, msg):
        m = Marker()
        m.header = msg.header
        m.ns, m.id, m.type, m.action = 'ball_big', 0, Marker.SPHERE, Marker.ADD
        m.pose = msg.pose
        m.pose.position.z = self.d / 2.0
        m.scale.x = m.scale.y = m.scale.z = self.d
        m.color = msg.color
        self.ball_pub.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = BallViz()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
