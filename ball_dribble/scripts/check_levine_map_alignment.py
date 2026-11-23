#!/usr/bin/env python3
"""Compare a Levine world-coordinate waypoint CSV with the active ROS /map.

Run with Gym active but ball MPC disabled. Read-only; no drive commands.
"""
import argparse
import math
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from nav_msgs.msg import OccupancyGrid


class MapReceiver(Node):
    def __init__(self):
        super().__init__('levine_alignment_check')
        self.msg = None
        qos = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST,
                         reliability=ReliabilityPolicy.RELIABLE,
                         durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self._got, qos)

    def _got(self, msg):
        self.msg = msg


def load_map(timeout):
    rclpy.init()
    node = MapReceiver()
    start = time.monotonic()
    try:
        while node.msg is None and time.monotonic()-start < timeout:
            rclpy.spin_once(node, timeout_sec=0.3)
        if node.msg is None:
            raise RuntimeError('No /map data received. Start Gym, and check map QoS.')
        return node.msg
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[1]
    parser.add_argument('--csv', type=Path, default=here/'paths'/'levine_ball_path.csv')
    parser.add_argument('--timeout', type=float, default=12.0)
    args = parser.parse_args()
    raw = np.genfromtxt(args.csv, delimiter=',', names=True)
    if not {'x','y'}.issubset(set(raw.dtype.names or [])):
        raise ValueError('Expected CSV header with x,y columns.')
    x = np.atleast_1d(raw['x']).astype(float)
    y = np.atleast_1d(raw['y']).astype(float)
    msg = load_map(args.timeout)
    W,H = int(msg.info.width),int(msg.info.height)
    res = float(msg.info.resolution)
    data = np.asarray(msg.data, dtype=np.int16).reshape(H,W)
    occupied = (data >= 65) | (data < 0)
    # Geometry follows nav_msgs OccupancyGrid, including origin rotation.
    org = msg.info.origin
    q = org.orientation
    yaw = math.atan2(2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z))
    dx,dy = x-org.position.x,y-org.position.y
    col = ( math.cos(yaw)*dx + math.sin(yaw)*dy)/res - 0.5
    row = (-math.sin(yaw)*dx + math.cos(yaw)*dy)/res - 0.5
    inside = (col>=0)&(row>=0)&(col<W)&(row<H)
    print(f'Map: {W} x {H}, resolution {res:.4f} m/cell, origin ({org.position.x:.3f}, {org.position.y:.3f}), yaw {math.degrees(yaw):.2f} deg')
    print(f'Map occupied/unknown cells: {np.count_nonzero(occupied)}/{W*H} ({100*np.mean(occupied):.1f}%)')
    print(f'Waypoints: {len(x)}, x=[{x.min():.3f}, {x.max():.3f}], y=[{y.min():.3f}, {y.max():.3f}]')
    print(f'Waypoints inside /map grid: {int(inside.sum())}/{len(x)}')
    if not np.any(inside):
        print('WARNING: All waypoints are outside /map grid; map and path do not align.')
        return
    distance = distance_transform_edt(~occupied)*res
    rr,cc = row[inside],col[inside]
    gap = map_coordinates(distance, [rr,cc], order=1, mode='nearest')
    cells = map_coordinates(data.astype(float), [rr,cc], order=0, mode='nearest')
    print(f'Waypoints on wall or unknown cell: {int(np.count_nonzero((cells>=65)|(cells<0)))}/{len(cells)}')
    print('Nearest wall distance of waypoint centers (m):')
    print(f'  min {gap.min():.3f}, p10 {np.percentile(gap,10):.3f}, median {np.median(gap):.3f}, p90 {np.percentile(gap,90):.3f}, max {gap.max():.3f}')
    print(f'Waypoints within 2m of a map wall/unknown: {np.mean(gap<2)*100:.1f}%')
    if np.median(gap)>5.0:
        print('WARNING: Most waypoints are very far from /map walls. Check map image, YAML origin/resolution, and path world coordinates.')
    elif np.mean((cells>=65)|(cells<0))>0.1:
        print('WARNING: Many waypoints lie on occupied/unknown cells. Check /map alignment before driving.')
    else:
        print('No obvious gross alignment error from sampled waypoint-wall distances. Visual map comparison remains necessary.')


if __name__=='__main__':
    main()
