#!/usr/bin/env python3
"""Real-car Levine ball pushing with Frenet sampling + constrained MPC.

Uses the exact ROS sensor topics from the existing stage-one package:
  /pf/pose/odom  (PF /map LiDAR pose), /odom (VESC speed), /scan, /map,
  /ball/position (rear_axle coordinates), /ball/camera_depth (raw optical depth)
  -> /drive, ~/mode, ~/reference_path, ~/predicted_path

Stanley ball_push_node is retained but NOT launched by the Frenet launch file.
Robot motion starts disabled unless enable_motion:=true is passed at launch.
"""
import math
import time
from pathlib import Path
from threading import RLock

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, OccupancyGrid, Path as PathMsg
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String, Bool

from .frenet_mpc import MPCConfig, LevinePath, FrenetPlanner, TrackingMPC, local_reference, bicycle_step
from .ball_recovery import (RecoverySettings, RecoverySupervisor,
                            car_point_to_world, approach_curve, direct_forward_curve,
                            rear_map_clear, rear_scan_clear)
from .mapwalls import MapWalls


def path_file(name):
    f = Path(name).expanduser()
    if f.exists():
        return f
    try:
        from ament_index_python.packages import get_package_share_directory
        f = Path(get_package_share_directory('ball_dribble')) / 'paths' / name
        if f.exists():
            return f
    except Exception:
        pass
    return Path(__file__).resolve().parents[1] / 'paths' / name


class BallFrenetMPCNode(Node):
    def __init__(self):
        super().__init__('ball_frenet_mpc_node')
        defaults = {
            'waypoint_file': 'levine_ball_path.csv', 'reference_is_ball_path': True,
            'solver_warm_start': True, 'control_period': 0.10,
            'enable_motion': False, 'reset_recovery': False,
            'odom_topic':'/pf/pose/odom', 'speed_topic':'/odom',
            'ball_topic':'/ball/position', 'depth_topic':'/ball/camera_depth',
            'scan_topic':'/scan', 'map_topic':'/map', 'drive_topic':'/drive',
            'global_frame':'map', 'ball_frame':'rear_axle',
            'odom_is_laser_pose':True, 'odom_offset_x':0.0, 'lidar_offset_x':0.27,
            'lidar_offset_y':0.0, 'lidar_yaw':0.0,
            'odom_timeout':0.50, 'speed_timeout':0.30,
            'wheelbase':0.46, 'max_speed':0.5,
            'ball_center_ahead':0.540, 'max_steer':0.4189,
            'latency_compensation':0.25,
            'contact_depth':0.29, 'depth_tolerance':0.035,
            'lateral_release':0.10, 'sensor_timeout':0.40,
            'reverse_speed':0.12, 'max_reverse':0.90,
            'min_reverse':0.20, 'require_rear_lidar':True,
            'allow_direct_approach':False,  # turn on only in sim initially
            'contact_confirm_time':0.0, 'contact_loss_grace':0.0,
            'retry_depth_hold':False,
            'start_in_hold':False,
        }
        for k,v in defaults.items():
            self.declare_parameter(k,v)
        self.p = {k:self.get_parameter(k).value for k in defaults}
        self.cfg = MPCConfig(wheelbase=float(self.p['wheelbase']),
                             max_speed=float(self.p['max_speed']),
                             solver_warm_start=bool(self.p['solver_warm_start']),
                             ball_center_ahead=float(self.p['ball_center_ahead']),
                             max_steer=float(self.p['max_steer']))
        self.rcfg = RecoverySettings(contact_depth=float(self.p['contact_depth']),
                                     depth_tolerance=float(self.p['depth_tolerance']),
                                     lateral_release=float(self.p['lateral_release']),
                                     sensor_timeout=float(self.p['sensor_timeout']),
                                     reverse_speed=float(self.p['reverse_speed']),
                                     max_reverse=float(self.p['max_reverse']),
                                     min_reverse=float(self.p['min_reverse']),
                                     allow_direct_approach=bool(self.p['allow_direct_approach']),
                                     contact_confirm_time=float(self.p['contact_confirm_time']),
                                     contact_loss_grace=float(self.p['contact_loss_grace']),
                                     retry_depth_hold=bool(self.p['retry_depth_hold']))
        self.path = LevinePath(path_file(self.p['waypoint_file']), self.cfg,
                               reference_is_ball_path=bool(self.p['reference_is_ball_path']))
        self.planner = FrenetPlanner(self.path,self.cfg)
        self.mpc = TrackingMPC(self.cfg)
        self.recovery = RecoverySupervisor(self.rcfg)
        if self.p['start_in_hold']:
            self.recovery._enter('HOLD',self.get_clock().now().nanoseconds*1e-9,'started in hold')
        self.lock=RLock()
        self.pose=None
        self.pose_stamp=None
        self.pose_arrival=None
        self.speed=0.0
        self.speed_vesc=None
        self.speed_arrival=None
        self.ball_xy=None
        self.ball_world=None
        self.ball_arrival=None
        self.depth=None
        self.depth_arrival=None
        self.scan=None
        self.scan_arrival=None
        self.walls=None
        self.steer_prev=0.0
        self.was_enabled=False
        self.last_mode=None
        self.last_reset=False
        qos=QoSProfile(depth=15,history=HistoryPolicy.KEEP_LAST,
                       reliability=ReliabilityPolicy.BEST_EFFORT)
        self.sensor_group=MutuallyExclusiveCallbackGroup()
        self.control_group=MutuallyExclusiveCallbackGroup()
        self.create_subscription(Odometry,self.p['odom_topic'],self.pose_cb,qos,callback_group=self.sensor_group)
        if self.p['speed_topic']:
            self.create_subscription(Odometry,self.p['speed_topic'],self.speed_cb,qos,callback_group=self.sensor_group)
        self.create_subscription(PointStamped,self.p['ball_topic'],self.ball_cb,qos,callback_group=self.sensor_group)
        self.create_subscription(Float32,self.p['depth_topic'],self.depth_cb,qos,callback_group=self.sensor_group)
        self.create_subscription(LaserScan,self.p['scan_topic'],self.scan_cb,qos,callback_group=self.sensor_group)
        map_qos=QoSProfile(depth=1,history=HistoryPolicy.KEEP_LAST,
                            reliability=ReliabilityPolicy.RELIABLE,
                            durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid,self.p['map_topic'],self.map_cb,map_qos,callback_group=self.sensor_group)
        self.drive_pub=self.create_publisher(AckermannDriveStamped,self.p['drive_topic'],1)
        self.mode_pub=self.create_publisher(String,'~/mode',1)
        self.recovery_info_pub=self.create_publisher(String,'~/recovery_info',1)
        self.ref_pub=self.create_publisher(PathMsg,'~/reference_path',1)
        self.pred_pub=self.create_publisher(PathMsg,'~/predicted_path',1)
        self.create_timer(float(self.p['control_period']),self.tick,callback_group=self.control_group)
        self.get_logger().info(f'Levine Frenet MPC ready, {len(self.path.xy)} waypoints, '
                               f'maximum speed {self.cfg.max_speed:.2f} m/s, '
                               f'reference_is_ball_path={self.p["reference_is_ball_path"]}, '
                               f'motion ENABLED={self.p["enable_motion"]}')

    @staticmethod
    def mono():
        return time.monotonic()

    def now(self):
        return self.get_clock().now().nanoseconds*1e-9

    def pose_cb(self,msg):
        if str(msg.header.frame_id).lstrip('/') not in ('',self.p['global_frame']):
            self.get_logger().error('PF odometry frame not map; controller stays stopped',throttle_duration_sec=2.0)
            with self.lock: self.pose=None
            return
        q=msg.pose.pose.orientation
        norm=math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
        if norm < 1e-8: return
        qw,qx,qy,qz=q.w/norm,q.x/norm,q.y/norm,q.z/norm
        yaw=math.atan2(2*(qw*qz+qx*qy),1-2*(qy*qy+qz*qz))
        x,y=float(msg.pose.pose.position.x),float(msg.pose.pose.position.y)
        if self.p['odom_is_laser_pose']:
            yaw-=float(self.p['lidar_yaw'])
            ox,oy=float(self.p['lidar_offset_x']),float(self.p['lidar_offset_y'])
            x-=math.cos(yaw)*ox-math.sin(yaw)*oy
            y-=math.sin(yaw)*ox+math.cos(yaw)*oy
        else:
            # Some gym bridges publish the vehicle centre, not the rear axle.
            # A separate parameter keeps this correction out of the real-car PF path.
            offset=float(self.p['odom_offset_x'])
            x-=math.cos(yaw)*offset
            y-=math.sin(yaw)*offset
        if not np.all(np.isfinite([x,y,yaw])): return
        with self.lock:
            self.pose=(x,y,yaw)
            self.speed=float(msg.twist.twist.linear.x)
            self.pose_stamp=msg.header.stamp.sec+1e-9*msg.header.stamp.nanosec
            self.pose_arrival=self.mono()

    def speed_cb(self,msg):
        v=float(msg.twist.twist.linear.x)
        if math.isfinite(v):
            with self.lock: self.speed_vesc,self.speed_arrival=v,self.mono()

    def ball_cb(self,msg):
        if msg.header.frame_id != self.p['ball_frame']:
            self.get_logger().warning('Ball position is not in rear_axle frame',throttle_duration_sec=2.0)
            return
        xy=np.array([msg.point.x,msg.point.y],float)
        if not np.isfinite(xy).all(): return
        with self.lock:
            self.ball_xy=xy
            if self.pose is not None:
                self.ball_world=car_point_to_world(self.pose,xy)
            self.ball_arrival=self.mono()

    def depth_cb(self,msg):
        z=float(msg.data)
        if 0.18 <= z <= 4.0 and math.isfinite(z):
            with self.lock:self.depth,self.depth_arrival=z,self.mono()

    def scan_cb(self,msg):
        angles=msg.angle_min+np.arange(len(msg.ranges))*msg.angle_increment
        ranges=np.asarray(msg.ranges,float)
        # Reject invalid data instead of treating sensor dropouts as free space.
        with self.lock: self.scan=(angles,ranges);self.scan_arrival=self.mono()

    def map_cb(self,msg):
        if str(msg.header.frame_id).lstrip('/') not in ('', self.p['global_frame']):
            with self.lock: self.walls=None
            self.get_logger().error('Occupancy map is not in the global frame: stopped')
            return
        try:
            walls=MapWalls(msg)
            with self.lock: self.walls=walls
        except Exception as exc:
            self.get_logger().error(f'Cannot load occupancy map: {exc}')

    def laser_obstacles(self,pose,scan,walls,ball_world):
        """Project front/side LiDAR returns to /map and remove known map walls + ball."""
        if scan is None or walls is None:
            return np.empty((0,2))
        angles,ranges=scan
        good=np.isfinite(ranges)&(ranges>0.16)&(ranges<9.0)
        if not np.any(good):
            return np.empty((0,2))
        ray=angles[good]+pose[2]+float(self.p['lidar_yaw'])
        r=ranges[good]
        theta=pose[2]
        origin=np.asarray(pose[:2])+np.array([
            float(self.p['lidar_offset_x'])*math.cos(theta)-float(self.p['lidar_offset_y'])*math.sin(theta),
            float(self.p['lidar_offset_x'])*math.sin(theta)+float(self.p['lidar_offset_y'])*math.cos(theta)])
        xy=origin + r[:,None]*np.column_stack((np.cos(ray),np.sin(ray)))
        # Wall hits are handled by the /map body checks, not as dynamic obstacles.
        xy=xy[walls.dist(xy[:,0],xy[:,1])>0.22]
        if ball_world is not None and len(xy):
            xy=xy[np.linalg.norm(xy-ball_world,axis=1)>0.18]
        return xy[::2]  # speed up the KD-tree without losing many 270-degree returns

    def publish_drive(self,velocity=0.0,steer=0.0):
        msg=AckermannDriveStamped()
        msg.header.stamp=self.get_clock().now().to_msg()
        msg.header.frame_id='rear_axle'
        msg.drive.speed=float(np.clip(velocity,-self.cfg.max_speed,self.cfg.max_speed))
        msg.drive.steering_angle=float(np.clip(steer,-self.cfg.max_steer,self.cfg.max_steer))
        self.drive_pub.publish(msg)

    def stop(self):
        self.publish_drive(0.0,0.0)
        self.steer_prev=0.0
        self.mpc.reset()

    def publish_path(self,pub,coords):
        msg=PathMsg()
        msg.header.frame_id=self.p['global_frame'];msg.header.stamp=self.get_clock().now().to_msg()
        for px,py in coords:
            pose=PoseStamped()
            pose.header=msg.header
            pose.pose.position.x=float(px);pose.pose.position.y=float(py)
            pose.pose.orientation.w=1.0
            msg.poses.append(pose)
        pub.publish(msg)

    def tick(self):
        # Read dynamic enable flag to allow a deliberate operator reset/enable.
        enabled=bool(self.get_parameter('enable_motion').value)
        reset=bool(self.get_parameter('reset_recovery').value)
        if reset and not self.last_reset:
            self.recovery.reset(self.now())
        self.last_reset=reset
        with self.lock:
            pose=self.pose
            pose_age=float('inf') if self.pose_arrival is None else self.mono()-self.pose_arrival
            v=self.speed
            if self.speed_arrival is not None and self.mono()-self.speed_arrival < float(self.p['speed_timeout']):
                v=self.speed_vesc
            ball=self.ball_xy.copy() if self.ball_xy is not None else None
            world=self.ball_world.copy() if self.ball_world is not None else None
            ball_age=float('inf') if self.ball_arrival is None else self.mono()-self.ball_arrival
            depth=self.depth
            depth_age=float('inf') if self.depth_arrival is None else self.mono()-self.depth_arrival
            scan=self.scan
            scan_age=float('inf') if self.scan_arrival is None else self.mono()-self.scan_arrival
            walls=self.walls
        if (not enabled or pose is None or pose_age>float(self.p['odom_timeout'])
                or not math.isfinite(v) or abs(v)>2.0):
            self.stop()
            self.was_enabled=False
            self.mode_pub.publish(String(data='DISABLED' if not enabled else 'NO_ODOMETRY'))
            return
        if not self.was_enabled:
            self.recovery.reset(self.now())
        self.was_enabled=True
        fresh=(ball is not None and world is not None and
               ball_age<self.rcfg.sensor_timeout and depth_age<self.rcfg.sensor_timeout
               and abs(ball_age-depth_age)<0.20)
        if not fresh:
            ball=None
            depth=None
        rear_safe=(walls is not None and rear_map_clear(pose,walls,self.cfg,self.rcfg)
                   and scan_age<self.rcfg.rear_scan_timeout and
                   rear_scan_clear(scan,self.rcfg,float(self.p['lidar_yaw'])))
        # Automatically disabled rear motion if no verified rear LiDAR coverage.
        if not bool(self.p['require_rear_lidar']):
            # This option is only appropriate with an independently checked rear sensor/spotter.
            rear_safe=(walls is not None and rear_map_clear(pose,walls,self.cfg,self.rcfg))
        obstacles=self.laser_obstacles(pose,scan,walls,world) if scan_age<self.rcfg.rear_scan_timeout else np.empty((0,2))
        plan=(approach_curve(pose,world,self.path,walls,self.cfg,self.rcfg.safe_clearance,obstacles)
              if fresh and scan_age<self.rcfg.rear_scan_timeout else None)
        # Centred ball that rolled ahead during braking can be recovered directly.
        # Require the synthetic/real depth to be farther than the contact depth,
        # and check the forward body path against the map and LiDAR.
        direct_plan = None
        if (bool(self.p['allow_direct_approach']) and fresh and
                scan_age<self.rcfg.rear_scan_timeout and
                depth is not None and depth>self.rcfg.contact_depth+self.rcfg.depth_tolerance):
            direct_plan = direct_forward_curve(pose,ball,walls,self.cfg,self.rcfg,obstacles)
        mode=self.recovery.step(self.now(),pose,v,
                                float(ball[1]) if ball is not None else None,
                                depth,fresh,rear_safe,plan,direct_plan=direct_plan)
        if mode != self.last_mode:
            self.get_logger().info(
                f'{self.last_mode} -> {mode}: {self.recovery.reason}; '
                f'ball_xy={tuple(np.round(ball, 3)) if ball is not None else None}, '
                f'depth={round(depth, 3) if depth is not None else None}')
            self.last_mode=mode
            self.mpc.reset()
        self.mode_pub.publish(String(data=mode))
        # Small, human-readable diagnostic topic for Foxglove during recovery.
        self.recovery_info_pub.publish(String(data=(
            f'mode={mode}, reason={self.recovery.reason}, '
            f'ball_xy={tuple(np.round(ball, 3)) if ball is not None else None}, '
            f'camera_depth={round(depth, 3) if depth is not None else None}, '
            f'direct_plan={direct_plan is not None}, '
            f'arc_plan={plan is not None}, speed={v:.3f}')))
        if mode == 'REVERSE':
            self.publish_drive(-self.rcfg.reverse_speed,0.0)
            return
        if mode not in ('PUSH','APPROACH','DIRECT_APPROACH'):
            self.stop()
            return
        # Require fresh map before planning a forward motion or recovery.
        if walls is None or scan_age>=self.rcfg.rear_scan_timeout:
            self.stop()
            self.get_logger().warning('No fresh /map or /scan: stopped',throttle_duration_sec=2.0)
            return
        state=np.array([*pose,float(np.clip(v,0,self.cfg.max_speed))])
        state=state[[0,1,3,2]]  # [x,y,v,yaw]
        delay=float(self.p['latency_compensation'])
        if delay>0 and state[2]>=0:
            state=bicycle_step(state,(0.0,self.steer_prev),min(delay,0.4),self.cfg.wheelbase)
        try:
            if mode=='PUSH':
                result=self.planner.plan(state,walls,speed=self.cfg.max_speed,obstacle_points=obstacles)
                if result is None:
                    self.stop();self.get_logger().warning(
                        f'No safe Frenet path: {self.planner.last_diagnostics}',
                        throttle_duration_sec=2.0)
                    return
                ref,poly,_=result
                velocity,steer,pred=self.mpc.solve(state,ref,self.steer_prev,
                                                   ball_lateral=float(ball[1]),use_ball=True)
            else:
                active_plan = direct_plan if mode == 'DIRECT_APPROACH' else plan
                if active_plan is None:
                    # During brief plan interruptions STOP, without changing state.
                    self.stop();return
                distance=float(np.linalg.norm(active_plan[-1]-np.asarray(pose[:2])))
                speed=self.rcfg.touch_speed if distance<0.3 else self.rcfg.approach_speed
                ref=local_reference(active_plan,state,self.cfg,speed)
                if ref is None:
                    self.stop();return
                poly=active_plan
                velocity,steer,pred=self.mpc.solve(state,ref,self.steer_prev,use_ball=False)
                velocity=min(velocity,speed)
            self.steer_prev=steer
            self.publish_drive(velocity,steer)
            self.publish_path(self.ref_pub,poly)
            self.publish_path(self.pred_pub,pred[:2].T)
        except Exception as exc:
            self.stop()
            self.get_logger().error(f'Frenet MPC error; stopped: {exc!r}',throttle_duration_sec=1.0)


def main(args=None):
    rclpy.init(args=args)
    node=BallFrenetMPCNode()
    ex=MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # During Ctrl+C the ROS context may already be invalid. Do not throw
        # a second exception while attempting the final zero-speed command.
        if rclpy.ok():
            try:
                node.stop()
            except Exception as exc:
                node.get_logger().warning(f'Final stop publish unavailable: {exc}')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
