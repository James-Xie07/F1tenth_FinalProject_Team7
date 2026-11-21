#!/usr/bin/env python3
"""ROS 2 (Jazzy) node: find the ball with a RealSense D435(i) and publish its position in the CAR frame.

  subscribes  /camera/camera/color/image_raw                    colour image
              /camera/camera/aligned_depth_to_color/image_raw   depth aligned to colour (launch realsense2_camera with align_depth.enable:=true)
              /camera/camera/color/camera_info                  intrinsics
  publishes   /ball/position   geometry_msgs/PointStamped   frame "rear_axle": x forward, y left, z up, metres; stamp = IMAGE time
              ~/debug_image    sensor_msgs/Image            colour image with the detection drawn (only if someone subscribes)

Method: HSV threshold for the ball colour -> biggest roughly round blob -> median depth inside the blob -> deproject with the camera
intrinsics -> rotate by the camera tilt and add the mount offset.  The depth is measured on the ball SURFACE, so the ball radius is added
to get the centre.

Set the camera mount (cam_x, cam_y, cam_z in metres from the REAR AXLE, cam_pitch_deg downward) and the colour range for your ball.
The D435 cannot measure closer than about 0.2-0.3 m: mount it so that the ball, when it sits in the scoop, is farther away than that
(for example above/behind the LiDAR), otherwise the controller has to rely on its contact model while the ball is in the scoop.
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from message_filters import ApproximateTimeSynchronizer, Subscriber


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector_node')
        params = dict(color_topic='/camera/camera/color/image_raw', depth_topic='/camera/camera/aligned_depth_to_color/image_raw',
                      info_topic='/camera/camera/color/camera_info', output_topic='/ball/position', frame='rear_axle',
                      cam_x=0.15, cam_y=0.0, cam_z=0.25, cam_pitch_deg=20.0,
                      hsv_low=[22, 70, 70], hsv_high=[45, 255, 255],       # tennis-ball yellow-green (OpenCV hue 0..179)
                      ball_radius=0.033, min_pixels=12, min_circularity=0.6,
                      min_range=0.25, max_range=4.0, depth_scale=0.001)    # depth image in millimetres -> metres
        for k, v in params.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in params}
        self.bridge = CvBridge()
        self.K = None
        pitch = math.radians(float(self.p['cam_pitch_deg']))
        # camera body axes: forward z_c, left = -x_c, up = -y_c (optical frame: x right, y down, z forward)
        # tilt down by `pitch` about the left axis, then place at the mount position
        c, s = math.cos(pitch), math.sin(pitch)
        R_body_from_cam = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)
        R_tilt = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)         # rotation about the left axis: nose down
        self.R = R_tilt @ R_body_from_cam
        self.t = np.array([self.p['cam_x'], self.p['cam_y'], self.p['cam_z']], float)
        self.pub = self.create_publisher(PointStamped, self.p['output_topic'], 1)
        self.dbg = self.create_publisher(Image, '~/debug_image', 1)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=5, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(CameraInfo, self.p['info_topic'], self.info_callback, qos)
        color = Subscriber(self, Image, self.p['color_topic'], qos_profile=qos)
        depth = Subscriber(self, Image, self.p['depth_topic'], qos_profile=qos)
        self.sync = ApproximateTimeSynchronizer([color, depth], queue_size=5, slop=0.03)
        self.sync.registerCallback(self.images_callback)
        self.get_logger().info('ball_detector_node ready')

    def info_callback(self, msg):
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])      # fx, fy, cx, cy

    def images_callback(self, color_msg, depth_msg):
        if self.K is None:
            return
        bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32) * float(self.p['depth_scale'])
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.p['hsv_low'], np.uint8), np.array(self.p['hsv_high'], np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < float(self.p['min_pixels']):
                continue
            perim = cv2.arcLength(cnt, True)
            circ = 4 * math.pi * area / max(perim * perim, 1e-6)
            if circ < float(self.p['min_circularity']):
                continue
            if best is None or area > best[0]:
                best = (area, cnt)
        if best is not None:
            (u, v), radius_px = cv2.minEnclosingCircle(best[1])
            r_in = max(1, int(0.5 * radius_px))
            y0, y1, x0, x1 = int(v) - r_in, int(v) + r_in + 1, int(u) - r_in, int(u) + r_in + 1
            patch = depth[max(y0, 0):y1, max(x0, 0):x1]
            patch = patch[(patch > float(self.p['min_range'])) & (patch < float(self.p['max_range']))]
            if patch.size >= 3:
                z = float(np.median(patch)) + float(self.p['ball_radius'])      # surface -> centre
                fx, fy, cx, cy = self.K
                pt_cam = np.array([(u - cx) * z / fx, (v - cy) * z / fy, z])    # optical frame
                pt = self.R @ pt_cam + self.t                                    # car frame, origin rear axle
                out = PointStamped()
                out.header.stamp = color_msg.header.stamp                        # IMAGE time: the controller corrects for the delay
                out.header.frame_id = self.p['frame']
                out.point.x, out.point.y, out.point.z = float(pt[0]), float(pt[1]), float(pt[2])
                self.pub.publish(out)
                if self.dbg.get_subscription_count() > 0:
                    cv2.circle(bgr, (int(u), int(v)), int(radius_px), (0, 255, 0), 2)
                    cv2.putText(bgr, f'{pt[0]:.2f},{pt[1]:.2f}', (int(u) + 5, int(v) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    self.dbg.publish(self.bridge.cv2_to_imgmsg(bgr, 'bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = BallDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
