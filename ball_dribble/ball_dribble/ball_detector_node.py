#!/usr/bin/env python3
"""ROS 2 (Jazzy) node: find the tennis ball with a RealSense D435i and publish its centre in the CAR frame.

The colour / blob / depth logic is taken from the team's standalone script (ball_detection_speed_v2.py), which is
already tuned on the real ball: HSV threshold, no circularity test (printed text and seams split the ball), fill-ratio
and aspect checks, reject blobs touching the image border, median depth of the central part of the ball with a 0.10 m
consistency check, ball centre = surface depth + ball radius.

  subscribes  /camera/camera/color/image_raw                    (launch realsense2_camera with align_depth.enable:=true)
              /camera/camera/aligned_depth_to_color/image_raw
              /camera/camera/color/camera_info
  publishes   /ball/position  geometry_msgs/PointStamped  frame "rear_axle": x forward, y left, z up [m]; stamp = IMAGE time
              ~/debug_image   sensor_msgs/Image           only when somebody subscribes

Set the camera mount: cam_x, cam_y, cam_z = camera position from the REAR AXLE [m] (x forward, y left, z up), cam_pitch_deg = tilt
downward. Measure them with a tape: a few centimetres of error here are errors in the ball position.
The D435i cannot measure closer than ~0.18-0.28 m: the ball in the scoop must be farther than that from the lens.
"""
import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PointStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image


def make_mask(bgr, h_min, h_max, s_min, v_min):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([min(h_min, h_max), s_min, v_min], np.uint8),
                       np.array([max(h_min, h_max), 255, 255], np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def find_ball(mask, min_area=150.0, min_radius=7.0, min_fill=0.35, min_aspect=0.60, max_aspect=1.60):
    """Largest compact blob -> (cx, cy, radius_px) or None. Same rules as the team's script."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    best, best_area = None, 0.0
    h_img, w_img = mask.shape
    for c in contours:
        area = float(cv2.contourArea(c))
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if not (min_aspect <= w / float(max(h, 1)) <= max_aspect):
            continue
        (cx, cy), r = cv2.minEnclosingCircle(c)
        if r < min_radius or area / max(math.pi * r * r, 1.0) < min_fill:
            continue
        if x <= 1 or y <= 1 or x + w >= w_img - 1 or y + h >= h_img - 1:       # cut by the image border
            continue
        if area > best_area:
            best_area, best = area, (int(round(cx)), int(round(cy)), float(r))
    return best


def surface_depth(depth_m, mask, cx, cy, radius_px, min_depth, max_depth):
    """Median depth of the central part of the coloured ball [m], or None when it is not trustworthy."""
    h, w = depth_m.shape
    r = max(3, int(round(radius_px * 0.40)))
    x0, x1, y0, y1 = max(cx - r, 0), min(cx + r + 1, w), max(cy - r, 0), min(cy + r + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    yy, xx = np.ogrid[y0:y1, x0:x1]
    patch = depth_m[y0:y1, x0:x1]
    ok = (((xx - cx) ** 2 + (yy - cy) ** 2) <= r * r) & (mask[y0:y1, x0:x1] > 0) & np.isfinite(patch) \
        & (patch >= min_depth) & (patch <= max_depth)
    d = patch[ok]
    if d.size < 6:
        return None
    med = float(np.median(d))
    good = d[np.abs(d - med) < 0.10]
    if good.size < max(6, int(0.6 * d.size)):
        return None
    return float(np.median(good))


def camera_to_car(pt_optical, pitch_rad, mount):
    """Optical frame (x right, y down, z forward) -> car frame (x forward, y left, z up) at the rear axle."""
    c, s = math.cos(pitch_rad), math.sin(pitch_rad)
    body_from_optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)     # optical -> (forward, left, up)
    tilt = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)                    # nose down by pitch
    return tilt @ body_from_optical @ np.asarray(pt_optical, float) + np.asarray(mount, float)


class BallDetector(Node):
    def __init__(self):
        super().__init__('ball_detector_node')
        params = dict(color_topic='/camera/camera/color/image_raw',
                      depth_topic='/camera/camera/aligned_depth_to_color/image_raw',
                      info_topic='/camera/camera/color/camera_info', output_topic='/ball/position', frame='rear_axle',
                      cam_x=0.15, cam_y=0.0, cam_z=0.25, cam_pitch_deg=20.0,
                      h_min=20, h_max=48, s_min=58, v_min=23,
                      min_area=150.0, min_radius=7.0, min_fill=0.35, min_aspect=0.60, max_aspect=1.60,
                      ball_radius=0.0335, min_depth=0.18, max_depth=4.0, depth_scale=0.001)
        for k, v in params.items():
            self.declare_parameter(k, v)
        self.p = {k: self.get_parameter(k).value for k in params}
        self.bridge = CvBridge()
        self.K = None
        self.pitch = math.radians(float(self.p['cam_pitch_deg']))
        self.mount = (float(self.p['cam_x']), float(self.p['cam_y']), float(self.p['cam_z']))
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
        self.K = (msg.k[0], msg.k[4], msg.k[2], msg.k[5])          # fx, fy, cx, cy

    def images_callback(self, color_msg, depth_msg):
        if self.K is None:
            return
        p = self.p
        bgr = self.bridge.imgmsg_to_cv2(color_msg, 'bgr8')
        depth_m = self.bridge.imgmsg_to_cv2(depth_msg, 'passthrough').astype(np.float32) * float(p['depth_scale'])
        mask = make_mask(bgr, int(p['h_min']), int(p['h_max']), int(p['s_min']), int(p['v_min']))
        det = find_ball(mask, float(p['min_area']), float(p['min_radius']), float(p['min_fill']),
                        float(p['min_aspect']), float(p['max_aspect']))
        out = bgr.copy() if self.dbg.get_subscription_count() > 0 else None
        if det is not None:
            cx, cy, r = det
            z_surface = surface_depth(depth_m, mask, cx, cy, r, float(p['min_depth']), float(p['max_depth']))
            if z_surface is not None:
                z = z_surface + float(p['ball_radius'])                    # surface -> centre
                fx, fy, ccx, ccy = self.K
                pt = camera_to_car([(cx - ccx) * z / fx, (cy - ccy) * z / fy, z], self.pitch, self.mount)
                msg = PointStamped()
                msg.header.stamp = color_msg.header.stamp                  # IMAGE time: the controller corrects for the delay
                msg.header.frame_id = p['frame']
                msg.point.x, msg.point.y, msg.point.z = float(pt[0]), float(pt[1]), float(pt[2])
                self.pub.publish(msg)
                if out is not None:
                    cv2.circle(out, (cx, cy), int(round(r)), (0, 255, 0), 2)
                    cv2.putText(out, f'x={pt[0]:.2f} y={pt[1]:+.2f}', (cx + 6, cy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if out is not None:
            self.dbg.publish(self.bridge.cv2_to_imgmsg(out, 'bgr8'))


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
