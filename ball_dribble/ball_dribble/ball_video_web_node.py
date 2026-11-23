#!/usr/bin/env python3
"""Serve the detector debug image as a throttled JPEG/MJPEG web page.

The source image remains on the robot's local ROS graph.  Only the JPEG stream
crosses the network, and the filtered Foxglove bridge intentionally does not
advertise either this source topic or the RealSense image topics.
"""
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
import socketserver
from threading import Condition, Thread
import time
from urllib.parse import urlsplit

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image


INDEX_HTML = b"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Team 7 Ball Camera</title>
  <style>
    :root { color-scheme: dark; font-family: system-ui, sans-serif; }
    body { margin: 0; background: #0b1017; color: #eef4fb; }
    main { width: min(1100px, 100%); margin: 0 auto; padding: 1rem; box-sizing: border-box; }
    header { display: flex; align-items: center; justify-content: space-between; gap: 1rem; flex-wrap: wrap; }
    h1 { margin: 0; font-size: 1.35rem; }
    #status { display: flex; align-items: center; gap: .5rem; color: #aebdca; }
    #dot { width: .7rem; height: .7rem; border-radius: 50%; background: #d4a72c; }
    figure { margin: 1rem 0 0; overflow: hidden; border-radius: .75rem; background: #020407; border: 1px solid #263546; }
    img { display: block; width: 100%; height: auto; min-height: 12rem; object-fit: contain; }
    nav { display: flex; gap: .75rem; margin-top: 1rem; align-items: center; flex-wrap: wrap; }
    button, a { border: 1px solid #3a526b; border-radius: .5rem; padding: .65rem .9rem; color: #eef4fb; background: #182433; font: inherit; text-decoration: none; cursor: pointer; }
    p { color: #aebdca; line-height: 1.5; }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Team 7 Ball Detector</h1>
      <div id="status"><span id="dot"></span><span id="status-text">Waiting for camera</span></div>
    </header>
    <figure><img id="camera" src="/stream.mjpg" alt="Ball detector camera stream"></figure>
    <nav>
      <button id="toggle" type="button">Pause stream</button>
      <a href="/snapshot.jpg" target="_blank" rel="noopener">Open snapshot</a>
    </nav>
    <p>This page receives a throttled JPEG stream. RealSense image and depth topics are not sent through Foxglove.</p>
  </main>
  <script>
    const camera = document.querySelector('#camera');
    const toggle = document.querySelector('#toggle');
    const dot = document.querySelector('#dot');
    const statusText = document.querySelector('#status-text');
    let running = true;
    camera.addEventListener('load', () => {
      dot.style.background = '#3fb950';
      statusText.textContent = 'Stream connected';
    });
    camera.addEventListener('error', () => {
      dot.style.background = '#f85149';
      statusText.textContent = 'Stream unavailable';
    });
    toggle.addEventListener('click', () => {
      running = !running;
      if (running) {
        camera.src = '/stream.mjpg?t=' + Date.now();
        toggle.textContent = 'Pause stream';
        statusText.textContent = 'Connecting';
        dot.style.background = '#d4a72c';
      } else {
        camera.removeAttribute('src');
        toggle.textContent = 'Resume stream';
        statusText.textContent = 'Paused';
        dot.style.background = '#768390';
      }
    });
  </script>
</body>
</html>
"""


class ThreadingHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_handler(node):
    class VideoHandler(BaseHTTPRequestHandler):
        server_version = 'BallVideoHTTP/1.0'

        def log_message(self, _format, *_args):
            # Browser reconnects are normal for an MJPEG stream; avoid terminal spam.
            return

        def _headers(self, status, content_type, content_length=None):
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
            self.send_header('Pragma', 'no-cache')
            self.send_header('X-Content-Type-Options', 'nosniff')
            if content_length is not None:
                self.send_header('Content-Length', str(content_length))
            self.end_headers()

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/':
                self._headers(HTTPStatus.OK, 'text/html; charset=utf-8', len(INDEX_HTML))
                self.wfile.write(INDEX_HTML)
            elif path == '/stream.mjpg':
                self._stream()
            elif path == '/snapshot.jpg':
                frame, _ = node.latest_frame()
                if frame is None:
                    body = b'No camera frame has been encoded yet.\n'
                    self._headers(HTTPStatus.SERVICE_UNAVAILABLE, 'text/plain; charset=utf-8', len(body))
                    self.wfile.write(body)
                else:
                    self._headers(HTTPStatus.OK, 'image/jpeg', len(frame))
                    self.wfile.write(frame)
            elif path == '/healthz':
                frame, sequence = node.latest_frame()
                body = ('ready sequence=%d\n' % sequence if frame is not None else 'waiting for camera\n').encode()
                status = HTTPStatus.OK if frame is not None else HTTPStatus.SERVICE_UNAVAILABLE
                self._headers(status, 'text/plain; charset=utf-8', len(body))
                self.wfile.write(body)
            else:
                body = b'Not found\n'
                self._headers(HTTPStatus.NOT_FOUND, 'text/plain; charset=utf-8', len(body))
                self.wfile.write(body)

        def _stream(self):
            self._headers(HTTPStatus.OK, 'multipart/x-mixed-replace; boundary=frame')
            last_sequence = -1
            try:
                while rclpy.ok() and not node.stopping:
                    frame, sequence = node.wait_for_frame(last_sequence, timeout=2.0)
                    if frame is None or sequence == last_sequence:
                        continue
                    last_sequence = sequence
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(('Content-Length: %d\r\n\r\n' % len(frame)).encode())
                    self.wfile.write(frame)
                    self.wfile.write(b'\r\n')
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                pass

    return VideoHandler


class BallVideoWebNode(Node):
    def __init__(self):
        super().__init__('ball_video_web_node')
        defaults = {
            'source_topic': '/ball_detector_node/debug_image',
            'bind_address': '0.0.0.0',
            'port': 8080,
            'max_fps': 10.0,
            'jpeg_quality': 65,
            'output_width': 640,
        }
        for key, value in defaults.items():
            self.declare_parameter(key, value)
        self.p = {key: self.get_parameter(key).value for key in defaults}

        port = int(self.p['port'])
        quality = int(self.p['jpeg_quality'])
        max_fps = float(self.p['max_fps'])
        width = int(self.p['output_width'])
        if not 1 <= port <= 65535:
            raise ValueError('port must be between 1 and 65535')
        if not 20 <= quality <= 95:
            raise ValueError('jpeg_quality must be between 20 and 95')
        if not 0.5 <= max_fps <= 30.0:
            raise ValueError('max_fps must be between 0.5 and 30')
        if width != 0 and width < 64:
            raise ValueError('output_width must be 0 or at least 64 pixels')

        self.bridge = CvBridge()
        self.quality = quality
        self.width = width
        self.min_period = 1.0 / max_fps
        self.last_encode_at = -float('inf')
        self.condition = Condition()
        self.jpeg = None
        self.sequence = 0
        self.stopping = False

        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                         reliability=ReliabilityPolicy.BEST_EFFORT)
        self.create_subscription(Image, str(self.p['source_topic']), self.image_callback, qos)

        address = str(self.p['bind_address'])
        self.httpd = ThreadingHTTPServer((address, port), make_handler(self))
        self.http_thread = Thread(target=self.httpd.serve_forever,
                                  kwargs={'poll_interval': 0.2}, daemon=True)
        self.http_thread.start()
        self.get_logger().info(
            f'Compressed ball video ready at http://<robot-host>:{port}/ '
            f'({width if width else "native"} px, {max_fps:.1f} FPS, JPEG quality {quality})')

    def image_callback(self, msg):
        now = time.monotonic()
        if now - self.last_encode_at < self.min_period:
            return
        self.last_encode_at = now
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            if self.width and image.shape[1] > self.width:
                height = max(1, int(round(image.shape[0] * self.width / image.shape[1])))
                image = cv2.resize(image, (self.width, height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode('.jpg', image,
                                       [cv2.IMWRITE_JPEG_QUALITY, self.quality])
            if not ok:
                raise RuntimeError('cv2.imencode returned false')
            frame = encoded.tobytes()
        except Exception as exc:
            self.get_logger().warning(f'Cannot encode camera frame: {exc}', throttle_duration_sec=2.0)
            return
        with self.condition:
            self.jpeg = frame
            self.sequence += 1
            self.condition.notify_all()

    def latest_frame(self):
        with self.condition:
            return self.jpeg, self.sequence

    def wait_for_frame(self, after_sequence, timeout):
        with self.condition:
            self.condition.wait_for(
                lambda: self.sequence != after_sequence or self.stopping,
                timeout=timeout)
            return self.jpeg, self.sequence

    def shutdown_web(self):
        with self.condition:
            self.stopping = True
            self.condition.notify_all()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.http_thread.join(timeout=2.0)


def main(args=None):
    rclpy.init(args=args)
    node = None
    try:
        node = BallVideoWebNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.shutdown_web()
            node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
