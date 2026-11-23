"""Standard-library smoke tests for the compressed camera HTTP endpoints."""
import importlib.util
import pathlib
import sys
import threading
import types
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import urlopen


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / 'ball_dribble' / 'ball_video_web_node.py'


def load_video_module():
    class Node:
        pass

    class Enum:
        KEEP_LAST = BEST_EFFORT = 0

    fake_modules = {
        'cv2': types.SimpleNamespace(),
        'cv_bridge': types.SimpleNamespace(CvBridge=object),
        'rclpy': types.SimpleNamespace(ok=lambda: True),
        'rclpy.node': types.SimpleNamespace(Node=Node),
        'rclpy.qos': types.SimpleNamespace(
            HistoryPolicy=Enum, QoSProfile=object, ReliabilityPolicy=Enum),
        'sensor_msgs': types.ModuleType('sensor_msgs'),
        'sensor_msgs.msg': types.SimpleNamespace(Image=object),
    }
    with patch.dict(sys.modules, fake_modules):
        spec = importlib.util.spec_from_file_location('ball_video_web_node_under_test', MODULE_PATH)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class FakeVideoNode:
    stopping = False

    def __init__(self):
        self.frame = b'fake-jpeg'
        self.sequence = 7

    def latest_frame(self):
        return self.frame, self.sequence

    def wait_for_frame(self, _after_sequence, timeout):
        del timeout
        self.stopping = True
        return self.frame, self.sequence


class VideoWebTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_video_module()
        cls.node = FakeVideoNode()
        cls.server = cls.module.ThreadingHTTPServer(
            ('127.0.0.1', 0), cls.module.make_handler(cls.node))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = 'http://127.0.0.1:%d' % cls.server.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=2.0)

    def test_index_page(self):
        with urlopen(self.base + '/', timeout=2.0) as response:
            body = response.read()
            self.assertEqual(response.status, 200)
            self.assertIn(b'/stream.mjpg', body)
            self.assertIn(b'Team 7 Ball Detector', body)

    def test_snapshot_and_health(self):
        with urlopen(self.base + '/snapshot.jpg', timeout=2.0) as response:
            self.assertEqual(response.headers.get_content_type(), 'image/jpeg')
            self.assertEqual(response.read(), self.node.frame)
        with urlopen(self.base + '/healthz', timeout=2.0) as response:
            self.assertIn(b'ready sequence=7', response.read())

    def test_stream_is_mjpeg(self):
        self.node.stopping = False
        with urlopen(self.base + '/stream.mjpg', timeout=2.0) as response:
            body = response.read()
            self.assertEqual(response.headers.get_content_type(), 'multipart/x-mixed-replace')
            self.assertIn(b'Content-Type: image/jpeg', body)
            self.assertIn(self.node.frame, body)

    def test_missing_route_returns_404(self):
        with self.assertRaises(HTTPError) as caught:
            urlopen(self.base + '/missing', timeout=2.0)
        self.assertEqual(caught.exception.code, 404)
        caught.exception.close()


if __name__ == '__main__':
    unittest.main()
