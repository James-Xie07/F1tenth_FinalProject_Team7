"""Wall distance field from a nav_msgs/OccupancyGrid (the /map topic)."""
import math

import numpy as np
from scipy import ndimage


class MapWalls:
    """dist(x, y) = distance to the nearest occupied cell [m]; grad(x, y) = its gradient (points away from the wall)."""

    def __init__(self, msg, occupied_threshold=65, unknown_is_wall=True):
        w, h, self.res = msg.info.width, msg.info.height, msg.info.resolution
        data = np.asarray(msg.data, dtype=np.int16).reshape(h, w)
        occupied = (data >= occupied_threshold) | ((data < 0) if unknown_is_wall else False)
        self.field = ndimage.distance_transform_edt(~occupied) * self.res
        self.field = ndimage.gaussian_filter(self.field, 0.7)
        o = msg.info.origin
        q = o.orientation
        self.ox, self.oy = o.position.x, o.position.y
        self.oyaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))

    def dist(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        c, s = math.cos(self.oyaw), math.sin(self.oyaw)
        dx, dy = x - self.ox, y - self.oy
        col = (c * dx + s * dy) / self.res - 0.5
        row = (-s * dx + c * dy) / self.res - 0.5          # OccupancyGrid row 0 is the lowest y (no image flip)
        out = ndimage.map_coordinates(self.field, [row.ravel(), col.ravel()], order=1, mode='nearest')
        return out.reshape(x.shape)

    def grad(self, x, y, h=0.02):
        gx = (self.dist(x + h, y) - self.dist(x - h, y)) / (2 * h)
        gy = (self.dist(x, y + h) - self.dist(x, y - h)) / (2 * h)
        return gx, gy
