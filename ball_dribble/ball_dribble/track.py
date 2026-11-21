"""Track geometry for the ball-pushing project.

The map file (.pgm) that was uploaded is corrupted, so the free space is rebuilt from the
'centerline' picture of the same map (white = free, black/grey = wall/unknown), calibrated to map
coordinates with the known grey-rectangle extent of the map (resolution 0.05, origin -13.115 -10.466).
`build()` writes track_cache.npz once; `Track` then answers wall-distance queries for the sim and for the
path generator.
"""
import numpy as np
from scipy import ndimage

import os
PICTURE = os.environ.get('BALL_TRACK_PICTURE', 'centerline.png')   # only used by build() for the old race2 map
DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'paths', 'race2_track.npz')
RES, ORIGIN = 0.05, (-13.115, -10.466)
MAP_W, MAP_H = 433, 414


def build(raceline_csv='raceline.csv', out='track_cache.npz', picture=PICTURE):
    from PIL import Image
    im = np.array(Image.open(picture).convert('RGB')).astype(int)
    r, g, b = im[..., 0], im[..., 1], im[..., 2]
    gray = (abs(r - g) < 6) & (abs(g - b) < 6) & (r > 190) & (r < 215)
    cols = np.where(gray.sum(0) > 200)[0]
    rows = np.where(gray.sum(1) > 200)[0]
    x0, x1, y0, y1 = cols.min(), cols.max() + 1, rows.min(), rows.max() + 1
    sx, sy = (x1 - x0) / (MAP_W * RES), (y1 - y0) / (MAP_H * RES)
    white = (r > 240) & (g > 240) & (b > 240)
    blue = (b > 180) & (r < 80) & (g < 80)
    teal = (g > 120) & (b > 120) & (r < 80) & ~blue
    free = white | blue | teal
    free[:, :x0] = free[:, x1:] = False
    free[:y0] = free[y1:] = False
    lab, _ = ndimage.label(free)
    d = np.genfromtxt(raceline_csv, delimiter=',', skip_header=1)
    px = (x0 + (d[:, 0] - ORIGIN[0]) * sx).astype(int)
    py = (y1 - (d[:, 1] - ORIGIN[1]) * sy).astype(int)
    ids = [i for i in np.unique(lab[py, px]) if i > 0]
    sizes = [(lab == i).sum() for i in ids]
    keep = ids[int(np.argmax(sizes))]
    free = ndimage.binary_closing(lab == keep, iterations=1)
    free = free[y0:y1, x0:x1]                       # crop to the map rectangle, row 0 = top
    pix = 1.0 / sx                                  # metres per picture pixel
    np.savez_compressed(out, free=free, pix=pix, origin=np.array(ORIGIN), height_m=free.shape[0] * pix)
    return out


class Track:
    """Wall-distance field. `dist(x, y)` is the distance to the nearest wall in metres (0 inside walls).

    Track()                      the demo track shipped in paths/ (the lab's race2 map)
    Track(cache='x.npz')         a track saved with save()
    Track.from_map_yaml('m.yaml')  any ROS map (yaml + pgm/png), e.g. the Levine corridor
    """

    def __init__(self, cache=None, free=None, pix=None, origin=None):
        if free is None:
            z = np.load(cache or DEFAULT_CACHE)
            free, pix, origin = z['free'], float(z['pix']), z['origin']
        self.free, self.pix = np.asarray(free, bool), float(pix)
        self.ox, self.oy = map(float, origin)
        self.H, self.W = self.free.shape
        self.field = ndimage.distance_transform_edt(self.free) * self.pix
        self.field = ndimage.gaussian_filter(self.field, 0.7)   # remove 1-px staircase so gradients are smooth

    @classmethod
    def from_map_yaml(cls, yaml_file, keep_largest=True):
        """Free space of a ROS map_server map: row 0 of the image is the top (largest y). Needs PyYAML and Pillow."""
        import os
        import yaml
        from PIL import Image
        meta = yaml.safe_load(open(yaml_file))
        img = Image.open(os.path.join(os.path.dirname(os.path.abspath(yaml_file)), meta['image'])).convert('L')
        pix = np.array(img, float)
        occ = (255 - pix) / 255.0 if not meta.get('negate', 0) else pix / 255.0
        free = occ < float(meta.get('free_thresh', 0.196))
        if keep_largest:                                   # drop the unreachable "free" pixels outside the walls
            lab, n = ndimage.label(free)
            if n > 1:
                free = lab == (1 + int(np.argmax(ndimage.sum(free, lab, range(1, n + 1)))))
        return cls(free=free, pix=float(meta['resolution']), origin=meta['origin'][:2])

    def save(self, out):
        np.savez_compressed(out, free=self.free, pix=self.pix, origin=np.array([self.ox, self.oy]))

    def _rc(self, x, y):
        col = (np.asarray(x, float) - self.ox) / self.pix
        row = self.H - (np.asarray(y, float) - self.oy) / self.pix
        return row, col

    def dist(self, x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        row, col = self._rc(x, y)
        out = ndimage.map_coordinates(self.field, [row.ravel(), col.ravel()], order=1, mode='nearest')
        return out.reshape(x.shape)

    def grad(self, x, y, h=0.02):
        gx = (self.dist(x + h, y) - self.dist(x - h, y)) / (2 * h)
        gy = (self.dist(x, y + h) - self.dist(x, y - h)) / (2 * h)
        return gx, gy


if __name__ == '__main__':
    build()
    t = Track()
    d = np.genfromtxt('raceline.csv', delimiter=',', skip_header=1)
    w = t.dist(d[:, 0], d[:, 1])
    print('free area %.1f m^2' % (t.free.sum() * t.pix ** 2), '| raceline wall distance min %.3f mean %.3f max %.3f' % (w.min(), w.mean(), w.max()))
