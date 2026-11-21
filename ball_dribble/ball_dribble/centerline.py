"""Closed centre-ish loop of the track, found by relaxing the raceline toward the ridge of the wall-distance field."""
import numpy as np
from .track import Track


def resample(pts, spacing):
    closed = np.vstack([pts, pts[:1]])
    seg = np.hypot(*np.diff(closed, axis=0).T)
    s = np.concatenate([[0], np.cumsum(seg)])
    n = int(round(s[-1] / spacing))
    ss = np.linspace(0, s[-1], n, endpoint=False)
    return np.column_stack([np.interp(ss, s, closed[:, 0]), np.interp(ss, s, closed[:, 1])])


def relax(track, raceline='raceline.csv', spacing=0.1, iters=600, step=0.01, smooth=0.35):
    d = np.genfromtxt(raceline, delimiter=',', skip_header=1)[:, :2]
    if np.hypot(*(d[0] - d[-1])) < 0.05:
        d = d[:-1]
    p = resample(d, spacing)
    for _ in range(iters):
        gx, gy = track.grad(p[:, 0], p[:, 1])
        p = p + step * np.column_stack([gx, gy]) / 0.05 * 0.1
        lap = 0.5 * (np.roll(p, 1, 0) + np.roll(p, -1, 0)) - p
        p = p + smooth * lap
        p = resample(p, spacing)
    return p


if __name__ == '__main__':
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    t = Track(); p = relax(t)
    np.save('centerline.npy', p)
    w = t.dist(p[:, 0], p[:, 1]); print('points', len(p), 'wall distance min %.2f mean %.2f max %.2f' % (w.min(), w.mean(), w.max()))
    ext = [t.ox, t.ox + t.W * t.pix, t.oy, t.oy + t.H * t.pix]
    fig, ax = plt.subplots(figsize=(8, 7)); ax.imshow(t.field, extent=ext, origin='upper'); ax.plot(p[:, 0], p[:, 1], 'r.', ms=2)
    ax.set_xlim(-8, 7); ax.set_ylim(-7.5, 6); ax.set_aspect('equal'); fig.savefig('centerline.png', dpi=70)
