import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # run from a source checkout
"""Builds ball_path.csv: a smooth closed path for the BALL that stays clear of the walls.

Starts from the raceline (only to get a closed loop and its normals), then solves a small QP that moves every
point sideways, inside the free corridor with `clearance` metres to the walls, to minimise curvature.
Columns: x, y, yaw, curvature, wall_distance. Run:  python3 make_ball_path.py [clearance]
"""
import sys
import numpy as np
import cvxpy as cp
from ball_dribble.track import Track


def corridor(track, pts, normals, clearance, reach=1.6, step=0.01):
    """Allowed sideways range [lo, hi] along each normal: the connected stretch of free space where the wall distance >= clearance."""
    a = np.arange(-reach, reach + step / 2, step)
    lo, hi = np.zeros(len(pts)), np.zeros(len(pts))
    for i, (p, n) in enumerate(zip(pts, normals)):
        d = track.dist(p[0] + a * n[0], p[1] + a * n[1])
        ok = d >= clearance
        if not ok.any():
            raise RuntimeError('no clear space near point %d' % i)
        idx = np.where(ok)[0]                    # start from the clear sample nearest to the base point (alpha = 0)
        centre = idx[np.argmin(np.abs(idx - len(a) // 2))]
        left = right = centre
        while left > 0 and ok[left - 1]:
            left -= 1
        while right < len(a) - 1 and ok[right + 1]:
            right += 1
        lo[i], hi[i] = a[left], a[right]
    return lo, hi


def build(clearance=0.30, spacing=0.05, out='ball_path.csv', track=None, iterations=25, kappa_cap=0.65):
    from centerline import relax, resample
    track = track or Track()
    base = resample(relax(track), spacing)
    weight = None
    for it in range(iterations + 1):
        q = _qp_pass(track, base, clearance, spacing, weight)
        kappa = _curvature(q)
        # re-weight: bends that are still sharper than the car can follow count more in the next pass
        weight = 1.0 + (np.abs(kappa) / kappa_cap) ** 4
        base = resample(q, spacing)
        weight = np.interp(np.linspace(0, 1, len(base), endpoint=False), np.linspace(0, 1, len(q), endpoint=False), weight)
    return finish(q, track, out)


def _curvature(q):
    t = np.roll(q, -1, 0) - np.roll(q, 1, 0)
    yaw = np.unwrap(np.arctan2(t[:, 1], t[:, 0]))
    ds = np.hypot(*(np.roll(q, -1, 0) - q).T)
    return np.angle(np.exp(1j * (np.roll(yaw, -1) - np.roll(yaw, 1)))) / (np.roll(ds, 1) + ds)


def _qp_pass(track, base, clearance, spacing, weight=None):
    import scipy.sparse as sp
    n_pts = len(base)
    tang = np.roll(base, -1, 0) - np.roll(base, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normals = np.column_stack([-tang[:, 1], tang[:, 0]])
    lo, hi = corridor(track, base, normals, clearance, reach=2.0)
    D = sp.diags([1, -2, 1], [-1, 0, 1], shape=(n_pts, n_pts), format='lil', dtype=float)
    D[0, n_pts - 1] = 1
    D[n_pts - 1, 0] = 1
    D = D.tocsr()
    alpha = cp.Variable(n_pts)
    px = base[:, 0] + cp.multiply(normals[:, 0], alpha)
    py = base[:, 1] + cp.multiply(normals[:, 1], alpha)
    w = np.ones(n_pts) if weight is None else weight
    cost = (cp.sum(cp.multiply(w, cp.square(D @ px) + cp.square(D @ py)))) / spacing ** 4   # = weighted sum of curvature^2
    prob = cp.Problem(cp.Minimize(cost + 1e-3 * cp.sum_squares(alpha)), [alpha >= lo, alpha <= hi])
    prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=80000)
    return base + alpha.value[:, None] * normals


def finish(q, track, out, smooth=0.02):
    """Fit a periodic spline (removes the zig-zag of the point-wise QP), resample evenly and take exact curvature."""
    from scipy.interpolate import splprep, splev
    q = np.vstack([q, q[:1]])
    tck, _ = splprep([q[:, 0], q[:, 1]], s=len(q) * smooth ** 2, per=True)
    u = np.linspace(0, 1, 20001)
    x, y = splev(u, tck)
    arc = np.concatenate([[0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])
    n = int(round(arc[-1] / 0.05))
    uu = np.interp(np.linspace(0, arc[-1], n, endpoint=False), arc, u)
    x, y = splev(uu, tck)
    dx, dy = splev(uu, tck, der=1)
    ddx, ddy = splev(uu, tck, der=2)
    kappa = (dx * ddy - dy * ddx) / (dx ** 2 + dy ** 2) ** 1.5
    yaw = np.arctan2(dy, dx)
    w = track.dist(x, y)
    arr = np.column_stack([x, y, yaw, kappa, w])
    np.savetxt(out, arr, delimiter=',', header='x,y,yaw,curvature,wall_distance', comments='')
    return arr


if __name__ == '__main__':
    c = float(sys.argv[1]) if len(sys.argv) > 1 else 0.30
    arr = build(clearance=c)
    k = np.abs(arr[:, 3])
    L = np.sum(np.hypot(*np.diff(np.vstack([arr[:, :2], arr[:1, :2]]), axis=0).T))
    print('clearance %.2f  length %.2f m  points %d  |kappa| max %.2f p95 %.2f  wall distance min %.3f' % (c, L, len(arr), k.max(), np.percentile(k, 95), arr[:, 4].min()))
