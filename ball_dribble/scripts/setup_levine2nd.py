"""Sets up the sim on the real Levine 2nd-floor map (levine_2nd_new.pgm, the SLAM scan from lab 5).

    python3 scripts/setup_levine2nd.py [levine_2nd_new.pgm] [clearance=0.25] [kcap=0.5]      (needs cvxpy)

Writes
    maps/levine_2nd.png, .yaml, _centerline.csv   cleaned map: free space = largest free blob, everything else is a wall
                                                  (the scan has grey 'unknown' outside the corridor that the gym would treat as free)
    paths/ball_path_levine2nd.csv                 ball path (same format as ball_path.csv), counter-clockwise
    config/gym_levine2nd.yaml                     gym_bridge config: this map + start pose on the path, ball resting in the scoop
    config/sim_levine2nd.yaml                     ball_sim_node + ball_push_node parameters for this map
The corridor is only ~1.5 m wide.  The ball path is the MOST CENTRED line whose curvature stays below `kcap` (so the car can
follow it, its minimum turning radius is 0.74 m): straights stay on the centre line, only the corners are cut as much as needed.
It starts from the contour of the corridor at a constant distance from the central hole of the map (smoothed), and a QP
(cvxpy, Clarabel) moves every point sideways within `clearance` metres of the walls.  (make_ball_path.py, which minimises
curvature only, hugs one wall on the straights: wall distance 0.24 m instead of 0.75 m.)
"""
import math
import pathlib
import sys

import numpy as np
import yaml
from PIL import Image
from scipy import ndimage as ndi

HERE = pathlib.Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

ORIGIN = [-15.174, -43.331, 0.0]      # from levine_2nd.yaml
RES = 0.05
GYM_CONFIG = pathlib.Path.home() / 'sim_ws/src/f1tenth_gym_ros/config/sim.yaml'


def clean_map(pgm):
    p = np.array(Image.open(pgm))
    lab, n = ndi.label(p >= 250)                                   # 254/255 = free, 205 = unknown, 0 = occupied
    main = lab == 1 + int(np.argmax(ndi.sum(lab > 0, lab, range(1, n + 1))))
    holes, m = ndi.label(~main)
    edge = set(np.unique(np.concatenate([holes[0], holes[-1], holes[:, 0], holes[:, -1]])))
    inner = [i for i in range(1, m + 1) if i not in edge]
    if not inner:
        raise RuntimeError('no enclosed hole in the map: not a loop?')
    hole = holes == max(inner, key=lambda i: (holes == i).sum())
    return main, hole


def loop_through_corridor(main, hole):
    """Closed contour at the distance from the hole that keeps the largest clearance to the outer wall."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d_hole = ndi.distance_transform_edt(~hole) * RES
    d_wall = ndi.distance_transform_edt(main) * RES
    H, W = main.shape
    best = None
    for level in np.arange(0.5, 1.5, 0.05):
        cs = plt.contour(d_hole, levels=[level])
        pts = max(cs.allsegs[0], key=len)
        plt.close('all')
        c = np.clip(pts[:, 0].round().astype(int), 0, W - 1)
        r = np.clip(pts[:, 1].round().astype(int), 0, H - 1)
        score = np.percentile(d_wall[r, c], 5)
        if best is None or score > best[0]:
            best = (score, level, pts)
    score, level, pts = best
    print('loop at %.2f m from the hole, wall distance (5th percentile) %.2f m' % (level, score))
    xy = np.column_stack([ORIGIN[0] + pts[:, 0] * RES, ORIGIN[1] + (H - pts[:, 1]) * RES])   # same convention as Track._rc
    xy = xy[:-1] if np.hypot(*(xy[0] - xy[-1])) < 0.2 else xy
    area = 0.5 * np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1])
    return xy if area > 0 else xy[::-1]                             # counter-clockwise


def smooth_loop(loop, spacing, smooth=0.06):
    """The contour has pixel staircases (curvature up to 26 1/m): fit a periodic spline and resample evenly."""
    from scipy.interpolate import splprep, splev
    from ball_dribble.centerline import resample
    q = resample(loop, 0.1)
    q = np.vstack([q, q[:1]])
    tck, _ = splprep([q[:, 0], q[:, 1]], s=len(q) * smooth ** 2, per=True)
    x, y = splev(np.linspace(0, 1, 40001), tck)
    return resample(np.column_stack([x, y])[:-1], spacing)


def centered_path(track, loop, clearance, kcap, out, spacing=0.05, w_smooth=0.02, w_cap=50.0):
    """Most centred closed path with |curvature| <= kcap (soft: a few points at map noise may exceed it slightly).

    minimise  sum alpha^2 + w_smooth * sum curvature^2 + w_cap * sum max(0, curvature - kcap)^2,  alpha = sideways offset of each point
    from the centre line, |alpha| limited by the corridor that keeps `clearance` to the walls.  Moving the points sideways changes their
    spacing, so the curvature limit is rescaled by the real spacing and the QP is solved four times."""
    import cvxpy as cp
    import scipy.sparse as sp
    import make_ball_path
    solve = cp.Problem.solve
    cp.Problem.solve = lambda self, *a, **k: solve(self, solver=cp.CLARABEL)     # OSQP (make_ball_path's choice) does not converge here
    base = smooth_loop(loop, spacing)
    n = len(base)
    tang = np.roll(base, -1, 0) - np.roll(base, 1, 0)
    tang /= np.linalg.norm(tang, axis=1, keepdims=True)
    normals = np.column_stack([-tang[:, 1], tang[:, 0]])
    lo, hi = make_ball_path.corridor(track, base, normals, clearance, reach=2.0)
    D = sp.diags([1, -2, 1], [-1, 0, 1], shape=(n, n), format='lil', dtype=float)
    D[0, n - 1] = 1
    D[n - 1, 0] = 1
    D = D.tocsr()
    cap = np.full(n, kcap)
    for _ in range(4):
        alpha = cp.Variable(n)
        kx = D @ (base[:, 0] + cp.multiply(normals[:, 0], alpha)) / spacing ** 2
        ky = D @ (base[:, 1] + cp.multiply(normals[:, 1], alpha)) / spacing ** 2
        k = cp.norm(cp.vstack([kx, ky]), axis=0)
        cp.Problem(cp.Minimize(cp.sum_squares(alpha) + w_smooth * (cp.sum_squares(kx) + cp.sum_squares(ky))
                               + w_cap * cp.sum_squares(cp.pos(k - cap))), [alpha >= lo, alpha <= hi]).solve()
        q = base + alpha.value[:, None] * normals
        cap = kcap * (np.hypot(*((np.roll(q, -1, 0) - np.roll(q, 1, 0)) / 2).T) / spacing) ** 2
    return make_ball_path.finish(q, track, out)


def main_(pgm, clearance, kcap):
    from ball_dribble.track import Track
    from ball_dribble.ball_control import Config
    from ball_dribble.physics import make_scoop, rider_x

    free, hole = clean_map(pgm)
    Image.fromarray(np.where(free, 255, 0).astype(np.uint8)).save(PKG / 'maps/levine_2nd.png')
    meta = dict(image='levine_2nd.png', resolution=RES, origin=ORIGIN, negate=0, occupied_thresh=0.65, free_thresh=0.196,
                centerline='levine_2nd_centerline.csv')
    yaml.safe_dump(meta, open(PKG / 'maps/levine_2nd.yaml', 'w'), default_flow_style=None, sort_keys=False)

    loop = loop_through_corridor(free, hole)
    track = Track.from_map_yaml(str(PKG / 'maps/levine_2nd.yaml'))
    w = track.dist(loop[:, 0], loop[:, 1])
    with open(PKG / 'maps/levine_2nd_centerline.csv', 'w') as f:
        f.write('# x_m, y_m, w_tr_right_m, w_tr_left_m\n# levine_2nd loop, counter-clockwise, widths = wall distance; made by scripts/setup_levine2nd.py\n')
        np.savetxt(f, np.column_stack([loop, w, w])[::5], fmt='%.4f', delimiter=',')

    out = PKG / 'paths/ball_path_levine2nd.csv'
    arr = centered_path(track, loop, clearance, kcap, str(out))
    kappa = np.abs(arr[:, 3])
    length = np.sum(np.hypot(*np.diff(np.vstack([arr[:, :2], arr[:1, :2]]), axis=0).T))
    print('ball path: %d points, %.2f m, |kappa| max %.2f p95 %.2f, wall distance min %.2f p5 %.2f mean %.2f m -> %s'
          % (len(arr), length, kappa.max(), np.percentile(kappa, 95), arr[:, 4].min(), np.percentile(arr[:, 4], 5), arr[:, 4].mean(), out))

    # start: the ball (and so the scoop) on the straightest stretch of the path, the car just behind it
    win = int(6.0 / 0.05)
    smooth = np.convolve(np.r_[kappa[-win:], kappa, kappa[:win]], np.ones(win) / win, mode='same')[win:-win]
    i = int(np.argmin(smooth))
    ball, yaw = arr[i, :2], arr[i, 2]
    xb = rider_x(make_scoop('channel', apex_x=Config().scoop_apex_x), 0.033)
    off = 0.0                                                       # the gym bridge's odom / base_link is the rear axle
    odom = ball - (xb - off) * np.array([math.cos(yaw), math.sin(yaw)])
    print('start: ball (%.2f, %.2f) yaw %.2f, gym pose (%.3f, %.3f)' % (ball[0], ball[1], yaw, odom[0], odom[1]))

    gym = yaml.safe_load(open(GYM_CONFIG))
    rp = gym['bridge']['ros__parameters']
    rp.update(sx=float(odom[0]), sy=float(odom[1]), stheta=float(yaw), num_agents=1, map_path=str(PKG / 'maps/levine_2nd'))
    (PKG / 'config/gym_levine2nd.yaml').write_text('# generated by scripts/setup_levine2nd.py from f1tenth_gym_ros/config/sim.yaml\n'
                                                 + yaml.safe_dump(gym, sort_keys=False))
    sim = yaml.safe_load(open(PKG / 'config/sim.yaml'))
    sim['ball_push_node']['ros__parameters']['waypoint_file'] = 'ball_path_levine2nd.csv'
    (PKG / 'config/sim_levine2nd.yaml').write_text('# generated by scripts/setup_levine2nd.py (config/sim.yaml with the Levine 2nd-floor path)\n'
                                                 + yaml.safe_dump(sim, sort_keys=False))


if __name__ == '__main__':
    pgm = sys.argv[1] if len(sys.argv) > 1 else str(pathlib.Path.home() / 'roboracer_ws/src/ese-6150-lab-5-pure-pursuit-group-7/map/levine_2nd_new.pgm')
    main_(pgm, float(sys.argv[2]) if len(sys.argv) > 2 else 0.25, float(sys.argv[3]) if len(sys.argv) > 3 else 0.5)
