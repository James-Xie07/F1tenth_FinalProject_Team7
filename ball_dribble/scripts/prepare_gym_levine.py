#!/usr/bin/env python3
"""Install local levine_blocked Gym map assets and a Gym-format CCW raceline.

This is a ONE-TIME development setup helper. It never modifies the real-car
levine2 map or levine_ball_path.csv. All overwritten Gym racelines are backed up.
Use --install to make filesystem changes; without it only prints a plan.
"""
import argparse
from datetime import datetime
from pathlib import Path
import shutil
import numpy as np


def create_raceline(src, dst):
    data = np.genfromtxt(src, delimiter=',', names=True)
    required = ('x', 'y', 'yaw', 'curvature')
    if not set(required).issubset(set(data.dtype.names or ())):
        raise ValueError(f'Expected CSV headers {required}, got {data.dtype.names}')
    x, y = np.asarray(data['x']), np.asarray(data['y'])
    if len(x) < 10 or not np.isfinite(np.c_[x, y]).all():
        raise ValueError('Invalid simulation waypoint file')
    if np.linalg.norm(np.array([x[0] - x[-1], y[0] - y[-1]])) < 1e-6:
        raise ValueError('Expected open waypoint loop; remove duplicate last point in input')
    arc = np.r_[0., np.cumsum(np.hypot(np.diff(x), np.diff(y)))]
    # Gym's CubicSplineND uses a provided s vector, so *explicitly* close the
    # data and s with one additional row; avoids N vs N+1 shape mismatch.
    closure = float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
    speed = np.full(len(x), 0.5)  # low-speed Gym reference, not a physical speed cap
    body = np.c_[arc, x, y, data['yaw'], data['curvature'], speed, np.zeros(len(x))]
    last = body[0].copy()
    last[0] = arc[-1] + closure
    body = np.vstack([body, last])
    # The installed Gym's loader defaults to skip_rows=3, with header_row=2.
    # Two comment rows followed by column names give the correct header row.
    with dst.open('w') as f:
        f.write('# Levine blocked, CCW Gym track\n')
        f.write('# Generated from levine_ccw_raceline.csv; vehicle centerline\n')
        f.write('s_m;x_m;y_m;psi_rad;kappa_radpm;vx_mps;ax_mps2\n')
        np.savetxt(f, body, delimiter=';', fmt='%.9f')
    return len(body), last[0]


def link_or_check(target, source, install):
    if target.exists() or target.is_symlink():
        if target.resolve() == source.resolve():
            print(f'Already linked: {target}')
            return
        raise RuntimeError(f'{target} already exists and is NOT the expected link. Check it manually.')
    if install:
        target.symlink_to(source.resolve())
    print(f'{"Linked" if install else "Would link"}: {target} -> {source}')


def main():
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parents[1]
    p.add_argument('--gym-root', type=Path,
                   default=Path.home()/'sim_ws/src/f1tenth_gym_ros')
    p.add_argument('--csv', type=Path, default=here/'paths/levine_ccw_raceline.csv')
    p.add_argument('--install', action='store_true', help='Apply changes (backup existing Gym raceline)')
    args = p.parse_args()
    gym = args.gym_root.expanduser().resolve()
    existing_maps = gym/'maps'
    target = gym/'f1tenth_gym/maps/levine_blocked'
    for suffix in ('.png', '.yaml'):
        source = existing_maps/f'levine_blocked{suffix}'
        if not source.exists():
            raise FileNotFoundError(f'Missing Gym map file: {source}')
    if not args.csv.is_file():
        raise FileNotFoundError(args.csv)
    print('Simulation reference: ', args.csv)
    print('Gym track folder:    ', target)
    print('Real-car levine2 files are not changed.')
    if not args.install:
        print('DRY RUN. Pass --install to prepare the Gym track.')
        return
    target.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.yaml'):
        link_or_check(target/f'levine_blocked{suffix}',
                      existing_maps/f'levine_blocked{suffix}', True)
    dst = target/'levine_blocked_raceline.csv'
    if dst.exists():
        backup = target/(dst.name + '.bak-' + datetime.now().strftime('%Y%m%d-%H%M%S'))
        shutil.copy2(dst, backup)
        print('Backed up old Gym raceline:', backup)
    n, length = create_raceline(args.csv, dst)
    print(f'Created {dst} with {n} rows including closure, {length:.2f} m CCW loop')
    print('IMPORTANT: verify overlay with /map in Foxglove BEFORE enabling motion.')


if __name__ == '__main__':
    main()
