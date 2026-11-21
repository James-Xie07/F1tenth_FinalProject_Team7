#!/usr/bin/env python3
"""Measure the ball's rolling deceleration on YOUR floor (the single most important number: lap time scales with it).

Roll the ball by hand across the floor (no car), record /ball/position (car stationary, so car frame = fixed frame):
  ros2 topic echo --csv /ball/position > roll.csv      (columns: sec, nanosec, ..., x, y, z)  -- or give a 3-column CSV t,x,y
Usage: python3 calibrate_ball.py roll.csv [more.csv ...]
Fits x(t) = x0 + v0 t - a t^2/2 per roll (while moving) and prints the median a.  Use ~0.8 x that as ball_roll_decel for margin.
"""
import sys
import numpy as np

def load(fn):
    rows = []
    for line in open(fn):
        parts = [p for p in line.replace('"', '').strip().split(',') if p]
        try:
            vals = [float(p) for p in parts]
        except ValueError:
            continue
        if len(vals) >= 5:   # ros2 topic echo --csv: sec,nanosec,frame..., x,y,z  (frame_id is non-numeric and dropped)
            rows.append((vals[0] + vals[1] * 1e-9, vals[-3], vals[-2]))
        elif len(vals) == 3:
            rows.append(tuple(vals))
    return np.array(rows)

def fit(d):
    t = d[:, 0] - d[0, 0]; p = d[:, 1:3]
    # keep samples while the ball moves faster than 0.1 m/s
    v = np.gradient(p, t, axis=0); sp = np.linalg.norm(v, axis=1)
    m = sp > 0.1
    if m.sum() < 8:
        return None
    t, p = t[m], p[m]
    d0 = (p[-1] - p[0]); u = d0 / np.linalg.norm(d0)
    s = (p - p[0]) @ u                       # distance along the roll
    c = np.polyfit(t, s, 2)                  # s = c2 t^2 + c1 t + c0 ; a = -2 c2
    return -2 * c[0]

vals = []
for fn in sys.argv[1:]:
    a = fit(load(fn))
    print(fn, 'decel = %s' % ('too few moving samples' if a is None else '%.3f m/s^2' % a))
    if a is not None and a > 0:
        vals.append(a)
if vals:
    print('median %.3f m/s^2  ->  suggested ball_roll_decel %.2f' % (np.median(vals), 0.8 * np.median(vals)))
