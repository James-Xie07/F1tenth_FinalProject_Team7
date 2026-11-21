import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # run from a source checkout
"""Robustness experiments for the ball pusher. Usage: python3 experiments.py <group>"""
import sys, copy
import numpy as np
from multiprocessing import Pool
from ball_dribble import sim
from ball_dribble.ball_control import Config
from ball_dribble.physics import BallParams, CarParams, CameraParams


def job(args):
    label, kw = args
    kw = dict(kw)
    cfgkw = kw.pop('cfg', {})
    bp = kw.pop('ball', {})
    cp = kw.pop('car', {})
    cmp_ = kw.pop('cam', {})
    r = sim.run(cfg=Config(**cfgkw), ball_prm=BallParams(**bp), car_prm=CarParams(**cp), cam_prm=CameraParams(**cmp_), **kw)
    return label, r


def table(rows, seeds):
    res = {}
    with Pool(2) as p:
        for label, r in p.imap_unordered(job, [(l, dict(kw, seed=s)) for l, kw in rows for s in seeds]):
            res.setdefault(label, []).append(r)
    for label, _ in rows:
        rs = res[label]
        laps = [r['lap_time'] for r in rs if r['lap_time']]
        print('%-34s laps %d/%d  lap %5.1f s (%.1f-%.1f) | contact %3.0f%% | ball losses %d (recovered %d) | car hits %d | ball-wall %d | min gap %+.3f'
              % (label, len(laps), len(rs), np.mean(laps) if laps else float('nan'), min(laps) if laps else float('nan'), max(laps) if laps else float('nan'),
                 100 * np.mean([r['contact'] for r in rs]), sum(r['loss_events'] for r in rs), sum(r['recovered'] for r in rs),
                 sum(r['car_hits'] for r in rs), sum(r['ball_wall_hits'] for r in rs), min(r['min_clearance'] for r in rs)) + ('' if laps else ' | ball ended %.2f m from a wall' % rs[0]['ball_end_wall']), flush=True)
    return res


GROUPS = {}
GROUPS['floor'] = [  # the controller assumes ball_roll_decel = 0.35; the real floor may differ
    ('truth 0.35 (matches)', dict()),
    ('truth 0.25 (smoother floor)', dict(ball=dict(roll_decel=0.25))),
    ('truth 0.20', dict(ball=dict(roll_decel=0.20))),
    ('truth 0.50 (carpet-ish)', dict(ball=dict(roll_decel=0.50))),
    ('truth 0.25, controller told 0.25', dict(ball=dict(roll_decel=0.25), cfg=dict(ball_roll_decel=0.25))),
    ('truth 0.60, controller told 0.60', dict(ball=dict(roll_decel=0.60), cfg=dict(ball_roll_decel=0.60))),
    ('truth 1.00, controller told 1.00', dict(ball=dict(roll_decel=1.00), cfg=dict(ball_roll_decel=1.00))),
]
GROUPS['sensing'] = [
    ('baseline', dict()),
    ('camera at 0.30 (scoop in blind zone)', dict(cam=dict(x=0.30))),
    ('detection 80% of frames', dict(cam=dict(p_detect=0.80))),
    ('detection 50% of frames', dict(cam=dict(p_detect=0.50))),
    ('camera latency 0.15 s', dict(cam=dict(latency=0.15))),
    ('camera noise x3', dict(cam=dict(bearing_sigma=0.024, range_sigma0=0.012, range_sigma2=0.012))),
    ('pose noise 3 cm / 1.5 deg', dict(pose_noise=(0.03, 0.026))),
]
GROUPS['actuation'] = [
    ('baseline', dict()),
    ('command latency 0.2 s', dict(car=dict(cmd_latency=0.2))),
    ('command latency 0.2 (told so)', dict(car=dict(cmd_latency=0.2), cfg=dict(cmd_delay=0.2, lead=0.28))),
    ('servo lag 0.2 s', dict(car=dict(steer_tau=0.2))),
    ('speed loop lag 0.5 s', dict(car=dict(speed_tau=0.5))),
    ('wheelbase 0.40 (model says 0.46)', dict(car=dict(wheelbase=0.40))),
    ('wheelbase 0.52 (model says 0.46)', dict(car=dict(wheelbase=0.52))),
]
GROUPS['disturb'] = []
for tk in (4.0, 9.0, 14.0, 19.0):
    GROUPS['disturb'] += [
        ('knock ahead +1.2 m/s at %.0f s' % tk, dict(disturb=[(tk, 1.2, 0.0)], t_max=90)),
        ('knock backward 0.8 m/s at %.0f s' % tk, dict(disturb=[(tk, -0.8, 0.0)], t_max=90)),
    ]
if __name__ == '__main__':
    seeds = (0, 1, 2) if sys.argv[1] != 'disturb' else (0,)
    for g in sys.argv[1:]:
        print('=== ' + g)
        table(GROUPS[g], seeds)
