"""ROS-free closed-loop simulation of the ball-pushing controller (car + scoop + rolling ball + depth camera)."""
import math
import numpy as np
from .track import Track
from .physics import (Car, CarParams, Camera, CameraParams, BallParams, ball_step, car_to_world, car_velocity_field,
                     footprint_points, make_scoop, wall_clearance, wrap)
from .ball_control import BallPusher, Config

_TRACK = None


def get_track():
    global _TRACK
    if _TRACK is None:
        _TRACK = Track()
    return _TRACK


def run(cfg=None, scoop='channel', seed=0, t_max=70.0, ball_prm=None, car_prm=None, cam_prm=None, start_s=0.0, ball_ahead=0.9,
        disturb=(), pose_noise=(0.01, 0.005), pose_latency=0.03, dt=0.002, control_period=0.05, stop_after_lap=True):
    cfg = cfg or Config()
    cfg.scoop_kind = scoop
    track = get_track()
    pusher = BallPusher(cfg, walls=track)
    path = pusher.path
    rng = np.random.default_rng(seed)
    ball_prm = ball_prm or BallParams()
    cam = Camera(cam_prm, seed=seed + 1)
    scoop_pts = make_scoop(scoop)
    foot = footprint_points(scoop_pts)

    xy0, yaw0, _, _ = path.at(start_s)
    car = Car((xy0[0], xy0[1], yaw0), car_prm)
    bxy, _, _, _ = path.at(start_s + ball_ahead)
    ball_p, ball_v = np.array(bxy, float), np.zeros(2)

    t = 0.0
    next_ctrl, next_cam = 0.0, 0.0
    det_queue = []
    pose_hist = []                      # (t, x, y, yaw, v)
    s_prev, progress, progress0 = None, 0.0, None
    touch_time, touch_run, loss_events, recov_ok, recov_fail, pending_loss = 0.0, 0.0, 0, 0, 0, None
    car_hits, in_hit, min_clear = 0, False, 9.0
    ball_hits, in_bhit = 0, False
    max_e, modes, lap_time, trace = 0.0, {}, None, []
    disturb = sorted(disturb)
    touching = False
    n_steps = int(t_max / dt)
    for k in range(n_steps):
        t = k * dt
        # ---------------- controller tick ----------------
        if t >= next_ctrl - 1e-9:
            next_ctrl += control_period
            tp = max(0.0, t - pose_latency)
            i = max(0, len(pose_hist) - int(round(pose_latency / dt)) - 1)
            _, x, y, yaw, vv = pose_hist[i] if pose_hist else (0, *car.pose, car.v)
            pm = (x + rng.normal(0, pose_noise[0]), y + rng.normal(0, pose_noise[0]), yaw + rng.normal(0, pose_noise[1]))
            vc, sc = pusher.update(t, pm, vv + rng.normal(0, 0.02), car.yaw_rate, t_pose=tp)
            car.command(t, vc, sc)
            modes[pusher.mode] = modes.get(pusher.mode, 0.0) + control_period
        # ---------------- camera ----------------
        if t >= next_cam - 1e-9:
            next_cam += 1.0 / cam.prm.rate
            z = cam.sense(car.pose, ball_p, ball_prm.radius, track)
            if z is not None:
                det_queue.append((t + cam.prm.latency, t, z))
        while det_queue and det_queue[0][0] <= t:
            _, t_img, z = det_queue.pop(0)
            pusher.observe(t_img, z)
        # ---------------- disturbance (a knock to the ball) ----------------
        while disturb and disturb[0][0] <= t:
            _, along, side = disturb.pop(0)
            s_b, _, psi, _, _ = path.project(ball_p)
            ball_v = ball_v + along * np.array([math.cos(psi), math.sin(psi)]) + side * np.array([-math.sin(psi), math.cos(psi)])
        # ---------------- physics ----------------
        car.step(t, dt)
        pose = car.pose
        pose_hist.append((t, *pose, car.v))
        if len(pose_hist) > 400:
            pose_hist.pop(0)
        sw = car_to_world(pose, scoop_pts)
        ball_p, ball_v, touching = ball_step(ball_p, ball_v, dt, ball_prm, sw, car_velocity_field(pose, car.v, car.yaw_rate), track)
        # ---------------- metrics ----------------
        if k % 5 == 0:
            clear = wall_clearance(track, pose, foot)
            min_clear = min(min_clear, clear)
            if clear < 0 and not in_hit:
                car_hits += 1
            in_hit = clear < 0
            wd = float(track.dist(ball_p[0], ball_p[1]))
            if wd < ball_prm.radius + 0.003 and not in_bhit and math.hypot(*ball_v) > 0.05:
                ball_hits += 1
            in_bhit = wd < ball_prm.radius + 0.003
            s_b, e_b, _, _, _ = path.project(ball_p)
            max_e = max(max_e, abs(e_b))
            if s_prev is None:
                progress0 = progress = s_b
            else:
                d = s_b - s_prev
                if d < -path.length / 2:
                    d += path.length
                elif d > path.length / 2:
                    d -= path.length
                progress += d
            s_prev = s_b
            if lap_time is None and progress - progress0 >= path.length:
                lap_time = t
                lap_touch = touch_time / max(t, 1e-9)
                if stop_after_lap:
                    break
            if lap_time is None:
                if touching:
                    touch_time += 5 * dt
                    if pending_loss is not None and touch_run == 0:
                        pass
                touch_run = touch_run + 5 * dt if not touching else 0.0
                if touch_run > 0.3 and pending_loss is None and t > 2.0:
                    pending_loss = t
                    loss_events += 1
                if touching and pending_loss is not None:
                    recov_ok += 1
                    pending_loss = None
                elif pending_loss is not None and t - pending_loss > 8.0:
                    recov_fail += 1
                    pending_loss = None
                    touch_run = -1e9
        if k % 10 == 0:
            trace.append((t, pose[0], pose[1], pose[2], car.v, ball_p[0], ball_p[1], float(touching), pusher.mode, progress, float(np.hypot(*(pusher.tracker.p - ball_p))) if pusher.tracker.has else -1.0))
    return dict(lap_time=lap_time, contact=(lap_touch if lap_time else touch_time / max(t, 1e-9)), loss_events=loss_events,
                recovered=recov_ok, failed_recoveries=recov_fail + (1 if pending_loss is not None else 0), car_hits=car_hits, ball_wall_hits=ball_hits,
                min_clearance=min_clear, max_ball_dev=max_e, modes=modes, trace=trace, t_end=t, reverse_count=0,
                progress=(progress - (progress0 or 0)) / path.length, ball_end_wall=float(track.dist(ball_p[0], ball_p[1])))


def report(name, r):
    lt = ('%.1f s' % r['lap_time']) if r['lap_time'] else 'NO LAP (reached %.0f%% by %.0f s)' % (100 * r['progress'], r['t_end'])
    print('%-28s lap %-22s contact %3.0f%% | losses %d (recovered %d, failed %d) | car hits %d, ball-wall hits %d | min wall gap %+.3f | max ball dev %.2f | reverse %d'
          % (name, lt, 100 * r['contact'], r['loss_events'], r['recovered'], r['failed_recoveries'], r['car_hits'], r['ball_wall_hits'],
             r['min_clearance'], r['max_ball_dev'], r['reverse_count']))


if __name__ == '__main__':
    import time
    t0 = time.time()
    r = run()
    report('default, v scoop', r)
    print('wall time %.1f s' % (time.time() - t0), r['modes'])
