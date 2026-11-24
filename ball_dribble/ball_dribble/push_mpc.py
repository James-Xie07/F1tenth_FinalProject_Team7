"""Rear-axle tracking MPC for Frank's PUSH mode.

The midpoint bicycle model, quadratic cost and actuator limits are adapted
from main's frenet_mpc.py. Ball perception and recovery live in ball_control.
"""
import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MPCConfig:
    dt: float = 0.10
    horizon: int = 10
    wheelbase: float = 0.46
    max_speed: float = 0.50
    max_steer: float = 0.4189
    max_steer_rate: float = 2.0
    max_accel: float = 0.45
    max_brake: float = 0.55
    q_xy: float = 14.0
    q_yaw: float = 10.0
    q_speed: float = 6.0
    q_ball_lateral: float = 28.0
    r_accel: float = 0.10
    r_steer: float = 0.60
    rd_steer: float = 5.0


def push_reference(path, cfg, progress, yaw, speed_target):
    """Sample the loaded ball path and convert it to a rear-axle reference."""
    reference = np.empty((4, cfg.horizon + 1))
    for k in range(cfg.horizon + 1):
        ball_xy, ball_yaw, ball_curvature, speed = path.at(progress)
        rear_curvature = float(path.rear_curvature(ball_curvature))
        # The ball at the scoop follows the velocity of a point ahead of the
        # axle. In a steady turn its tangent leads the vehicle heading.
        turn = path.x_b * rear_curvature
        rear_yaw = ball_yaw - math.atan(turn)
        reference[:2, k] = ball_xy - path.x_b * np.array([math.cos(rear_yaw), math.sin(rear_yaw)])
        reference[2, k] = float(np.clip(min(speed, speed_target), 0.0, cfg.max_speed))
        reference[3, k] = rear_yaw
        progress += cfg.dt * reference[2, k] * math.hypot(1.0, turn)
    reference[3] = np.unwrap(reference[3])
    reference[3] += math.tau * round((yaw - reference[3, 0]) / math.tau)
    return reference


def bicycle_step(state, control, dt, wheelbase):
    x, y, v, yaw = state
    accel, steer = control
    mid_speed = v + 0.5 * dt * accel
    yaw_step = dt * mid_speed * math.tan(steer) / wheelbase
    mid_yaw = yaw + 0.5 * yaw_step
    return np.array([
        x + dt * mid_speed * math.cos(mid_yaw),
        y + dt * mid_speed * math.sin(mid_yaw),
        v + dt * accel,
        yaw + yaw_step,
    ])


def linearize(state, control, dt, wheelbase):
    h = 1e-5
    A = np.column_stack([
        (bicycle_step(state + dx, control, dt, wheelbase)
         - bicycle_step(state - dx, control, dt, wheelbase)) / (2 * h)
        for dx in h * np.eye(4)
    ])
    B = np.column_stack([
        (bicycle_step(state, control + du, dt, wheelbase)
         - bicycle_step(state, control - du, dt, wheelbase)) / (2 * h)
        for du in h * np.eye(2)
    ])
    C = bicycle_step(state, control, dt, wheelbase) - A @ state - B @ control
    return A, B, C


class TrackingMPC:
    def __init__(self, cfg):
        # The standalone Stanley baseline can run without solver dependencies.
        import cvxpy as cp

        if cp.OSQP not in cp.installed_solvers():
            raise RuntimeError('PUSH MPC requires the OSQP solver')
        self.cp = cp
        self.cfg = cfg
        n = cfg.horizon
        self.X = cp.Variable((4, n + 1))
        self.U = cp.Variable((2, n))
        self.initial = cp.Parameter(4)
        self.reference = cp.Parameter((4, n + 1))
        self.ball_normal = cp.Parameter((2, n + 1))
        self.ball_goal = cp.Parameter(n + 1)
        self.last_steer = cp.Parameter()
        self.A = [cp.Parameter((4, 4)) for _ in range(n)]
        self.B = [cp.Parameter((4, 2)) for _ in range(n)]
        self.C = [cp.Parameter(4) for _ in range(n)]
        accel, steer = self.U[0], self.U[1]
        delta_steer = steer[1:] - steer[:-1]
        steer_step = cfg.max_steer_rate * cfg.dt
        constraints = [
            self.X[:, 0] == self.initial,
            self.X[2] >= 0.0,
            self.X[2] <= cfg.max_speed,
            accel >= -cfg.max_brake,
            accel <= cfg.max_accel,
            cp.abs(steer) <= cfg.max_steer,
            cp.abs(steer[0] - self.last_steer) <= steer_step,
            cp.abs(delta_steer) <= steer_step,
        ]
        constraints.extend(
            self.X[:, k + 1] == self.A[k] @ self.X[:, k] + self.B[k] @ self.U[:, k] + self.C[k]
            for k in range(n)
        )
        error = self.X - self.reference
        lateral = cp.sum(cp.multiply(self.ball_normal, self.X[:2]), axis=0) - self.ball_goal
        cost = (
            cfg.q_xy * cp.sum_squares(error[:2])
            + cfg.q_yaw * cp.sum_squares(error[3])
            + cfg.q_speed * cp.sum_squares(error[2])
            + cfg.q_ball_lateral * cp.sum_squares(lateral)
            + cfg.r_accel * cp.sum_squares(accel)
            + cfg.r_steer * cp.sum_squares(steer)
            + cfg.rd_steer * cp.sum_squares(delta_steer)
        )
        self.problem = cp.Problem(cp.Minimize(cost), constraints)
        self.last_plan = None
        # Compile the QP while the node starts, before its first PUSH tick.
        self.solve(np.zeros(4), np.zeros((4, n + 1)), 0.0, 0.0)
        self.reset()

    def solve(self, state, reference, steer_previous, ball_lateral):
        cfg, cp = self.cfg, self.cp
        nominal = np.asarray(state, dtype=float).copy()
        if not np.isfinite(nominal).all() or not np.isfinite(reference).all() or not math.isfinite(ball_lateral):
            raise ValueError('nonfinite PUSH MPC input')
        nominal[2] = np.clip(nominal[2], 0.0, cfg.max_speed)
        self.initial.value = nominal
        self.reference.value = reference
        self.last_steer.value = float(np.clip(steer_previous, -cfg.max_steer, cfg.max_steer))
        headings = reference[3]
        normal = np.vstack((-np.sin(headings), np.cos(headings)))
        self.ball_normal.value = normal
        self.ball_goal.value = np.sum(normal * reference[:2], axis=0) + np.clip(ball_lateral, -0.06, 0.06)

        if self.last_plan is None:
            controls = np.vstack((np.zeros(cfg.horizon), np.full(cfg.horizon, self.last_steer.value)))
        else:
            controls = np.column_stack((self.last_plan[:, 1:], self.last_plan[:, -1]))
        for k, control in enumerate(controls.T):
            lin_state = nominal.copy()
            lin_state[2] = max(0.12, lin_state[2])
            self.A[k].value, self.B[k].value, self.C[k].value = linearize(lin_state, control, cfg.dt, cfg.wheelbase)
            nominal = bicycle_step(nominal, control, cfg.dt, cfg.wheelbase)

        # Match main's simulator setting. The numeric Jacobian can change the
        # sparsity of the QP, so rebuild OSQP's workspace every cycle.
        self.problem.solve(solver=cp.OSQP, warm_start=False, eps_abs=1e-3, eps_rel=1e-3, max_iter=2500)
        if self.problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or self.U.value is None:
            raise RuntimeError(f'PUSH MPC solve failed: {self.problem.status}')
        if not np.isfinite(self.U.value).all():
            raise RuntimeError('nonfinite PUSH MPC solution')
        self.last_plan = self.U.value.copy()
        velocity = float(np.clip(self.initial.value[2] + cfg.dt * self.last_plan[0, 0], 0.0, cfg.max_speed))
        steering = float(np.clip(self.last_plan[1, 0], -cfg.max_steer, cfg.max_steer))
        return velocity, steering

    def reset(self):
        self.last_plan = None
