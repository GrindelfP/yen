"""
RADAU — Radau IIA implicit Runge-Kutta method (order 5).

Source: Hairer E., Wanner G.
        "Solving Ordinary Differential Equations II: Stiff and
        Differential-Algebraic Problems", 2nd ed., Springer, 1996.
        Section IV.8.

"""

from __future__ import annotations

import warnings
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import norm, solve

from .radau_constants import (
    C1, C2, C3,
    A11, A12, A13,
    A21, A22, A23,
    A31, A32, A33,
    B1, B2, B3,
    E1, E2, E3,
    STEP_EXPO, STEP_SAFE, STEP_MIN_FAC, STEP_MAX_FAC,
    NEWTON_MAXITER, JAC_REUSE_RATE,
)

F   = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
JAC = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


def _estimate_jacobian(
    f: F,
    t: float,
    y: NDArray[np.float64],
    params: NDArray[np.float64],
    f_now: NDArray[np.float64],
) -> NDArray[np.float64]:
    n   = y.shape[0]
    eps = np.sqrt(np.finfo(np.float64).eps)
    jac = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        y_eps    = y.copy()
        y_eps[i] += eps * max(1.0, abs(y[i]))
        jac[:, i] = (f(t, y_eps, params) - f_now) / (eps * max(1.0, abs(y[i])))
    return jac


def _build_system_jacobian(
    dfdy: NDArray[np.float64],
    h: float,
    n: int,
) -> NDArray[np.float64]:

    size    = 3 * n
    sys_jac = np.eye(size, dtype=np.float64)

    A = np.array([
        [A11, A12, A13],
        [A21, A22, A23],
        [A31, A32, A33],
    ])

    for i in range(3):
        for j in range(3):
            sys_jac[i*n:(i+1)*n, j*n:(j+1)*n] -= h * A[i, j] * dfdy

    return sys_jac


def _error_norm(
    err_vec: NDArray[np.float64],
    scale: NDArray[np.float64],
) -> float:
    return float(np.sqrt(np.mean((err_vec / scale) ** 2)))


def _radau_step(
    f: F,
    t: float,
    y: NDArray[np.float64],
    h: float,
    params: NDArray[np.float64],
    dfdy: NDArray[np.float64],
    k1_prev: NDArray[np.float64] | None,
    newton_tol: float,
) -> tuple[
    bool,                        # converged
    NDArray[np.float64],         # y_next
    NDArray[np.float64],         # Z1
    NDArray[np.float64],         # Z2
    NDArray[np.float64],         # Z3
    float,                       # newton_rate
]:

    n = y.shape[0]

    if k1_prev is not None:
        k1 = k1_prev.copy()
    else:
        k1 = f(t, y, params)
    k2 = k1.copy()
    k3 = k1.copy()

    sys_jac = _build_system_jacobian(dfdy, h, n)

    newton_rate = 1.0
    converged   = False

    for it in range(NEWTON_MAXITER):
        Z1 = h * (A11*k1 + A12*k2 + A13*k3)
        Z2 = h * (A21*k1 + A22*k2 + A23*k3)
        Z3 = h * (A31*k1 + A32*k2 + A33*k3)

        r1 = k1 - f(t + C1*h, y + Z1, params)
        r2 = k2 - f(t + C2*h, y + Z2, params)
        r3 = k3 - f(t + C3*h, y + Z3, params)

        res_vec = np.concatenate((r1, r2, r3))

        delta = solve(sys_jac, res_vec)

        delta_norm = norm(delta) / np.sqrt(3 * n)

        if it > 0:
            newton_rate = delta_norm / delta_norm_prev

            if newton_rate >= 1.0:
                break

        delta_norm_prev = delta_norm

        k1 -= delta[0:n]
        k2 -= delta[n:2*n]
        k3 -= delta[2*n:]

        if delta_norm < newton_tol:
            converged = True
            break

    Z1 = h * (A11*k1 + A12*k2 + A13*k3)
    Z2 = h * (A21*k1 + A22*k2 + A23*k3)
    Z3 = h * (A31*k1 + A32*k2 + A33*k3)

    y_next = y + Z3
    return converged, y_next, Z1, Z2, Z3, newton_rate


def _radau_integrate(
    f: F,
    y0: NDArray[np.float64],
    dt_init: float,
    t_eval: NDArray[np.float64],
    params: NDArray[np.float64],
    rtol: float,
    atol: float,
    n_max_steps: int,
    max_jac_reuse: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integration loop with dense output support."""
    n_vars = y0.shape[0]
    n_out  = t_eval.shape[0]
    y_out  = np.empty((n_out, n_vars), dtype=np.float64)

    t = t_eval[0]
    y = y0.copy()
    y_out[0] = y

    h       = dt_init
    h_abs   = abs(h)
    nstep   = 0
    t_idx   = 1  # Next t_eval point to fill
    t_end   = t_eval[-1]

    h_min = np.finfo(np.float64).eps * 100
    newton_tol = max(10.0 * np.finfo(np.float64).eps / rtol, min(0.03, rtol ** 0.5))

    f_now = f(t, y, params)
    dfdy  = _estimate_jacobian(f, t, y, params, f_now)
    jac_age = 0
    k1_prev = f_now

    h_abs_old      = None
    error_norm_old = None

    # Pre-calculate denominators for Lagrange interpolation
    d1 = C1 * (C1 - C2) * (C1 - C3)
    d2 = C2 * (C2 - C1) * (C2 - C3)
    d3 = C3 * (C3 - C1) * (C3 - C2)

    while t_idx < n_out and nstep < n_max_steps:
        if h_abs < h_min:
            warnings.warn(f"Step ({h_abs:.2e}) fell smaller than minimal.", RuntimeWarning)
            break

        if t + h_abs > t_end:
            h_abs = t_end - t

        h = h_abs
        converged, y_next, Z1, Z2, Z3, newton_rate = _radau_step(
            f, t, y, h, params, dfdy, k1_prev, newton_tol
        )

        if not converged:
            f_now = f(t, y, params); dfdy = _estimate_jacobian(f, t, y, params, f_now); jac_age = 0
            h_abs *= 0.5
            continue

        f0 = f(t, y, params)
        err_vec = h * f0 + E1*Z1 + E2*Z2 + E3*Z3
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        error_norm = _error_norm(err_vec, scale)

        if error_norm_old is None or h_abs_old is None or error_norm == 0.0:
            factor = error_norm ** (-STEP_EXPO)
        else:
            multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
            factor = min(1.0, multiplier) * error_norm ** (-STEP_EXPO)

        factor = STEP_SAFE * min(STEP_MAX_FAC, max(STEP_MIN_FAC, factor))
        h_new = h_abs * factor

        if error_norm <= 1.0:
            nstep += 1
            t_next = t + h

            # Perform interpolation for any points in t_eval caught by this step
            while t_idx < n_out and t_eval[t_idx] <= t_next:
                s = (t_eval[t_idx] - t) / h
                l1 = (s * (s - C2) * (s - C3)) / d1
                l2 = (s * (s - C1) * (s - C3)) / d2
                l3 = (s * (s - C1) * (s - C2)) / d3
                y_out[t_idx] = y + l1 * Z1 + l2 * Z2 + l3 * Z3
                t_idx += 1

            error_norm_old = error_norm; h_abs_old = h_abs; h_abs = h_new; jac_age += 1
            if jac_age >= max_jac_reuse or newton_rate > JAC_REUSE_RATE:
                f_now = f(t_next, y_next, params); dfdy = _estimate_jacobian(f, t_next, y_next, params, f_now)
                jac_age = 0; k1_prev = f_now
            else:
                k1_prev = f(t_next, y_next, params)
            t = t_next; y = y_next
        else:
            h_abs = h_new; error_norm_old = None; h_abs_old = None

    return t_eval[:t_idx], y_out[:t_idx]


def _radau_integrate_adaptive(
    f: F,
    y0: NDArray[np.float64],
    dt_init: float,
    t_max: float,
    params: NDArray[np.float64],
    rtol: float,
    atol: float,
    n_max_steps: int,
    max_jac_reuse: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Legacy adaptive-step integration loop."""
    n_vars = y0.shape[0]
    capacity = 1000
    t_arr = np.empty(capacity, dtype=np.float64)
    y_arr = np.empty((capacity, n_vars), dtype=np.float64)

    t = 0.0; y = y0.copy(); t_arr[0] = t; y_arr[0] = y
    h = dt_init; h_abs = abs(h); idx = 1; nstep = 0
    h_min = np.finfo(np.float64).eps * 100
    newton_tol = max(10.0 * np.finfo(np.float64).eps / rtol, min(0.03, rtol ** 0.5))

    f_now = f(t, y, params); dfdy = _estimate_jacobian(f, t, y, params, f_now); jac_age = 0; k1_prev = f_now
    h_abs_old = None; error_norm_old = None

    while t < t_max and nstep < n_max_steps:
        if h_abs < h_min: break
        if t + h_abs > t_max: h_abs = t_max - t
        h = h_abs
        converged, y_next, Z1, Z2, Z3, newton_rate = _radau_step(f, t, y, h, params, dfdy, k1_prev, newton_tol)

        if not converged:
            f_now = f(t, y, params); dfdy = _estimate_jacobian(f, t, y, params, f_now); jac_age = 0
            h_abs *= 0.5; continue

        f0 = f(t, y, params); err_vec = h * f0 + E1*Z1 + E2*Z2 + E3*Z3
        scale = atol + rtol * np.maximum(np.abs(y), np.abs(y_next))
        error_norm = _error_norm(err_vec, scale)

        if error_norm_old is None or h_abs_old is None or error_norm == 0.0:
            factor = error_norm ** (-STEP_EXPO)
        else:
            multiplier = h_abs / h_abs_old * (error_norm_old / error_norm) ** 0.25
            factor = min(1.0, multiplier) * error_norm ** (-STEP_EXPO)

        factor = STEP_SAFE * min(STEP_MAX_FAC, max(STEP_MIN_FAC, factor))
        h_new = h_abs * factor

        if error_norm <= 1.0:
            nstep += 1; error_norm_old = error_norm; h_abs_old = h_abs; h_abs = h_new; jac_age += 1
            if jac_age >= max_jac_reuse or newton_rate > JAC_REUSE_RATE:
                f_now = f(t + h, y_next, params); dfdy = _estimate_jacobian(f, t + h, y_next, params, f_now)
                jac_age = 0; k1_prev = f_now
            else:
                k1_prev = f(t + h, y_next, params)
            t = t + h; y = y_next
            if idx >= capacity:
                new_cap = capacity * 2; new_t = np.empty(new_cap); new_y = np.empty((new_cap, n_vars))
                new_t[:capacity] = t_arr; new_y[:capacity] = y_arr; t_arr = new_t; y_arr = new_y; capacity = new_cap
            t_arr[idx] = t; y_arr[idx] = y; idx += 1
        else:
            h_abs = h_new; error_norm_old = None; h_abs_old = None

    return t_arr[:idx], y_arr[:idx]


class RADAUSolver:

    def __init__(
        self,
        function: F,
        y0: NDArray[np.float64],
        params: NDArray[np.float64],
        rtol: float = 1e-6,
        atol: float = 1e-6,
        n_max_steps: int = 100_000,
        max_jac_reuse: int = 20,
    ) -> None:
        self._f             = function
        self._y0            = np.asarray(y0,     dtype=np.float64)
        self._params        = np.asarray(params, dtype=np.float64)
        self.rtol           = float(rtol)
        self.atol           = float(atol)
        self.n_max_steps    = int(n_max_steps)
        self.max_jac_reuse  = int(max_jac_reuse)

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(
        self,
        t_max: float | None = None,
        dt_init: float | None = None,
        t_min: float = 0.0,
        t_eval: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate from ``t_min`` to ``t_max`` or on ``t_eval`` grid."""

        if t_eval is not None:
            # Dense-output mode
            t_eval = np.asarray(t_eval, dtype=np.float64)
            if dt_init is None:
                dt_init = 1e-3
            self._t, self._y = _radau_integrate(
                self._f, self._y0, dt_init, t_eval, self._params,
                self.rtol, self.atol, self.n_max_steps, self.max_jac_reuse,
            )
        elif t_max is not None and dt_init is not None:
            # Legacy adaptive-grid mode
            self._t, self._y = _radau_integrate_adaptive(
                self._f, self._y0, dt_init, float(t_max), self._params,
                self.rtol, self.atol, self.n_max_steps, self.max_jac_reuse,
            )
        else:
            raise ValueError(
                "Provide either:\n"
                "  solver.solve(t_max, dt_init)                # adaptive-grid mode\n"
                "  solver.solve(t_eval=..., dt_init=...)       # dense-output mode"
            )

        return self._t, self._y

    @property
    def t(self) -> NDArray[np.float64]:
        if self._t is None:
            raise RuntimeError("Call solve() first.")
        return self._t

    @property
    def y(self) -> NDArray[np.float64]:
        if self._y is None:
            raise RuntimeError("Call solve() first.")
        return self._y
