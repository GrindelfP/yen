from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@njit
def _hermite_interp(t0, y0, f0, t1, y1, f1, t_target):
    h = t1 - t0
    theta = (t_target - t0) / h
    return (
        (1 - theta) * y0 + theta * y1 +
        theta * (theta - 1) * (
            (1 - 2*theta) * (y1 - y0) +
            h * (theta - 1) * f0 +
            h * theta * f1
        )
    )

@njit
def _rkf45_step(
        f: F,
        t: float,
        y: NDArray[np.float64],
        dt: float,
        params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    k1 = f(t, y, params)
    k2 = f(t + 0.25 * dt, y + dt * (0.25 * k1), params)
    k3 = f(t + 0.375 * dt, y + dt * (3.0 / 32.0 * k1 + 9.0 / 32.0 * k2), params)
    k4 = f(t + (12.0 / 13.0) * dt, y + dt * (1932.0 / 2197.0 * k1 - 7200.0 / 2197.0 * k2 + 7296.0 / 2197.0 * k3),
           params)
    k5 = f(t + dt, y + dt * (439.0 / 216.0 * k1 - 8.0 * k2 + 3680.0 / 513.0 * k3 - 845.0 / 4104.0 * k4), params)
    k6 = f(t + 0.5 * dt,
           y + dt * (-8.0 / 27.0 * k1 + 2.0 * k2 - 3544.0 / 2565.0 * k3 + 1859.0 / 4104.0 * k4 - 11.0 / 40.0 * k5),
           params)

    y5 = y + dt * (
                16.0 / 135.0 * k1 + 6656.0 / 12825.0 * k3 + 28561.0 / 56430.0 * k4 - 9.0 / 50.0 * k5 + 2.0 / 55.0 * k6)

    y4 = y + dt * (25.0 / 216.0 * k1 + 1408.0 / 2565.0 * k3 + 2197.0 / 4104.0 * k4 - 1.0 / 5.0 * k5)

    y_err = y5 - y4

    return y5, y_err


@njit
def _rkf45_integrate(
        f,
        y0,
        dt_initial,
        t_eval,
        params,
        atol,
        rtol,
        max_step,
):
    n_vars = y0.shape[0]
    n_out  = t_eval.shape[0]

    y_out = np.empty((n_out, n_vars), dtype=np.float64)
    y_out[0] = y0

    t = t_eval[0]
    y = y0.copy()
    dt = dt_initial
    t_max = t_eval[-1]

    f_cur = f(t, y, params)

    eval_idx = 1

    while eval_idx < n_out and t < t_max:
        if t + dt > t_max:
            dt = t_max - t

        y_new, y_err = _rkf45_step(f, t, y, dt, params)

        scale = atol + np.abs(y) * rtol
        error_norm = np.max(np.abs(y_err) / scale)

        if error_norm <= 1.0:
            t_new = t + dt
            f_new = f(t_new, y_new, params)

            while eval_idx < n_out and t_eval[eval_idx] <= t_new:
                y_out[eval_idx] = _hermite_interp(
                    t, y, f_cur, t_new, y_new, f_new, t_eval[eval_idx]
                )
                eval_idx += 1

            t = t_new
            y = y_new
            f_cur = f_new

        if error_norm == 0.0:
            factor = 5.0
        else:
            factor = 0.9 * (1.0 / error_norm) ** 0.25

        factor = min(5.0, max(0.1, factor))
        dt = dt * factor
        if dt > max_step:
            dt = max_step

    return t_eval, y_out


class RK45Solver:
    """Adaptive-step Runge-Kutta-Fehlberg 4(5) ODE solver.

    The right-hand side ``f`` must have the signature::

        f(t: float, y: NDArray[float64], params: NDArray[float64]) -> NDArray[float64]

    Both ``_rkf45_step`` and ``_rkf45_integrate`` are decorated with ``@njit``
    and are compiled on the first call. For maximum performance, decorate
    ``f`` with ``@njit`` as well before passing it to the solver.
    """

    def __init__(
            self,
            function: F,
            y0: NDArray[np.float64],
            params: NDArray[np.float64],
            atol: float = 1e-6,
            rtol: float = 1e-3,
            n_max_steps: float = np.inf,
    ) -> None:
        self._f = function
        self._y0 = np.asarray(y0, dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)

        self.rtol = rtol
        self.atol = atol
        self.max_step = n_max_steps

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(
            self,
            t_max: float | None = None,
            dt_initial: float = 1e-3,
            t_min: float = 0.0,
            t_eval: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if t_eval is None and t_max is not None:
            n_steps = round((t_max - t_min) / dt_initial)
            t_eval = np.linspace(t_min, t_max, n_steps + 1)
        elif t_eval is None:
            raise ValueError("Either provide t_max or t_eval!")
        t_eval = np.asarray(t_eval, dtype=np.float64)

        self._t, self._y = _rkf45_integrate(
            self._f, self._y0, dt_initial,
            t_eval, self._params, self.atol, self.rtol, self.max_step
        )
        return self._t, self._y

    @property
    def t(self) -> NDArray[np.float64]:
        """Time grid from the last ``solve()`` call."""
        if self._t is None:
            raise RuntimeError("Call solve() first.")
        return self._t

    @property
    def y(self) -> NDArray[np.float64]:
        """Solution array from the last ``solve()`` call, shape (N, n_vars)."""
        if self._y is None:
            raise RuntimeError("Call solve() first.")
        return self._y
