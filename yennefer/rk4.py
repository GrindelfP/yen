from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit


F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@njit
def _rk4_step(
    f: F,
    t: float,
    y: NDArray[np.float64],
    dt: float,
    params: NDArray[np.float64],
) -> NDArray[np.float64]:
    k1 = f(t,             y,              params)
    k2 = f(t + 0.5 * dt,  y + 0.5*dt*k1, params)
    k3 = f(t + 0.5 * dt,  y + 0.5*dt*k2, params)
    k4 = f(t + dt,        y +      dt*k3, params)
    return y + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


@njit
def _rk4_integrate(f, y0, t_eval, params):
    n_vars = y0.shape[0]
    n_out = t_eval.shape[0]

    y_out = np.empty((n_out, n_vars), dtype=np.float64)
    y_out[0] = y0

    t = t_eval[0]
    y = y0.copy()

    eval_idx = 1

    while eval_idx < n_out:
        t_next_eval = t_eval[eval_idx]
        h = t_next_eval - t

        y_next = _rk4_step(f, t, y, h, params)

        y_out[eval_idx] = y_next

        t = t_next_eval
        y = y_next
        eval_idx += 1

    return t_eval, y_out


class RK4Solver:
    """Fixed-step Runge-Kutta 4th-order ODE solver.

    The right-hand side ``f`` must have the signature::

        f(t: float, y: NDArray[float64], params: NDArray[float64]) -> NDArray[float64]

    Both ``_rk4_step`` and ``_rk4_integrate`` are decorated with ``@njit``
    and are compiled on the first call.  For maximum performance, decorate
    ``f`` with ``@njit`` as well before passing it to the solver.
    """

    def __init__(
        self,
        function: F,
        y0:       NDArray[np.float64],
        params:   NDArray[np.float64],
    ) -> None:
        self._f      = function
        self._y0     = np.asarray(y0,     dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(
            self,
            t_max: float | None = None,
            dt: float | None = None,
            t_min: float | None = 0.0,
            t_eval: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if t_eval is None and t_min is not None and t_max is not None and dt is not None:
            n_steps = round((t_max - t_min) / dt)
            t_eval = np.linspace(t_min, t_max, n_steps + 1)
        elif t_eval is None and t_min is None and t_max is None and dt is None:
            raise AttributeError("Either provide t_min, t_max, t_step or time space t_eval!")
        t_eval = np.asarray(t_eval, dtype=np.float64)
        self._t, self._y = _rk4_integrate(self._f, self._y0, t_eval, self._params)
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
