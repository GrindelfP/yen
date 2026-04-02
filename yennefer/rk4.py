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
def _rk4_integrate(
    f: F,
    y0: NDArray[np.float64],
    dt: float,
    t_max: float,
    params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_steps = int(t_max / dt) + 2
    n_vars  = y0.shape[0]
    t_arr = np.empty(n_steps, dtype=np.float64)
    y_arr = np.empty((n_steps, n_vars), dtype=np.float64)
    t_arr[0] = 0.0
    y_arr[0] = y0

    idx = 1
    t   = 0.0
    y   = y0

    while t < t_max:
        h = dt
        if t + h > t_max:                    # last step
            h = t_max - t
        y = _rk4_step(f, t, y, h, params)
        t = t + h

        t_arr[idx] = t
        y_arr[idx] = y
        idx += 1

    return t_arr[:idx], y_arr[:idx]


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

    def solve(self, t_max: float, dt: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run the integration from ``t = 0`` to ``t_max``.

        Returns
        -------
        t : NDArray[float64], shape (N,)
            Time grid.
        y : NDArray[float64], shape (N, n_vars)
            Solution at each time point.
        """
        self._t, self._y = _rk4_integrate(
            self._f, self._y0, dt, t_max, self._params
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
