from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


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
        f: F,
        y0: NDArray[np.float64],
        dt_initial: float,
        t_max: float,
        params: NDArray[np.float64],
        atol: float,
        rtol: float,
        max_step: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_vars = y0.shape[0]

    capacity = 1000
    t_arr = np.empty(capacity, dtype=np.float64)
    y_arr = np.empty((capacity, n_vars), dtype=np.float64)

    t_arr[0] = 0.0
    y_arr[0] = y0

    t = 0.0
    y = y0
    dt = dt_initial
    step_idx = 1

    while t < t_max:
        if t + dt > t_max:
            dt = t_max - t

        y_new, y_err = _rkf45_step(f, t, y, dt, params)

        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        error_norm = np.max(np.abs(y_err) / scale)

        if error_norm <= 1.0:
            t = t + dt
            y = y_new

            if step_idx >= capacity:
                new_capacity = capacity * 2
                new_t_arr = np.empty(new_capacity, dtype=np.float64)
                new_y_arr = np.empty((new_capacity, n_vars), dtype=np.float64)
                new_t_arr[:capacity] = t_arr
                new_y_arr[:capacity] = y_arr
                t_arr = new_t_arr
                y_arr = new_y_arr
                capacity = new_capacity

            t_arr[step_idx] = t
            y_arr[step_idx] = y
            step_idx += 1

        if error_norm == 0.0:
            factor = 5.0
        else:
            factor = 0.9 * (1.0 / error_norm) ** 0.25

        factor = min(5.0, max(0.1, factor))
        dt = dt * factor

        if dt > max_step:
            dt = max_step

    return t_arr[:step_idx], y_arr[:step_idx]


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
            max_step: float = np.inf,
    ) -> None:
        self._f = function
        self._y0 = np.asarray(y0, dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)

        self.rtol = rtol
        self.atol = atol
        self.max_step = max_step

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(
            self,
            t_max: float,
            dt_initial: float = 1e-3,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run the adaptive integration from ``t = 0`` to ``t_max``.

        Parameters
        ----------
        t_max : float
            Maximum integration time.
        dt_initial : float, optional
            Initial step size guess, by default 1e-3.
        atol : float, optional
            Absolute tolerance for error control, by default 1e-6.
        rtol : float, optional
            Relative tolerance for error control, by default 1e-3.
        max_step : float, optional
            Maximum allowed step size, by default infinity.

        Returns
        -------
        t : NDArray[float64], shape (N,)
            Time grid (adaptively determined).
        y : NDArray[float64], shape (N, n_vars)
            Solution at each time point.
        """
        self._t, self._y = _rkf45_integrate(
            self._f, self._y0, dt_initial, t_max, self._params, self.atol, self.rtol, self.max_step
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
