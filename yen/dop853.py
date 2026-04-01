from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from numba import njit
from .dop853_constants import N_STAGES, A, B, C, E3, E5

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@njit
def _dop853_step(
        f: F,
        t: float,
        y: NDArray[np.float64],
        dt: float,
        params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    n_vars = y.shape[0]

    K = np.empty((N_STAGES, n_vars), dtype=np.float64)

    for i in range(N_STAGES):
        dy = np.zeros(n_vars, dtype=np.float64)
        for j in range(i):
            if A[i, j] != 0.0:
                dy += A[i, j] * K[j]
        K[i] = f(t + C[i] * dt, y + dt * dy, params)

    y_new = y + dt * (B @ K)

    err5 = dt * (E5 @ K)
    err3 = dt * (E3 @ K)

    return y_new, err5, err3


@njit
def _dop853_integrate(
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

        y_new, err5, err3 = _dop853_step(f, t, y, dt, params)

        scale = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol
        err5_scaled = err5 / scale
        err3_scaled = err3 / scale

        err5_norm_2 = np.sum(err5_scaled ** 2)
        err3_norm_2 = np.sum(err3_scaled ** 2)
        denom = err5_norm_2 + 0.01 * err3_norm_2

        if denom > 0:
            error_norm = (err5_norm_2 / np.sqrt(denom)) / np.sqrt(float(n_vars))
        else:
            error_norm = 0.0

        if error_norm <= 1.0:
            # Шаг успешен
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
            factor = 10.0
        else:
            factor = 0.9 * (1.0 / error_norm) ** 0.125

        factor = min(10.0, max(0.2, factor))
        dt = dt * factor

        if dt > max_step:
            dt = max_step

    return t_arr[:step_idx], y_arr[:step_idx]


class DOP853Solver:
    """Explicit 8th-order Runge-Kutta solver (Dormand-Prince 8(5,3)).

    The right-hand side ``f`` must have the signature::
        f(t: float, y: NDArray[float64], params: NDArray[float64]) -> NDArray[float64]

    Best used with strict tolerances (e.g., rtol < 1e-6, atol < 1e-8).
    """

    def __init__(
            self,
            function: F,
            y0: NDArray[np.float64],
            params: NDArray[np.float64],
    ) -> None:
        self._f = function
        self._y0 = np.asarray(y0, dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(
            self,
            t_max: float,
            dt_initial: float = 1e-3,
            atol: float = 1e-8,
            rtol: float = 1e-6,
            max_step: float = np.inf,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._t, self._y = _dop853_integrate(
            self._f, self._y0, dt_initial, t_max, self._params, atol, rtol, max_step
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
