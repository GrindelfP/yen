from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.typing import NDArray
from numba import njit
from .dop853_constants import N_STAGES, A, B, BHH, C, E3, E5

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@njit
def _dop853_step(
        f: F,
        t: float,
        y: NDArray[np.float64],
        dt: float,
        params: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Compute one DOP853 step.

    Returns
    -------
    y_new : ndarray
        8th-order solution at t + dt.
    err5 : ndarray
        Error estimate vector (difference: 8th-order minus embedded 5th-order).
    err3 : ndarray
        Secondary error estimate vector (3rd-order residual).
    """
    n_vars = y.shape[0]
    K = np.empty((N_STAGES, n_vars), dtype=np.float64)

    for i in range(N_STAGES):
        dy = np.zeros(n_vars, dtype=np.float64)
        for j in range(i):
            if A[i, j] != 0.0:
                dy += A[i, j] * K[j]
        K[i] = f(t + C[i] * dt, y + dt * dy, params)

    y_new = y.copy()
    for i in range(N_STAGES):
        if B[i] != 0.0:
            y_new += dt * B[i] * K[i]

    # ── Error estimation (Hairer/Wanner DOP853 formula) ──────────────────────
    # Primary error:  err5 = h * sum(E5[i] * K[i])   where E5 = B - BHH
    # Secondary error: err3 = h * sum(E3[i] * K[i])
    #
    # Combined norm per component j:
    #   sk_j  = atol + rtol * max(|y_j|, |y_new_j|)
    #   err_j = err5_j / (sk_j * stden_j)
    # where  stden = h * sum(BHH[i] * K[i])  (the embedded 5th-order increment)
    #
    # error_norm = sqrt( mean(err_j^2) )
    # with denom = err5_norm^2 + 0.01 * err3_norm^2 used as a safety denominator.
    #
    # NOTE: stden is returned separately so that _dop853_integrate can compute
    # the combined norm without recomputing BHH@K.

    err5 = np.zeros(n_vars, dtype=np.float64)
    err3 = np.zeros(n_vars, dtype=np.float64)
    stden = np.zeros(n_vars, dtype=np.float64)

    for i in range(N_STAGES):
        if E5[i] != 0.0:
            err5 += E5[i] * K[i]
        if E3[i] != 0.0:
            err3 += E3[i] * K[i]
        if BHH[i] != 0.0:
            stden += BHH[i] * K[i]

    err5 *= dt
    err3 *= dt
    stden *= dt

    return y_new, err5, err3, stden


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
    y = y0.copy()
    dt = dt_initial
    step_idx = 1

    while t < t_max:
        if t + dt > t_max:
            dt = t_max - t

        y_new, err5, err3, stden = _dop853_step(f, t, y, dt, params)

        # ── Combined DOP853 error norm (Hairer eq. III.5.7) ──────────────────
        # sk = atol + rtol * max(|y|, |y_new|)
        # For each component j:
        #   e5j = err5[j] / (sk[j] * |stden[j]|)  if stden[j] != 0
        #         err5[j] / sk[j]                  otherwise
        # err5_norm^2 = mean(e5j^2)
        # err3_norm^2 = mean(e3j^2)   with e3j = err3[j] / (sk[j] * |stden[j]|)
        # error_norm  = sqrt(err5_norm^2 / (err5_norm^2 + 0.01 * err3_norm^2))
        sk = atol + np.maximum(np.abs(y), np.abs(y_new)) * rtol

        err5_norm_sq = 0.0
        err3_norm_sq = 0.0
        for j in range(n_vars):
            denom_j = sk[j] * max(abs(stden[j]), 1e-300)
            e5j = err5[j] / denom_j
            e3j = err3[j] / denom_j
            err5_norm_sq += e5j * e5j
            err3_norm_sq += e3j * e3j

        err5_norm_sq /= float(n_vars)
        err3_norm_sq /= float(n_vars)

        combined_denom = err5_norm_sq + 0.01 * err3_norm_sq
        if combined_denom > 0.0:
            error_norm = np.sqrt(err5_norm_sq / combined_denom)
        else:
            error_norm = 0.0

        if error_norm <= 1.0:
            # Step accepted
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

        # Step-size control: PI controller exponent 1/8 for 8th-order method
        if error_norm == 0.0:
            factor = 10.0
        else:
            factor = min(10.0, max(0.2, 0.9 * (1.0 / error_norm) ** 0.125))

        dt = min(dt * factor, max_step)

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
