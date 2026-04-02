"""
DOP853 — Dormand-Prince 8(5,3) explicit Runge-Kutta method.

Source: Hairer E., Nørsett S.P., Wanner G.
        "Solving Ordinary Differential Equations I", 2nd ed., Springer, 1993.

Based on the DOP853 implementation by Ernst Hairer and Jacob Williams.
Original Fortran source: https://github.com/jacobwilliams/dop853
See LICENSE file for full license text.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numba import njit

from .dop853_constants import *

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]

@njit
def _dop853_integrate(
        f: F,
        y0: NDArray[np.float64],
        dt_init: float,
        t_max: float,
        params: NDArray[np.float64],
        rtol: float,
        atol: float,
        n_max_steps: int,
        fac1: float = 0.333,
        fac2: float = 6.0,
        safe: float = 0.9,
        beta: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    n_vars = y0.shape[0]

    # # Pre-allocate arrays
    # t_arr = np.empty(n_max_steps + 1, dtype=np.float64)
    # y_arr = np.empty((n_max_steps + 1, n_vars), dtype=np.float64)

    # Pre-allocate arrays
    capacity = 1000
    t_arr = np.empty(capacity, dtype=np.float64)
    y_arr = np.empty((capacity, n_vars), dtype=np.float64)

    t = 0.0
    y = np.copy(y0)

    t_arr[0] = t
    y_arr[0] = y

    h = dt_init
    idx = 1
    nstep = 0

    k1 = f(t, y, params)

    facold = 1.0e-4
    expo1 = 1.0 / 8.0 - beta * 0.2
    facc1 = 1.0 / fac1
    facc2 = 1.0 / fac2

    posneg = 1.0 if t_max > 0.0 else -1.0
    reject = False
    last = False

    while t < t_max and nstep < n_max_steps:

        if np.abs(h) < np.abs(t) * np.finfo(np.float64).eps:
            # Step size too small
            break

        if (t + 1.01 * h - t_max) * posneg > 0.0:
            h = t_max - t
            last = True

        nstep += 1

        # --- 12 Stages computation ---
        y_tmp = y + h * A21 * k1
        k2 = f(t + C2 * h, y_tmp, params)

        y_tmp = y + h * (A31 * k1 + A32 * k2)
        k3 = f(t + C3 * h, y_tmp, params)

        y_tmp = y + h * (A41 * k1 + A43 * k3)
        k4 = f(t + C4 * h, y_tmp, params)

        y_tmp = y + h * (A51 * k1 + A53 * k3 + A54 * k4)
        k5 = f(t + C5 * h, y_tmp, params)

        y_tmp = y + h * (A61 * k1 + A64 * k4 + A65 * k5)
        k6 = f(t + C6 * h, y_tmp, params)

        y_tmp = y + h * (A71 * k1 + A74 * k4 + A75 * k5 + A76 * k6)
        k7 = f(t + C7 * h, y_tmp, params)

        y_tmp = y + h * (A81 * k1 + A84 * k4 + A85 * k5 + A86 * k6 + A87 * k7)
        k8 = f(t + C8 * h, y_tmp, params)

        y_tmp = y + h * (A91 * k1 + A94 * k4 + A95 * k5 + A96 * k6 + A97 * k7 + A98 * k8)
        k9 = f(t + C9 * h, y_tmp, params)

        y_tmp = y + h * (A101 * k1 + A104 * k4 + A105 * k5 + A106 * k6 + A107 * k7 + A108 * k8 + A109 * k9)
        k10 = f(t + C10 * h, y_tmp, params)

        y_tmp = y + h * (
                    A111 * k1 + A114 * k4 + A115 * k5 + A116 * k6 + A117 * k7 + A118 * k8 + A119 * k9 + A1110 * k10)
        k11 = f(t + C11 * h, y_tmp, params)

        t_next = t + h
        y_tmp = y + h * (
                    A121 * k1 + A124 * k4 + A125 * k5 + A126 * k6 + A127 * k7 + A128 * k8 + A129 * k9 + A1210 * k10 + A1211 * k11)
        k12 = f(t_next, y_tmp, params)

        k_final = B1 * k1 + B6 * k6 + B7 * k7 + B8 * k8 + B9 * k9 + B10 * k10 + B11 * k11 + B12 * k12
        y_next = y + h * k_final

        # --- Error estimation ---
        err = 0.0
        err2 = 0.0

        for i in range(n_vars):
            sk = atol + rtol * max(abs(y[i]), abs(y_next[i]))
            erri = k_final[i] - BHH1 * k1[i] - BHH2 * k9[i] - BHH3 * k12[i]
            err2 += (erri / sk) ** 2

            erri_2 = (ER1 * k1[i] + ER6 * k6[i] + ER7 * k7[i] + ER8 * k8[i] +
                      ER9 * k9[i] + ER10 * k10[i] + ER11 * k11[i] + ER12 * k12[i])
            err += (erri_2 / sk) ** 2

        deno = err + 0.01 * err2
        if deno <= 0.0:
            deno = 1.0

        err = abs(h) * err * np.sqrt(1.0 / (n_vars * deno))

        # --- Computation of hnew ---
        fac11 = err ** expo1
        fac = fac11 / (facold ** beta)
        fac = max(facc2, min(facc1, fac / safe))
        h_new = h / fac

        if err <= 1.0:

            # --- ALLOCATION ---
            if idx >= capacity:
                new_capacity = capacity * 2

                new_t_arr = np.empty(new_capacity, dtype=np.float64)
                new_t_arr[:capacity] = t_arr
                t_arr = new_t_arr

                new_y_arr = np.empty((new_capacity, n_vars), dtype=np.float64)
                new_y_arr[:capacity, :] = y_arr
                y_arr = new_y_arr

                capacity = new_capacity
            # -------------------------------------

            # Step is accepted
            facold = max(err, 1.0e-4)
            k_new_eval = f(t_next, y_next, params)  # New k1 mapping

            k1 = k_new_eval
            y = y_next
            t = t_next

            t_arr[idx] = t
            y_arr[idx] = y
            idx += 1

            if last:
                h = h_new
                break

            if reject:
                h_new = posneg * min(abs(h_new), abs(h))

            reject = False
        else:
            # Step is rejected
            h_new = h / min(facc1, fac11 / safe)
            reject = True
            last = False

        h = h_new

    return t_arr[:idx], y_arr[:idx]


class DOP853Solver:
    """Adaptive-step Runge-Kutta 8th-order ODE solver (Dormand & Prince 8(5,3)).

    The right-hand side ``f`` must have the signature::

        f(t: float, y: NDArray[float64], params: NDArray[float64]) -> NDArray[float64]

    The core method ``_dop853_integrate`` is decorated with ``@njit``
    and is compiled on the first call. For maximum performance, decorate
    ``f`` with ``@njit`` as well before passing it to the solver.
    """

    def __init__(
            self,
            function: F,
            y0: NDArray[np.float64],
            params: NDArray[np.float64],
            rtol: float = 1e-9,
            atol: float = 1e-9,
            n_max_steps: int = 10000,
    ) -> None:
        self._f = function
        self._y0 = np.asarray(y0, dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)

        self.rtol = rtol
        self.atol = atol
        self.n_max_steps = n_max_steps

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(self, t_max: float, dt_init: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Run the integration from ``t = 0`` to ``t_max``.

        ``dt`` serves as the initial step size guess (`h_init`) for the adaptive algorithm.

        Returns
        -------
        t : NDArray[float64], shape (N_accepted_steps,)
            Time grid at accepted adaptive steps.
        y : NDArray[float64], shape (N_accepted_steps, n_vars)
            Solution at each accepted time point.
        """
        self._t, self._y = _dop853_integrate(
            self._f,
            self._y0,
            dt_init,
            t_max,
            self._params,
            self.rtol,
            self.atol,
            self.n_max_steps
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
        """Solution array from the last ``solve()`` call."""
        if self._y is None:
            raise RuntimeError("Call solve() first.")
        return self._y
