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


# ==============================================================================
# Dense output (contd8)
# ==============================================================================
# The 7th-order interpolation polynomial from Hairer §II.6.
# cont[0..7] are the 8 coefficient vectors, each of length n_vars.
# Given s = (t_interp - t_old) / h  and  s1 = 1 - s:
#
#   conpar = cont4 + s*(cont5 + s1*(cont6 + s*cont7))
#   y(t)   = cont0 + s*(cont1 + s1*(cont2 + s*(cont3 + s1*conpar)))
#
# This matches the Fortran `contd8` function exactly.

@njit
def _contd8(s: float, cont: NDArray[np.float64], n_vars: int) -> NDArray[np.float64]:
    """Evaluate the dense output polynomial at normalised position s ∈ [0, 1]."""
    s1 = 1.0 - s
    conpar = (cont[4] + s * (cont[5] + s1 * (cont[6] + s * cont[7])))
    return cont[0] + s * (cont[1] + s1 * (cont[2] + s * (cont[3] + s1 * conpar)))


@njit
def _build_cont(
        h: float,
        y: NDArray[np.float64],
        y_next: NDArray[np.float64],
        k1: NDArray[np.float64],
        k6: NDArray[np.float64],
        k7: NDArray[np.float64],
        k8: NDArray[np.float64],
        k9: NDArray[np.float64],
        k10: NDArray[np.float64],
        k11: NDArray[np.float64],   # called k2 in Fortran after stage-11
        k12: NDArray[np.float64],   # called k3 in Fortran after stage-12
        k_new1: NDArray[np.float64], # f(t_next, y_next) — the FSAL k1 of next step, called k4 in Fortran
        k14: NDArray[np.float64],
        k15: NDArray[np.float64],
        k16: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Build the 8-slot cont array for the dense output polynomial.

    Follows Hairer / Williams dp86co lines 841-883 exactly.
    Variable names in comments refer to the Fortran k-labels.
    """
    n = y.shape[0]
    # cont is stored as a (8, n_vars) array for convenience
    cont = np.empty((8, n), dtype=np.float64)

    for i in range(n):
        ydiff = y_next[i] - y[i]
        bspl  = h * k1[i] - ydiff

        # cont[0] = y_old
        cont[0, i] = y[i]
        # cont[1] = ydiff
        cont[1, i] = ydiff
        # cont[2] = bspl
        cont[2, i] = bspl
        # cont[3] = ydiff - h*f(t_next,y_next) - bspl   (k4 in Fortran = k_new1)
        cont[3, i] = ydiff - h * k_new1[i] - bspl

        # cont[4..7]: first pass (without the extra stages)
        cont[4, i] = (D41*k1[i] + D46*k6[i] + D47*k7[i] + D48*k8[i]
                    + D49*k9[i] + D410*k10[i] + D411*k11[i] + D412*k12[i])
        cont[5, i] = (D51*k1[i] + D56*k6[i] + D57*k7[i] + D58*k8[i]
                    + D59*k9[i] + D510*k10[i] + D511*k11[i] + D512*k12[i])
        cont[6, i] = (D61*k1[i] + D66*k6[i] + D67*k7[i] + D68*k8[i]
                    + D69*k9[i] + D610*k10[i] + D611*k11[i] + D612*k12[i])
        cont[7, i] = (D71*k1[i] + D76*k6[i] + D77*k7[i] + D78*k8[i]
                    + D79*k9[i] + D710*k10[i] + D711*k11[i] + D712*k12[i])

    # cont[4..7]: add contributions from the three extra dense-output stages
    # and multiply by h  (Fortran lines 876-883)
    for i in range(n):
        cont[4, i] = h * (cont[4, i] + D413*k_new1[i] + D414*k14[i] + D415*k15[i] + D416*k16[i])
        cont[5, i] = h * (cont[5, i] + D513*k_new1[i] + D514*k14[i] + D515*k15[i] + D516*k16[i])
        cont[6, i] = h * (cont[6, i] + D613*k_new1[i] + D614*k14[i] + D615*k15[i] + D616*k16[i])
        cont[7, i] = h * (cont[7, i] + D713*k_new1[i] + D714*k14[i] + D715*k15[i] + D716*k16[i])

    return cont


# ==============================================================================
# Core integrator — adaptive-grid mode (legacy)
# ==============================================================================

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
            break

        if (t + 1.01 * h - t_max) * posneg > 0.0:
            h = t_max - t
            last = True

        nstep += 1

        # --- 12 Stages ---
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

        y_tmp = y + h * (A111 * k1 + A114 * k4 + A115 * k5 + A116 * k6 + A117 * k7 + A118 * k8 + A119 * k9 + A1110 * k10)
        k11 = f(t + C11 * h, y_tmp, params)

        t_next = t + h
        y_tmp = y + h * (A121 * k1 + A124 * k4 + A125 * k5 + A126 * k6 + A127 * k7 + A128 * k8 + A129 * k9 + A1210 * k10 + A1211 * k11)
        k12 = f(t_next, y_tmp, params)

        k_final = B1 * k1 + B6 * k6 + B7 * k7 + B8 * k8 + B9 * k9 + B10 * k10 + B11 * k11 + B12 * k12
        y_next = y + h * k_final

        # --- Error estimation ---
        err = 0.0
        err2 = 0.0

        for i in range(n_vars):
            sk = atol + rtol * max(abs(y[i]), abs(y_next[i]))
            erri_3 = k_final[i] - BHH1 * k1[i] - BHH2 * k9[i] - BHH3 * k12[i]
            err2 += (erri_3 / sk) ** 2
            erri_8 = (ER1 * k1[i] + ER6 * k6[i] + ER7 * k7[i] + ER8 * k8[i] +
                      ER9 * k9[i] + ER10 * k10[i] + ER11 * k11[i] + ER12 * k12[i])
            err += (erri_8 / sk) ** 2

        err *= h ** 2
        err2 *= h ** 2

        deno = n_vars * (err + 0.01 * err2)
        if deno <= 0.0:
            err = 0.0
        else:
            err = err / np.sqrt(deno)

        # --- Step size control ---
        fac11 = err ** expo1
        fac = fac11 / (facold ** beta)
        fac = max(facc2, min(facc1, fac / safe))
        h_new = h / fac

        if err <= 1.0:

            # --- dynamic array growth ---
            if idx >= capacity:
                new_capacity = capacity * 2
                new_t_arr = np.empty(new_capacity, dtype=np.float64)
                new_t_arr[:capacity] = t_arr
                t_arr = new_t_arr
                new_y_arr = np.empty((new_capacity, n_vars), dtype=np.float64)
                new_y_arr[:capacity, :] = y_arr
                y_arr = new_y_arr
                capacity = new_capacity

            facold = max(err, 1.0e-4)
            k1 = f(t_next, y_next, params)   # FSAL
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
            h_new = h / min(facc1, fac11 / safe)
            reject = True
            last = False

        h = h_new

    return t_arr[:idx], y_arr[:idx]


# ==============================================================================
# Core integrator — t_eval (dense output) mode
# ==============================================================================

@njit
def _dop853_integrate_t_eval(
        f: F,
        y0: NDArray[np.float64],
        dt_init: float,
        t_eval: NDArray[np.float64],
        params: NDArray[np.float64],
        rtol: float,
        atol: float,
        n_max_steps: int,
        fac1: float = 0.333,
        fac2: float = 6.0,
        safe: float = 0.9,
        beta: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Integrate and evaluate at the prescribed t_eval points via dense output.

    Uses the full 7th-order DOP853 interpolation polynomial (`contd8`) to
    produce accurate values at every t_eval point without storing the entire
    adaptive grid.
    """
    n_vars = y0.shape[0]
    n_out  = t_eval.shape[0]
    t_min  = t_eval[0]
    t_max  = t_eval[-1]

    y_out = np.empty((n_out, n_vars), dtype=np.float64)
    y_out[0] = y0

    t = t_min
    y = np.copy(y0)
    h = dt_init
    nstep = 0
    eval_idx = 1   # next t_eval index to fill

    k1 = f(t, y, params)

    facold = 1.0e-4
    expo1  = 1.0 / 8.0 - beta * 0.2
    facc1  = 1.0 / fac1
    facc2  = 1.0 / fac2

    posneg = 1.0 if t_max > t_min else -1.0
    reject = False
    last   = False

    while eval_idx < n_out and t < t_max and nstep < n_max_steps:

        if np.abs(h) < np.abs(t) * np.finfo(np.float64).eps:
            break

        if (t + 1.01 * h - t_max) * posneg > 0.0:
            h = t_max - t
            last = True

        nstep += 1

        # ---- 12 main stages -----------------------------------------------
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

        y_tmp = y + h * (A111 * k1 + A114 * k4 + A115 * k5 + A116 * k6 + A117 * k7 + A118 * k8 + A119 * k9 + A1110 * k10)
        k11 = f(t + C11 * h, y_tmp, params)

        t_next = t + h
        y_tmp = y + h * (A121 * k1 + A124 * k4 + A125 * k5 + A126 * k6 + A127 * k7 + A128 * k8 + A129 * k9 + A1210 * k10 + A1211 * k11)
        k12 = f(t_next, y_tmp, params)

        k_final = B1 * k1 + B6 * k6 + B7 * k7 + B8 * k8 + B9 * k9 + B10 * k10 + B11 * k11 + B12 * k12
        y_next = y + h * k_final

        # ---- Error estimation ---------------------------------------------
        err = 0.0
        err2 = 0.0

        for i in range(n_vars):
            sk = atol + rtol * max(abs(y[i]), abs(y_next[i]))
            erri_3 = k_final[i] - BHH1 * k1[i] - BHH2 * k9[i] - BHH3 * k12[i]
            err2 += (erri_3 / sk) ** 2
            erri_8 = (ER1 * k1[i] + ER6 * k6[i] + ER7 * k7[i] + ER8 * k8[i] +
                      ER9 * k9[i] + ER10 * k10[i] + ER11 * k11[i] + ER12 * k12[i])
            err += (erri_8 / sk) ** 2

        err *= h ** 2
        err2 *= h ** 2

        deno = n_vars * (err + 0.01 * err2)
        if deno <= 0.0:
            err = 0.0
        else:
            err = err / np.sqrt(deno)

        # ---- Step size control --------------------------------------------
        fac11 = err ** expo1
        fac = fac11 / (facold ** beta)
        fac = max(facc2, min(facc1, fac / safe))
        h_new = h / fac

        if err <= 1.0:
            # Step accepted — compute FSAL k1 for next step
            k_new1 = f(t_next, y_next, params)

            # ---- Three extra dense-output stages (Hairer §II.6) -----------
            # These are only evaluated when the step is accepted and we need
            # to fill at least one t_eval point inside [t, t_next].
            need_dense = eval_idx < n_out and t_eval[eval_idx] <= t_next

            if need_dense:
                y_tmp = y + h * (A141 * k1 + A147 * k7 + A148 * k8 + A149 * k9
                                + A1410 * k10 + A1411 * k11 + A1412 * k12 + A1413 * k_new1)
                k14 = f(t + C14 * h, y_tmp, params)

                y_tmp = y + h * (A151 * k1 + A156 * k6 + A157 * k7 + A158 * k8
                                + A1511 * k11 + A1512 * k12 + A1513 * k_new1 + A1514 * k14)
                k15 = f(t + C15 * h, y_tmp, params)

                y_tmp = y + h * (A161 * k1 + A166 * k6 + A167 * k7 + A168 * k8
                                + A169 * k9 + A1613 * k_new1 + A1614 * k14 + A1615 * k15)
                k16 = f(t + C16 * h, y_tmp, params)

                cont = _build_cont(h, y, y_next,
                                   k1, k6, k7, k8, k9, k10, k11, k12,
                                   k_new1, k14, k15, k16)

                # Fill every t_eval point that falls within [t, t_next]
                while eval_idx < n_out and t_eval[eval_idx] <= t_next:
                    s = (t_eval[eval_idx] - t) / h
                    y_out[eval_idx] = _contd8(s, cont, n_vars)
                    eval_idx += 1

            facold = max(err, 1.0e-4)
            k1 = k_new1
            y  = y_next
            t  = t_next

            if last:
                h = h_new
                break

            if reject:
                h_new = posneg * min(abs(h_new), abs(h))

            reject = False
        else:
            h_new = h / min(facc1, fac11 / safe)
            reject = True
            last   = False

        h = h_new

    # Guard: if t_eval[-1] coincides exactly with the last accepted t,
    # it may not have been filled yet (floating-point edge case).
    if eval_idx == n_out - 1:
        y_out[eval_idx] = y

    return t_eval, y_out


# ==============================================================================
# Public solver class
# ==============================================================================

class DOP853Solver:
    """Adaptive-step Runge-Kutta 8th-order ODE solver (Dormand & Prince 8(5,3)).

    The right-hand side ``f`` must have the signature::

        f(t: float, y: NDArray[float64], params: NDArray[float64]) -> NDArray[float64]

    The core integration kernels are decorated with ``@njit`` and compiled on
    the first call.  For maximum performance, decorate ``f`` with ``@njit`` as
    well before passing it to the solver.

    Parameters
    ----------
    function : callable
        RHS function with signature ``f(t, y, params)``.
    y0 : array_like
        Initial state vector.
    params : array_like
        Parameter array forwarded unchanged to ``f``.
    rtol : float
        Relative error tolerance (default ``1e-9``).
    atol : float
        Absolute error tolerance (default ``1e-9``).
    n_max_steps : int
        Maximum number of adaptive steps (default ``10000``).
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
        self._y0: NDArray[np.float64] = np.asarray(y0, dtype=np.float64)
        self._params: NDArray[np.float64] = np.asarray(params, dtype=np.float64)

        self.rtol = rtol
        self.atol = atol
        self.n_max_steps = int(n_max_steps)

        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    # ------------------------------------------------------------------
    def solve(
            self,
            t_max: float | None = None,
            dt_initial: float | None = None,
            t_min: float = 0.0,
            t_eval: NDArray[np.float64] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Integrate from ``t_min`` to ``t_max``.

        Two calling modes are supported (legacy API is preserved):

        **Adaptive-grid mode** (original behaviour)::

            t, y = solver.solve(t_max, dt_initial)

        Returns values at the adaptive steps chosen by the error controller.

        **Dense-output mode** (new)::

            t_eval = np.arange(t_min, t_max + step / 2, step)
            t, y   = solver.solve(t_max, dt_initial, t_min, t_eval)
            # — or —
            t, y   = solver.solve(t_eval=t_eval, dt_initial=dt_initial)

        Returns values evaluated at every point in ``t_eval`` using the
        7th-order DOP853 dense-output polynomial (Hairer §II.6), so the
        result is as accurate as the integration itself.

        Parameters
        ----------
        t_max : float, optional
            Final integration time.  Required when ``t_eval`` is *not* given.
        dt_initial : float
            Initial step-size guess for the adaptive controller.
        t_min : float
            Initial time (default ``0.0``).  Ignored when ``t_eval`` is given
            (``t_eval[0]`` is used instead).
        t_eval : array_like, optional
            Prescribed output time grid.  When supplied, the solver uses dense
            output to evaluate the solution at each point.  ``t_eval[0]``
            overrides ``t_min`` and ``t_eval[-1]`` overrides ``t_max``.

        Returns
        -------
        t : NDArray[float64]
            Time points.
        y : NDArray[float64], shape ``(len(t), n_vars)``
            Solution at each time point.
        """
        if t_eval is not None:
            # Dense-output mode
            t_eval = np.asarray(t_eval, dtype=np.float64)
            if dt_initial is None:
                raise ValueError("dt_initial is required.")
            self._t, self._y = _dop853_integrate_t_eval(
                self._f,
                self._y0,
                dt_initial,
                t_eval,
                self._params,
                self.rtol,
                self.atol,
                self.n_max_steps,
            )
        elif t_max is not None and dt_initial is not None:
            # Legacy adaptive-grid mode
            self._t, self._y = _dop853_integrate(
                self._f,
                self._y0,
                dt_initial,
                t_max,
                self._params,
                self.rtol,
                self.atol,
                self.n_max_steps,
            )
        else:
            raise ValueError(
                "Provide either:\n"
                "  solver.solve(t_max, dt_initial)               # adaptive-grid mode\n"
                "  solver.solve(t_eval=..., dt_initial=...)      # dense-output mode"
            )

        return self._t, self._y

    # ------------------------------------------------------------------
    @property
    def t(self) -> NDArray[np.float64]:
        """Time grid from the last ``solve()`` call."""
        if self._t is None:
            raise RuntimeError("Call solve() first.")
        return self._t

    @property
    def y(self) -> NDArray[np.float64]:
        """Solution array from the last ``solve()`` call, shape ``(N, n_vars)``."""
        if self._y is None:
            raise RuntimeError("Call solve() first.")
        return self._y
