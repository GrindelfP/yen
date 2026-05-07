from __future__ import annotations
from typing import Callable
from numpy.typing import NDArray
from numba import njit
from .radau_constants import *

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]
JAC = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


@njit
def _estimate_jacobian(f: F, t: float, y: NDArray[np.float64], params: NDArray[np.float64],
                       f_now: NDArray[np.float64]) -> NDArray[np.float64]:
    """Numerical approximation of the Jacobian matrix."""
    n = y.shape[0]
    eps = 1e-8
    jac = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        y_eps = y.copy()
        y_eps[i] += eps
        f_eps = f(t, y_eps, params)
        jac[:, i] = (f_eps - f_now) / eps
    return jac


@njit
def _radau_integrate(
        f: F,
        y0: NDArray[np.float64],
        dt_init: float,
        t_max: float,
        params: NDArray[np.float64],
        rtol: float,
        atol: float,
        n_max_steps: int,
        max_newton_iter: int = 10,
        newton_tol: float = 1e-6
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    print("*** WARNING! This method is not fully impemented and can potentially result in errors! WARNING! ***")
    n_vars = y0.shape[0]
    capacity = 1000
    t_arr = np.empty(capacity, dtype=np.float64)
    y_arr = np.empty((capacity, n_vars), dtype=np.float64)

    t = 0.0
    y = y0.copy()
    t_arr[0] = t
    y_arr[0] = y

    h = dt_init
    idx = 1
    nstep = 0

    while t < t_max and nstep < n_max_steps:
        if t + h > t_max:
            h = t_max - t

        # Initial guess for stages (K = f(t+ch, y+hAK))
        f_now = f(t, y, params)
        k1, k2, k3 = f_now.copy(), f_now.copy(), f_now.copy()

        # Jacobian for Newton method
        dfdy = _estimate_jacobian(f, t, y, params, f_now)

        # Newton iteration to solve the implicit system
        converged = False
        for _ in range(max_newton_iter):
            # Compute residuals
            res1 = k1 - f(t + C1 * h, y + h * (A11 * k1 + A12 * k2 + A13 * k3), params)
            res2 = k2 - f(t + C2 * h, y + h * (A21 * k1 + A22 * k2 + A23 * k3), params)
            res3 = k3 - f(t + C3 * h, y + h * (A31 * k1 + A32 * k2 + A33 * k3), params)

            # System Jacobian: J_sys = I - h(A ⊗ J)
            # Building a 3n x 3n matrix for the Newton step
            size = 3 * n_vars
            sys_jac = np.eye(size)
            for i in range(3):
                for j in range(3):
                    # Coefficients from Butcher A matrix
                    a_val = 0.0
                    if i == 0:
                        a_val = (A11 if j == 0 else A12 if j == 1 else A13)
                    elif i == 1:
                        a_val = (A21 if j == 0 else A22 if j == 1 else A23)
                    else:
                        a_val = (A31 if j == 0 else A32 if j == 1 else A33)

                    sys_jac[i * n_vars:(i + 1) * n_vars, j * n_vars:(j + 1) * n_vars] -= h * a_val * dfdy

            res_vec = np.concatenate((res1, res2, res3))
            delta = np.linalg.solve(sys_jac, res_vec)

            k1 -= delta[0:n_vars]
            k2 -= delta[n_vars:2 * n_vars]
            k3 -= delta[2 * n_vars:]

            if np.linalg.norm(delta) < newton_tol:
                converged = True
                break

        # If Newton didn't converge, reduce step size
        if not converged:
            h /= 2.0
            continue

        # Step acceptance and simple error control (simplified for RADAU)
        y_next = y + h * (B1 * k1 + B2 * k2 + B3 * k3)
        t_next = t + h

        # Dynamic allocation check
        if idx >= capacity:
            capacity *= 2
            new_t = np.empty(capacity, dtype=np.float64)
            new_y = np.empty((capacity, n_vars), dtype=np.float64)
            new_t[:idx] = t_arr[:idx]
            new_y[:idx] = y_arr[:idx]
            t_arr, y_arr = new_t, new_y

        t, y = t_next, y_next
        t_arr[idx] = t
        y_arr[idx] = y
        idx += 1
        nstep += 1

    return t_arr[:idx], y_arr[:idx]


class RADAUSolver:
    """Implicit Runge-Kutta RADAU IIA 5th-order ODE solver.
    Excellent for stiff problems.
    """

    def __init__(
            self,
            function: F,
            y0: NDArray[np.float64],
            params: NDArray[np.float64],
            rtol: float = 1e-6,
            atol: float = 1e-6,
            n_max_steps: int = 10000,
    ) -> None:
        self._f = function
        self._y0 = np.asarray(y0, dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)
        self.rtol = rtol
        self.atol = atol
        self.n_max_steps = int(n_max_steps)
        self._t: NDArray[np.float64] | None = None
        self._y: NDArray[np.float64] | None = None

    def solve(self, t_max: float, dt_init: float) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        self._t, self._y = _radau_integrate(
            self._f, self._y0, dt_init, t_max, self._params, self.rtol, self.atol, self.n_max_steps
        )
        return self._t, self._y

    @property
    def t(self) -> NDArray[np.float64]:
        if self._t is None: raise RuntimeError("Call solve() first.")
        return self._t

    @property
    def y(self) -> NDArray[np.float64]:
        if self._y is None: raise RuntimeError("Call solve() first.")
        return self._y
