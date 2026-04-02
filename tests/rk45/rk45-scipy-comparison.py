"""
RK45 vs SciPy RK45: Van der Pol oscillator (mu=1).

Van der Pol:
    dx/dt = y
    dy/dt = mu * (1 - x^2) * y - x

At mu=1 the system is moderately nonlinear and non-stiff — a clean RK45 benchmark.
Both solvers use the same method, so step-count and trajectory differences
reflect implementation details rather than algorithmic differences.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

from yennefer import RK45Solver


# ==============================================================================
# 1. Right-hand side
# ==============================================================================
@njit
def van_der_pol(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    mu = params[0]
    return np.array([y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]], dtype=np.float64)


def van_der_pol_scipy(t: float, y: np.ndarray) -> list:
    mu = 1.0
    return [y[1], mu * (1.0 - y[0] ** 2) * y[1] - y[0]]


# ==============================================================================
# 2. Benchmark
# ==============================================================================
def run_comparison() -> None:
    y0      = np.array([2.0, 0.0], dtype=np.float64)
    params  = np.array([1.0], dtype=np.float64)   # mu = 1
    t_max   = 50.0
    atol    = rtol = 1e-6

    print("=== Van der Pol (mu=1): RK45 comparison ===")
    print(f"{'Parameter':<28} | {'My RK45':>16} | {'SciPy RK45':>16}")
    print("-" * 68)

    # --- Warm-up (numba compilation) ---
    _solver_wm = RK45Solver(van_der_pol, y0, params, atol=atol, rtol=rtol)
    _solver_wm.solve(0.1, dt_initial=1e-3)

    # --- My RK45 ---
    t0 = time.perf_counter()
    solver = RK45Solver(van_der_pol, y0, params, atol=atol, rtol=rtol)
    t_our, y_our = solver.solve(t_max, dt_initial=1e-3)
    our_time = time.perf_counter() - t0

    # --- SciPy RK45 ---
    t0 = time.perf_counter()
    res = solve_ivp(
        van_der_pol_scipy,
        [0.0, t_max],
        y0,
        method="RK45",
        atol=atol,
        rtol=rtol,
        first_step=1e-3,
        dense_output=False,
    )
    scipy_time = time.perf_counter() - t0

    diff = np.linalg.norm(y_our[-1] - res.y[:, -1])

    print(f"{'Steps accepted':<28} | {len(t_our):>16} | {len(res.t):>16}")
    print(f"{'Calc time (s)':<28} | {our_time:>16.4f} | {scipy_time:>16.4f}")
    print(f"{'y[0] at T_final':<28} | {y_our[-1, 0]:>16.8f} | {res.y[0, -1]:>16.8f}")
    print(f"{'y[1] at T_final':<28} | {y_our[-1, 1]:>16.8f} | {res.y[1, -1]:>16.8f}")
    print("-" * 68)
    print(f"||y_our - y_scipy|| at T_final : {diff:.3e}")

    # ===========================================================================
    # 3. Plots
    # ===========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RK45 vs SciPy RK45: Van der Pol (mu=1)", fontsize=14, fontweight="bold")

    # --- Trajectories ---
    ax = axes[0]
    ax.plot(t_our,  y_our[:, 0],  label="My RK45",    lw=1.5)
    ax.plot(res.t,  res.y[0, :],  label="SciPy RK45", lw=1.5, linestyle="--", alpha=0.8)
    ax.set_title("x(t) trajectory")
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Step size history ---
    ax = axes[1]
    ax.step(t_our[:-1], np.diff(t_our), where="post", label="My RK45",    alpha=0.8, lw=1)
    ax.step(res.t[:-1], np.diff(res.t), where="post", label="SciPy RK45", alpha=0.8, lw=1, linestyle="--")
    ax.set_yscale("log")
    ax.set_title("Step-size history")
    ax.set_xlabel("t")
    ax.set_ylabel("dt  [log scale]")
    ax.legend()
    ax.grid(True, which="both", alpha=0.2)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_comparison()
