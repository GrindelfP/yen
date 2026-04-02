"""
RK4: Convergence study on the Kepler orbit + efficiency vs SciPy DOP853.

Problem: Kepler two-body orbit (GM=1, e=0.5, a=1), one full period T=2*pi.
Because the orbit is exactly periodic, y(T) = y0 is the analytic reference —
no reference solver needed.

What we measure
---------------
* Global error ||y(T) - y0|| as a function of step size h.
* Log-log slope should be ~4 (confirming O(h^4) global order).
* Number of RHS evaluations: RK4 uses exactly 4 per step → f_evals = 4 * N.
* Wall time vs final error: efficiency curve against SciPy DOP853.
"""

import time

import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from scipy.integrate import solve_ivp

from yennefer import RK4Solver


# ==============================================================================
# 1. Right-hand side
# ==============================================================================
@njit
def kepler(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3], dtype=np.float64)


def kepler_scipy(t: float, y: np.ndarray) -> list:
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return [y[2], y[3], -y[0] / r3, -y[1] / r3]


# ==============================================================================
# 2. Setup
# ==============================================================================
e    = 0.5
a    = 1.0
r_p  = a * (1.0 - e)
v_p  = np.sqrt((1.0 + e) / (a * (1.0 - e)))
T    = 2.0 * np.pi * a ** 1.5          # exact period

y0     = np.array([r_p, 0.0, 0.0, v_p], dtype=np.float64)
params = np.array([], dtype=np.float64)

H_VALUES = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]


# ==============================================================================
# 3. Convergence sweep
# ==============================================================================
def run_convergence_study():

    # --- Numba warm-up ---
    _w = RK4Solver(kepler, y0, params)
    _w.solve(T, dt=0.1)

    errors   = []
    f_evals  = []
    times    = []

    print(f"Kepler orbit, one period T = 2π ≈ {T:.6f}")
    print(f"\n{'h':>8} | {'steps':>8} | {'f_evals':>8} | {'error':>12} | {'time (s)':>10}")
    print("-" * 58)

    for h in H_VALUES:
        solver = RK4Solver(kepler, y0, params)

        t0 = time.perf_counter()
        t_arr, y_arr = solver.solve(T, dt=h)
        elapsed = time.perf_counter() - t0

        err  = np.linalg.norm(y_arr[-1] - y0)
        nfe  = 4 * len(t_arr)           # 4 RHS calls per RK4 step

        errors.append(err)
        f_evals.append(nfe)
        times.append(elapsed)

        print(f"{h:>8.4f} | {len(t_arr):>8} | {nfe:>8} | {err:>12.4e} | {elapsed:>10.5f}")

    # --- Estimate slope ---
    # Use the linear (convergent) region
    log_h = np.log10(H_VALUES)
    log_e = np.log10(errors)
    slope, _ = np.polyfit(log_h[2:-1], log_e[2:-1], 1)
    print(f"\nEstimated convergence order (log-log slope): {slope:.2f}  (expected ≈ 4.0)")

    # --- SciPy DOP853 reference points ---
    scipy_times  = []
    scipy_errors = []
    scipy_fevals = []

    for tol in [1e-3, 1e-5, 1e-7, 1e-9, 1e-11]:
        t0 = time.perf_counter()
        res = solve_ivp(kepler_scipy, [0.0, T], y0, method="DOP853",
                        atol=tol, rtol=tol, dense_output=False)
        scipy_times.append(time.perf_counter() - t0)
        scipy_errors.append(np.linalg.norm(res.y[:, -1] - y0))
        scipy_fevals.append(res.nfev)

    # ===========================================================================
    # 4. Plots
    # ===========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("RK4 Convergence: Kepler orbit (e=0.5, one period)", fontsize=13, fontweight="bold")

    # --- Log-log: error vs h ---
    ax1.loglog(H_VALUES, errors, "o-", color="steelblue", lw=1.5, label="RK4 error")
    # Reference line h^4 through the middle of data
    h_ref = np.array([H_VALUES[0], H_VALUES[-1]])
    mid   = len(H_VALUES) // 2
    c4    = errors[mid] / H_VALUES[mid] ** 4
    ax1.loglog(h_ref, c4 * h_ref ** 4, "k--", lw=1, alpha=0.7, label=r"$\propto h^4$")
    ax1.set_xlabel("step size  h", fontsize=11)
    ax1.set_ylabel(r"$\|y(T) - y_0\|$", fontsize=11)
    ax1.set_title(f"Convergence order ≈ {slope:.2f}")
    ax1.legend()
    ax1.grid(True, which="both", alpha=0.3)

    # --- Efficiency: error vs f_evals ---
    ax2.loglog(f_evals, errors, "o-", color="steelblue", lw=1.5, label="RK4  (4 evals/step)")
    ax2.loglog(scipy_fevals, scipy_errors, "s--", color="crimson", lw=1.5, label="SciPy DOP853")
    ax2.set_xlabel("RHS evaluations", fontsize=11)
    ax2.set_ylabel(r"$\|y(T) - y_0\|$", fontsize=11)
    ax2.set_title("Efficiency: accuracy vs cost")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)
    ax2.invert_xaxis()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_convergence_study()
