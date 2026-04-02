"""
RK4: Lotka-Volterra — long-time Hamiltonian conservation test.

Equations (prey x, predator y):
    dx/dt = alpha*x - beta*x*y
    dy/dt = delta*x*y - gamma*y

First integral (Hamiltonian):
    H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y) = const

RK4 is not symplectic, so H drifts secularly over long integrations.
The test demonstrates:
  1. Short-time accuracy — trajectories in phase space look correct.
  2. Long-time degradation — H(t) - H(0) grows with time (rate ∝ h^4).
  3. h-dependence of drift — coarse and fine step give visibly different results.

Parameters: classic Wolf (1978) set.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from yennefer import RK4Solver


# ==============================================================================
# 1. Right-hand side and Hamiltonian
# ==============================================================================
ALPHA = 1.0
BETA  = 0.1
DELTA = 0.075
GAMMA = 1.5

@njit
def lotka_volterra(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    alpha, beta, delta, gamma = params[0], params[1], params[2], params[3]
    return np.array([
        alpha  * y[0] - beta  * y[0] * y[1],
        delta  * y[0] * y[1] - gamma * y[1],
    ], dtype=np.float64)


def hamiltonian(y_arr: np.ndarray) -> np.ndarray:
    """H = delta*x - gamma*ln(x) + beta*y - alpha*ln(y)"""
    x, y = y_arr[:, 0], y_arr[:, 1]
    return DELTA * x - GAMMA * np.log(x) + BETA * y - ALPHA * np.log(y)


# ==============================================================================
# 2. Test
# ==============================================================================
def run_lv_test() -> None:
    y0     = np.array([10.0, 5.0], dtype=np.float64)
    params = np.array([ALPHA, BETA, DELTA, GAMMA], dtype=np.float64)
    t_max  = 600.0           # ≈ 15–20 predator-prey cycles

    H_COARSE = 0.02
    H_FINE   = 0.005

    print("=== Lotka-Volterra conservation test (RK4) ===")
    print(f"alpha={ALPHA}, beta={BETA}, delta={DELTA}, gamma={GAMMA}")
    print(f"y0 = {y0},  T_max = {t_max}")

    # --- Numba warm-up ---
    _w = RK4Solver(lotka_volterra, y0, params)
    _w.solve(1.0, dt=H_COARSE)

    results = {}
    for h, label in [(H_COARSE, "coarse"), (H_FINE, "fine")]:
        solver = RK4Solver(lotka_volterra, y0, params)
        t_arr, y_arr = solver.solve(t_max, dt=h)

        H_arr  = hamiltonian(y_arr)
        dH     = H_arr - H_arr[0]
        max_dH = np.max(np.abs(dH))

        results[label] = dict(t=t_arr, y=y_arr, dH=dH, H0=H_arr[0], max_dH=max_dH, h=h)

        print(f"\n  h = {h}  ({label})")
        print(f"  Steps        : {len(t_arr)}")
        print(f"  H(0)         : {H_arr[0]:.10f}")
        print(f"  Max |dH|     : {max_dH:.4e}")

    ratio = results["coarse"]["max_dH"] / results["fine"]["max_dH"]
    expected = (H_COARSE / H_FINE) ** 4
    print(f"\n  Drift ratio (coarse/fine): {ratio:.1f}  (expected ≈ {expected:.0f} for O(h^4))")

    # ===========================================================================
    # 3. Plots
    # ===========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        f"RK4: Lotka-Volterra conservation  (T={t_max}, α={ALPHA}, β={BETA}, δ={DELTA}, γ={GAMMA})",
        fontsize=13, fontweight="bold",
    )

    # --- Phase portraits ---
    ax = axes[0, 0]
    for label, color in [("coarse", "tomato"), ("fine", "steelblue")]:
        r = results[label]
        ax.plot(r["y"][:, 0], r["y"][:, 1], lw=0.6, alpha=0.7,
                color=color, label=f"h={r['h']}")
    ax.plot(*y0, "k*", ms=10, label="start")
    ax.set_xlabel("prey  x")
    ax.set_ylabel("predator  y")
    ax.set_title("Phase portrait")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Trajectories x(t) ---
    ax = axes[0, 1]
    for label, color in [("coarse", "tomato"), ("fine", "steelblue")]:
        r = results[label]
        ax.plot(r["t"], r["y"][:, 0], lw=0.7, alpha=0.8,
                color=color, label=f"h={r['h']}")
    ax.set_xlabel("t")
    ax.set_ylabel("prey  x(t)")
    ax.set_title("Prey population over time")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Hamiltonian drift: coarse ---
    ax = axes[1, 0]
    r = results["coarse"]
    ax.plot(r["t"], r["dH"], color="tomato", lw=0.8)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("H(t) − H(0)")
    ax.set_title(f"Hamiltonian drift  (h={r['h']}, max|dH|={r['max_dH']:.2e})")
    ax.grid(True, alpha=0.3)

    # --- Hamiltonian drift: fine ---
    ax = axes[1, 1]
    r = results["fine"]
    ax.plot(r["t"], r["dH"], color="steelblue", lw=0.8)
    ax.axhline(0, color="black", lw=0.7, ls="--", alpha=0.5)
    ax.set_xlabel("t")
    ax.set_ylabel("H(t) − H(0)")
    ax.set_title(f"Hamiltonian drift  (h={r['h']}, max|dH|={r['max_dH']:.2e})")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_lv_test()
