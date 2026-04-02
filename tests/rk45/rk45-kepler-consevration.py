"""
RK45: Kepler orbit — conservation of energy and angular momentum.

Two-body problem in the gravitational field (GM = 1):
    dx/dt  = vx
    dy/dt  = vy
    dvx/dt = -x / r^3
    dvy/dt = -y / r^3

Conserved quantities:
    E  = 0.5 * (vx^2 + vy^2) - 1/r          (orbital energy)
    Lz = x * vy - y * vx                      (angular momentum)

Initial conditions for an ellipse with semi-major axis a = 1, eccentricity e:
    periapsis  r_p  = a(1-e)
    speed      v_p  = sqrt((1+e) / (a(1-e)))  [vis-viva at periapsis]
    period     T    = 2*pi*a^(3/2) = 2*pi

After N full periods the satellite must return to y0.
Drift in E and Lz measures the long-term quality of the integrator.

Analogue of jacobi-energy-preservation.py used for DOP853.
"""

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

from yennefer import RK45Solver


# ==============================================================================
# 1. Right-hand side
# ==============================================================================
@njit
def kepler(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    r3 = (y[0] ** 2 + y[1] ** 2) ** 1.5
    return np.array([y[2], y[3], -y[0] / r3, -y[1] / r3], dtype=np.float64)


# ==============================================================================
# 2. Invariants
# ==============================================================================
def orbital_energy(y_arr: np.ndarray) -> np.ndarray:
    r = np.sqrt(y_arr[:, 0] ** 2 + y_arr[:, 1] ** 2)
    v2 = y_arr[:, 2] ** 2 + y_arr[:, 3] ** 2
    return 0.5 * v2 - 1.0 / r


def angular_momentum(y_arr: np.ndarray) -> np.ndarray:
    return y_arr[:, 0] * y_arr[:, 3] - y_arr[:, 1] * y_arr[:, 2]


# ==============================================================================
# 3. Test
# ==============================================================================
def run_kepler_test() -> None:
    e        = 0.5          # eccentricity
    a        = 1.0          # semi-major axis
    n_periods = 10          # how many full orbits to integrate

    r_peri = a * (1.0 - e)                            # periapsis distance = 0.5
    v_peri = np.sqrt((1.0 + e) / (a * (1.0 - e)))     # periapsis speed = sqrt(3)
    T      = 2.0 * np.pi * a ** 1.5                   # Kepler period = 2*pi

    y0     = np.array([r_peri, 0.0, 0.0, v_peri], dtype=np.float64)
    t_max  = n_periods * T

    print(f"=== Kepler orbit test (e={e}, a={a}, {n_periods} periods) ===")
    print(f"Periapsis r_p = {r_peri:.4f}, speed v_p = {v_peri:.6f}")
    print(f"Period T = 2*pi = {T:.6f}, T_max = {t_max:.4f}")

    solver = RK45Solver(
        kepler, y0, np.array([], dtype=np.float64),
        atol=1e-8, rtol=1e-8,
    )
    t_arr, y_arr = solver.solve(t_max, dt_initial=1e-3)

    # --- Conserved quantities ---
    E  = orbital_energy(y_arr)
    Lz = angular_momentum(y_arr)

    dE  = E  - E[0]
    dLz = Lz - Lz[0]

    max_dE  = np.max(np.abs(dE))
    max_dLz = np.max(np.abs(dLz))

    # --- Periodicity check ---
    final_err = np.linalg.norm(y_arr[-1] - y0)

    print(f"\nSteps accepted : {len(t_arr)}")
    print(f"Initial energy : E0  = {E[0]:.10f}")
    print(f"Initial L_z    : Lz0 = {Lz[0]:.10f}")
    print(f"Max |dE|       : {max_dE:.4e}")
    print(f"Max |dLz|      : {max_dLz:.4e}")
    print(f"||y(T_max) - y0|| : {final_err:.4e}")

    if max_dE < 1e-5 and max_dLz < 1e-5:
        print("\n✅ PASSED!")
    else:
        print("\n❌ FAILED!")

    # ===========================================================================
    # 4. Plots
    # ===========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"RK45: Kepler orbit  (e={e}, {n_periods} periods, atol=rtol=1e-8)",
        fontsize=13, fontweight="bold",
    )

    # Orbit
    ax = axes[0]
    ax.plot(y_arr[:, 0], y_arr[:, 1], "b-", lw=0.8, alpha=0.9, label="Trajectory")
    ax.plot(0.0, 0.0, "yo", ms=12, label="Focus (GM=1)")
    ax.plot(y0[0], y0[1], "g^", ms=8, label="Periapsis (start)")
    ax.set_aspect("equal")
    ax.set_title("Orbit in x–y plane")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Energy drift
    ax = axes[1]
    ax.plot(t_arr, dE, color="crimson", lw=0.8)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Energy drift  E(t) − E(0)")
    ax.set_xlabel("t")
    ax.set_ylabel("dE")
    ax.grid(True, alpha=0.3)

    # Angular momentum drift
    ax = axes[2]
    ax.plot(t_arr, dLz, color="steelblue", lw=0.8)
    ax.axhline(0, color="black", lw=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Angular momentum drift  Lz(t) − Lz(0)")
    ax.set_xlabel("t")
    ax.set_ylabel("dLz")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_kepler_test()
