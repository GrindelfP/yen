import numpy as np
import matplotlib.pyplot as plt
from yennefer import RADAUSolver


# ==============================================================================
# 1. Roberson's Problem Definition
# ==============================================================================
def rober(t, y, params):
    """
    Stiff chemical kinetics system.
    y1: A, y2: B, y3: C
    """
    dy = np.empty(3, dtype=np.float64)
    # k1 = 0.04, k2 = 3e7, k3 = 1e4
    dy[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2]
    dy[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] ** 2
    dy[2] = 3.0e7 * y[1] ** 2
    return dy


# ==============================================================================
# 2. Accuracy and Conservation Test
# ==============================================================================
def run_rober_test():
    print("=== Test: Radau IIA - Roberson's Problem (Extreme Stiffness) ===")

    y0 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    t_max = 1.0e5  # Long-term integration

    # RADAU needs tight tolerances for this highly sensitive problem
    solver = RADAUSolver(
        function=rober,
        y0=y0,
        params=np.array([]),
        atol=1e-10,
        rtol=1e-10
    )

    t_arr, y_arr = solver.solve(t_max=t_max, dt_init=1e-6)

    mass_sums = np.sum(y_arr, axis=1)
    mass_errors = np.abs(mass_sums - 1.0)
    max_mass_error = np.max(mass_errors)

    print(f"Total Steps: {len(t_arr)}")
    print(f"Max Mass Conservation Error: {max_mass_error:.4e}")
    print(f"Final concentrations: A={y_arr[-1, 0]:.4e}, B={y_arr[-1, 1]:.4e}, C={y_arr[-1, 2]:.4e}")

    # ==============================================================================
    # 3. Visualization (Logarithmic Scale)
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    ax1.semilogx(t_arr, y_arr[:, 0], label='y1 (Component A)')
    ax1.semilogx(t_arr, y_arr[:, 1] * 1e4, label='y2 * 10^4 (Component B)')
    ax1.semilogx(t_arr, y_arr[:, 2], label='y3 (Component C)')

    ax1.set_title("Roberson's Chemical Kinetics (RADAU IIA)", fontsize=14)
    ax1.set_xlabel("Time (log scale)")
    ax1.set_ylabel("Concentration")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    ax2.loglog(t_arr, mass_errors + 1e-18, color='red', label='|Sum(y) - 1.0|')
    ax2.set_title("Mass Conservation Accuracy", fontsize=12)
    ax2.set_xlabel("Time (log scale)")
    ax2.set_ylabel("Absolute Error")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    if max_mass_error < 1e-8:
        print("✅ PASSED: Accuracy and mass conservation are maintained.")
    else:
        print("❌ FAILED: Mass conservation drift is too high.")


if __name__ == "__main__":
    run_rober_test()
