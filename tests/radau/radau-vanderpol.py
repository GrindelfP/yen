import numpy as np
import matplotlib.pyplot as plt
from yennefer import RADAUSolver


# ==============================================================================
# 1. Stiff Van der Pol Equation
# ==============================================================================
def vanderpol_stiff(t, y, params):
    mu = 1000.0
    dy = np.empty(2, dtype=np.float64)
    dy[0] = y[1]
    dy[1] = mu * (1.0 - y[0] ** 2) * y[1] - y[0]
    return dy


# ==============================================================================
# 2. Test Execution
# ==============================================================================
def run_stiff_test():
    print("=== Test: Radau IIA - Stiff Van der Pol (mu=1000) ===")

    y0 = np.array([2.0, 0.0], dtype=np.float64)
    t_max = 3000.0

    solver = RADAUSolver(
        function=vanderpol_stiff,
        y0=y0,
        params=np.array([]),
        atol=1e-6,
        rtol=1e-6
    )

    t_arr, y_arr = solver.solve(t_max=t_max, dt_init=1e-4)

    print(f"Total Steps: {len(t_arr)}")
    print(f"Final State: {y_arr[-1]}")

    # ==============================================================================
    # 3. Visualization
    # ==============================================================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(t_arr, y_arr[:, 0], color='royalblue', label='y[0] (Position)')
    ax1.set_title(f"Stiff Van der Pol Oscillator (mu=1000) - RADAU IIA", fontsize=14)
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    dt_arr = np.diff(t_arr)
    ax2.semilogy(t_arr[:-1], dt_arr, color='crimson', label='Step Size (dt)')
    ax2.set_title("Solver Efficiency: Adaptive Step Size", fontsize=12)
    ax2.set_xlabel("Time (t)")
    ax2.set_ylabel("dt (log scale)")
    ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()

    max_val = np.max(y_arr[:, 0])
    if 1.9 < max_val < 2.1:
        print("✅ PASSED: Limit cycle amplitude is correct.")
    else:
        print(f"❌ FAILED: Unexpected amplitude {max_val:.4f}")


if __name__ == "__main__":
    run_stiff_test()
