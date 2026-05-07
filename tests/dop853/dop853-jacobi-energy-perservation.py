import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from yennefer import DOP853Solver


@njit
def arenstorf(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    mu = 0.012277471
    mup = 1.0 - mu
    d1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mup) ** 2 + y[1] ** 2) ** 1.5

    return np.array([
        y[2],
        y[3],
        y[0] + 2.0 * y[3] - mup * (y[0] + mu) / d1 - mu * (y[0] - mup) / d2,
        y[1] - 2.0 * y[2] - mup * y[1] / d1 - mu * y[1] / d2
    ], dtype=np.float64)


def calculate_jacobi(y_row):
    mu = 0.012277471
    mup = 1.0 - mu
    x, y, vx, vy = y_row[0], y_row[1], y_row[2], y_row[3]

    d1 = np.sqrt((x + mu) ** 2 + y ** 2)
    d2 = np.sqrt((x - mup) ** 2 + y ** 2)

    return (x ** 2 + y ** 2) + 2.0 * (mup / d1 + mu / d2) - (vx ** 2 + vy ** 2)


def run_energy_test():
    y0 = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])
    t_period = 17.0652165601579625588917206249

    print(f"--- Jacobi test (T_max = {t_period}) ---")

    # Решаем с твоими лучшими настройками
    solver = DOP853Solver(arenstorf, y0, np.array([]), atol=1e-14, rtol=1e-14)
    t_arr, y_arr = solver.solve(t_max=t_period, dt_initial=1e-4)

    # Считаем значения интеграла
    c_values = np.array([calculate_jacobi(row) for row in y_arr])
    c0 = c_values[0]
    deviations = c_values - c0

    max_err = np.max(np.abs(deviations))

    print(f"Points: {len(t_arr)}")
    print(f"Initial C: {c0:.16f}")
    print(f"Max dC: {max_err:.4e}")

    plt.figure(figsize=(10, 5))
    plt.plot(t_arr, deviations, color='crimson', label='Relative Error in C')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5)
    plt.yscale('log')
    plt.title("Conservation of Jacobi Integral (DOP853)")
    plt.xlabel("Time (t)")
    plt.ylabel("|C(t) - C(0)|")
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    print("\n[INFO] Plot done. If dC < 1e-10, solver is adequate.")
    plt.show()


if __name__ == "__main__":
    run_energy_test()
