import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time
from yennefer import RADAUSolver


def vanderpol_stiff(t, y, params=None):
    mu = 1000.0
    return np.array([
        y[1],
        mu * (1.0 - y[0] ** 2) * y[1] - y[0]
    ], dtype=np.float64)


def run_duel():
    y0 = np.array([2.0, 0.0], dtype=np.float64)
    t_span = (0, 3000.0)
    tol = 1e-6

    print(f"--- DUEL: Yennefer RADAU vs SciPy RADAU (tol={tol}) ---")

    start_sci = time.perf_counter()
    sol_sci = solve_ivp(
        vanderpol_stiff,
        t_span,
        y0,
        method='Radau',
        atol=tol,
        rtol=tol
    )
    end_sci = time.perf_counter()

    def f(t, y, p): return vanderpol_stiff(t, y)

    solver = RADAUSolver(
        f,
        y0,
        np.array([]),
        atol=tol,
        rtol=tol
    )

    start_yen = time.perf_counter()
    t_yen, y_yen = solver.solve(t_max=t_span[1], dt_init=1e-4)
    end_yen = time.perf_counter()

    print(f"\n[SciPy Radau]")
    print(f"Steps: {len(sol_sci.t)}")
    print(f"Time:  {end_sci - start_sci:.4f}s")

    print(f"\n[Yennefer Radau]")
    print(f"Steps: {len(t_yen)}")
    print(f"Time:  {end_yen - start_yen:.4f}s")

    plt.figure(figsize=(12, 6))
    plt.plot(sol_sci.t, sol_sci.y[0], 'r-', alpha=0.6, label='SciPy (Position)')
    plt.plot(t_yen, y_yen[:, 0], 'b--', alpha=0.8, label='Yennefer (Position)')

    plt.title("Duel: Van der Pol Oscillator (mu=1000)")
    plt.xlabel("Time")
    plt.ylabel("y[0]")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    plt.figure(figsize=(12, 4))
    plt.semilogy(sol_sci.t[:-1], np.diff(sol_sci.t), 'r', alpha=0.5, label='dt SciPy')
    plt.semilogy(t_yen[:-1], np.diff(t_yen), 'b', alpha=0.5, label='dt Yennefer')
    plt.title("Step Size Adaptation Comparison")
    plt.ylabel("dt")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    run_duel()
