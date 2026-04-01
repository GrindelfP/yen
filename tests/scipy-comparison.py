import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numba import njit
import time

from yen import DOP853Solver


@njit
def arenstorf(t, y, params):
    mu = 0.012277471
    mup = 1.0 - mu
    d1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mup) ** 2 + y[1] ** 2) ** 1.5
    return np.array([
        y[2],
        y[3],
        y[0] + 2.0 * y[3] - mup * (y[0] + mu) / d1 - mu * (y[0] - mup) / d2,
        y[1] - 2.0 * y[2] - mup * y[1] / d1 - mu * y[1] / d2
    ])


def arenstorf_scipy(t, y):
    return arenstorf(t, y, np.array([]))


def run_head_to_head_comparison():
    y0 = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])
    t_period = 17.0652165601579625588917206249
    atol = rtol = 1e-12

    print(f"{'Param':<25} | {'My DOP853':<20} | {'SciPy (Fortran)':<20}")
    print("-" * 70)

    start_time = time.time()
    your_solver = DOP853Solver(arenstorf, y0, np.array([]), atol=atol, rtol=rtol)
    t_your, y_your = your_solver.solve(t_max=t_period, dt_init=1e-4)
    your_duration = time.time() - start_time

    start_time = time.time()
    res_sci = solve_ivp(arenstorf_scipy, [0, t_period], y0, method='DOP853',
                        atol=atol, rtol=rtol, first_step=1e-4)
    scipy_duration = time.time() - start_time

    print(f"{'Steps':<25} | {len(t_your):<20} | {len(res_sci.t):<20}")
    print(f"{'Calculation time (s)':<25} | {your_duration:<20.4f} | {scipy_duration:<20.4f}")

    err_your = np.linalg.norm(y_your[-1] - y0)
    err_sci = np.linalg.norm(res_sci.y[:, -1] - y0)
    print(f"{'Final error':<25} | {err_your:<20.4e} | {err_sci:<20.4e}")

    diff = np.linalg.norm(y_your[-1] - res_sci.y[:, -1])
    print("-" * 70)
    print(f"Difference between results in T_final: {diff:.2e}")

    plt.figure(figsize=(12, 6))
    plt.step(t_your[:-1], np.diff(t_your), where='post', label='My DOP853', alpha=0.7)
    plt.step(res_sci.t[:-1], np.diff(res_sci.t), where='post', label='SciPy DOP853', alpha=0.7, linestyle='--')
    plt.yscale('log')
    plt.title('Step Size History Comparison')
    plt.xlabel('t')
    plt.ylabel('dt (log scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.show()


if __name__ == "__main__":
    run_head_to_head_comparison()
