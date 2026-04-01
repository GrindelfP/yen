import numpy as np
from numba import njit

from yen import DOP853Solver


@njit
def oscillator(t, y, params):
    return np.array([y[1], -y[0]])


y0 = np.array([1.0, 0.0])
t_max = 10.0

solver = DOP853Solver(oscillator, y0, np.array([]), atol=1e-12, rtol=1e-12)
t, y = solver.solve(t_max, 0.01)

true_final_y = np.array([np.cos(t_max), -np.sin(t_max)])
error = np.linalg.norm(y[-1] - true_final_y)

print(f"Steps: {len(t)}")
print(f"Final ABS error: {error:.2e}")

if error < 1e-10:
    print("✅ PASSED!")
else:
    print("❌ FAILED!")
