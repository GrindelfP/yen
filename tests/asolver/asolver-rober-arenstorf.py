import numpy as np
from numba import njit
from yennefer import ASolver


# --- ROBERSON PROBLEM (STIFF) ---
@njit
def rober(t, y, params):
    dy = np.empty(3, dtype=np.float64)
    dy[0] = -0.04 * y[0] + 1.0e4 * y[1] * y[2]
    dy[1] = 0.04 * y[0] - 1.0e4 * y[1] * y[2] - 3.0e7 * y[1] ** 2
    dy[2] = 3.0e7 * y[1] ** 2
    return dy


# --- ARENSTORF ORBIT (NON-STIFF, HIGH PRECISION) ---
@njit
def arenstorf(t, y, params):
    mu = 0.012277471
    mup = 1.0 - mu
    d1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mup) ** 2 + y[1] ** 2) ** 1.5
    dy = np.empty(4, dtype=np.float64)
    dy[0], dy[1] = y[2], y[3]
    dy[2] = y[0] + 2 * y[3] - mup * (y[0] + mu) / d1 - mu * (y[0] - mup) / d2
    dy[3] = y[1] - 2 * y[2] - mup * y[1] / d1 - mu * y[1] / d2
    return dy


def test_asolver():
    # 1. TEST: ROBERSON (Should pick RADAU)
    print("\n--- Testing Roberson's Problem ---")
    y0_rober = np.array([1.0, 0.0, 0.0])
    asolver_rober = ASolver(rober, y0_rober, [], rtol=1e-10, atol=1e-10)
    t_r, y_r = asolver_rober.solve(t_max=1e5, dt_init=1e-6)

    mass_error = np.max(np.abs(np.sum(y_r, axis=1) - 1.0))
    print(f"Result: {type(asolver_rober._solver).__name__} used.")
    print(f"Max Mass Error: {mass_error:.2e}")
    assert "RADAUSolver" in str(type(asolver_rober._solver))
    assert mass_error < 1e-8

    # 2. TEST: ARENSTORF (Should pick DOP853)
    print("\n--- Testing Arenstorf Orbit ---")
    y0_orbit = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224])
    t_period = 17.06521656015796

    asolver_orbit = ASolver(arenstorf, y0_orbit, [], rtol=1e-13, atol=1e-13)
    t_o, y_o = asolver_orbit.solve(t_max=t_period, dt_init=1e-4)

    orbit_error = np.linalg.norm(y_o[-1] - y0_orbit)
    print(f"Result: {type(asolver_orbit._solver).__name__} used.")
    print(f"Orbit Closure Error: {orbit_error:.2e}")
    assert "DOP853Solver" in str(type(asolver_orbit._solver))
    assert orbit_error < 1e-8


if __name__ == "__main__":
    test_asolver()
