import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from numba import njit

from yen import DOP853Solver


# ==============================================================================
# 1. Arenstorf Orbit
# ==============================================================================
@njit
def arenstorf(t, y, params):
    mu = 0.012277471
    mup = 1.0 - mu

    d1 = ((y[0] + mu) ** 2 + y[1] ** 2) ** 1.5
    d2 = ((y[0] - mup) ** 2 + y[1] ** 2) ** 1.5

    dy = np.empty(4, dtype=np.float64)
    dy[0] = y[2]
    dy[1] = y[3]
    dy[2] = y[0] + 2.0 * y[3] - mup * (y[0] + mu) / d1 - mu * (y[0] - mup) / d2
    dy[3] = y[1] - 2.0 * y[2] - mup * y[1] / d1 - mu * y[1] / d2

    return dy


# ==============================================================================
# 2. Test
# ==============================================================================
MU_VAL = 0.012277471


def run_simulation_and_get_data():
    print("=== Test: Arenstorf Orbit ===")

    y0 = np.array([0.994, 0.0, 0.0, -2.00158510637908252240537862224], dtype=np.float64)
    t_period = 17.0652165601579625588917206249

    solver = DOP853Solver(
        function=arenstorf,
        y0=y0,
        params=np.array([], dtype=np.float64),
        atol=1e-13,
        rtol=1e-13,
        n_max_steps=100000
    )

    t_arr, y_arr = solver.solve(t_max=t_period, dt_init=1e-4)

    final_y = y_arr[-1]
    error = np.linalg.norm(final_y - y0)

    print(f"Steps: {len(t_arr)}")
    print(f"Initial condition: {y0}")
    print(f"Final condition:  {final_y}")
    print(f"Final error: {error:.4e}")

    if error < 1e-8:
        print("✅ PASSED!")
    else:
        print("❌ FAILED!")


    # For animation
    x_rot = y_arr[:, 0]
    y_rot = y_arr[:, 1]

    x_in = np.zeros_like(t_arr)
    y_in = np.zeros_like(t_arr)
    earth_x = np.zeros_like(t_arr)
    earth_y = np.zeros_like(t_arr)
    moon_x = np.zeros_like(t_arr)
    moon_y = np.zeros_like(t_arr)

    for i in range(len(t_arr)):
        t = t_arr[i]
        cos_t = np.cos(t)
        sin_t = np.sin(t)

        x_in[i] = x_rot[i] * cos_t - y_rot[i] * sin_t
        y_in[i] = x_rot[i] * sin_t + y_rot[i] * cos_t

        # Earth
        earth_x[i] = (-MU_VAL) * cos_t
        earth_y[i] = (-MU_VAL) * sin_t

        # Moon
        moon_x[i] = (1.0 - MU_VAL) * cos_t
        moon_y[i] = (1.0 - MU_VAL) * sin_t

    return t_arr, x_rot, y_rot, x_in, y_in, earth_x, earth_y, moon_x, moon_y

# ==============================================================================
# 3. Animation
# ==============================================================================
def animate():
    t_arr, x_rot, y_rot, x_in, y_in, earth_x, earth_y, moon_x, moon_y = run_simulation_and_get_data()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('DOP853: Arenstorf Orbit Comparison', fontsize=16, fontweight='bold')

    for ax in (ax1, ax2):
        ax.set_aspect('equal')
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')

    ax1.set_title('Rotating Frame (Static Earth & Moon)')
    ax1.plot(-MU_VAL, 0, 'bo', markersize=14, label='Earth')
    ax1.plot(1.0 - MU_VAL, 0, 'o', color='gray', markersize=7, label='Moon')
    orbit_rot_line, = ax1.plot([], [], 'r-', lw=1, alpha=0.8, label='Trajectory')
    sat_rot, = ax1.plot([], [], 'ro', markersize=5)
    ax1.legend(loc='upper right')

    ax2.set_title('Inertial Frame (Moving Earth & Moon)')
    ax2.plot(earth_x, earth_y, 'b--', alpha=0.15)
    ax2.plot(moon_x, moon_y, 'k--', alpha=0.15)
    orbit_in_line, = ax2.plot([], [], 'r-', lw=1, alpha=0.8, label='Trajectory')
    sat_in, = ax2.plot([], [], 'ro', markersize=5)
    earth_point, = ax2.plot([], [], 'bo', markersize=14, label='Earth')
    moon_point, = ax2.plot([], [], 'o', color='gray', markersize=7, label='Moon')
    ax2.legend(loc='upper right')

    time_template = 'Time = %.3f'
    dt_template = 'Step dt = %.2e'
    time_text = ax1.text(0.05, 0.93, '', transform=ax1.transAxes, fontsize=11, fontweight='bold')
    dt_text = ax1.text(0.05, 0.88, '', transform=ax1.transAxes, fontsize=10)

    def init():
        orbit_rot_line.set_data([], [])
        sat_rot.set_data([], [])
        orbit_in_line.set_data([], [])
        sat_in.set_data([], [])
        earth_point.set_data([], [])
        moon_point.set_data([], [])
        time_text.set_text('')
        dt_text.set_text('')
        return orbit_rot_line, sat_rot, orbit_in_line, sat_in, earth_point, moon_point, time_text, dt_text

    def update(frame):
        orbit_rot_line.set_data(x_rot[:frame], y_rot[:frame])
        sat_rot.set_data([x_rot[frame]], [y_rot[frame]])

        orbit_in_line.set_data(x_in[:frame], y_in[:frame])
        sat_in.set_data([x_in[frame]], [y_in[frame]])
        earth_point.set_data([earth_x[frame]], [earth_y[frame]])
        moon_point.set_data([moon_x[frame]], [moon_y[frame]])

        time_text.set_text(time_template % t_arr[frame])
        if frame > 0:
            current_dt = t_arr[frame] - t_arr[frame - 1]
            dt_text.set_text(dt_template % current_dt)

        return orbit_rot_line, sat_rot, orbit_in_line, sat_in, earth_point, moon_point, time_text, dt_text

    ani = FuncAnimation(
        fig,
        update,
        frames=len(t_arr),
        init_func=init,
        blit=True,
        interval=50,
        repeat=True
    )

    plt.show()


if __name__ == "__main__":
    animate()
