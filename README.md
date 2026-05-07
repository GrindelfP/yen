# YENNEFER ODE SOLVER 


**Yennefer** is a high-performance Python library for solving ordinary differential equations (ODEs), built on top of NumPy and accelerated via [Numba](https://numba.readthedocs.io/) JIT compilation.

---

## Features

- Three integration methods covering fixed-step and adaptive workflows.
- Numba `@njit`-compiled solver cores.
- Unified, minimal API across all solvers.
- Pass arbitrary physical parameters to the RHS without global state.
- Works seamlessly with `@njit`-decorated right-hand sides for maximum performance.
---

## Installation

```bash
pip install yennefer
```

---

## Solvers

| Class | Method | Order | Step control |
|---|---|---|---|
| `RK4Solver` | Classical Runge–Kutta | 4 | Fixed |
| `RK45Solver` | Runge–Kutta–Fehlberg 4(5) | 4/5 | Adaptive |
| `DOP853Solver` | Dormand–Prince 8(5,3) | 8 | Adaptive |

---

## Example — Josephson Junction (RCSJ model)

The RCSJ model of a Josephson junction:

$$
\begin{cases}
\frac{dV}{dt} = I_c + A\sin(u) - \beta V - \sin\varphi\\
\frac{d\varphi}{dt} = V \\
\frac{du}{dt} = \omega
\end{cases}
\tag{1}
$$

```python
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from yennefer import RK4Solver, RK45Solver, DOP853Solver

@njit
def jj_system(t, y, p):
    V, ph, u = y
    Ic, A, beta, omega = p[0], p[1], p[2], p[3]
    return np.array([
        Ic + A * np.sin(u) - beta * V - np.sin(ph),
        V,
        omega,
    ])

y0     = np.array([0.0, 0.0, 0.0])
params = np.array([0.1, 0.5, 0.2, 2.0])  # Ic, A, beta, omega
```

#### Fixed step with RK4

```python
solver = RK4Solver(jj_system, y0, params)
T, Y = solver.solve(300.0, 5e-2)
```

#### Adaptive step with RK45

```python
solver = RK45Solver(jj_system, y0, params)
T, Y = solver.solve(300.0, dt_initial=5e-2)
```

#### High-accuracy with DOP853

```python
solver = DOP853Solver(jj_system, y0, params)
T, Y = solver.solve(300.0, dt_initial=5e-2)
```

#### Plot

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

ax1.plot(T, Y[:, 0], label="V(t)")
ax1.set_xlabel("t"); ax1.set_ylabel("V"); ax1.legend()

ax2.plot(T, Y[:, 1], label="φ(t)")
ax2.set_xlabel("t"); ax2.set_ylabel("φ"); ax2.legend()

plt.tight_layout()
plt.show()
```

More precise example is presented in [example usage IPython notebook](https://github.com/GrindelfP/yen/blob/main/example_josephon_junction.ipynb).

Another example is [here](https://github.com/GrindelfP/yen/blob/main/example_noised_oscillator.ipynb).

___

### RK4Solver — Fixed-Step 4th-Order Runge–Kutta

The classical RK4 method. Every step uses exactly 4 RHS evaluations. Best suited for smooth problems where you know the required step size in advance, or when uniform output is needed.

```python
solver = RK4Solver(function, y0, params)
t, y = solver.solve(t_max, dt)
```

| Parameter | Description |
|---|---|
| `function` | RHS `f(t, y, params) -> dy` |
| `y0` | Initial state vector (`np.float64` array) |
| `params` | Parameter vector passed verbatim to `f` |
| `t_max` | Integration horizon |
| `dt` | Fixed step size |

### RK45Solver — Adaptive Runge–Kutta–Fehlberg 4(5)

Embedded 4(5) pair with automatic step-size control. The 5th-order solution advances the state; the difference between 4th and 5th order estimates drives the error controller. 6 RHS evaluations per accepted step. Good general-purpose choice for moderate accuracy requirements.

```python
solver = RK45Solver(function, y0, params, atol=1e-6, rtol=1e-3, max_step=np.inf)
t, y = solver.solve(t_max, dt_initial=1e-3)
```

| Parameter | Description |
|---|---|
| `atol` | Absolute error tolerance (default `1e-6`) |
| `rtol` | Relative error tolerance (default `1e-3`) |
| `max_step` | Maximum allowed step size (default `∞`) |
| `dt_initial` | Initial step size guess |

### DOP853Solver — Dormand–Prince 8(5,3)

Implementation of the classic Hairer–Nørsett–Wanner DOP853 method from *Solving ODEs I* (1993), based on the original Fortran source by Hairer & Williams. Dual embedded error estimators (5th and 3rd order), FSAL reuse (effectively 11 new RHS calls per accepted step). The go-to solver for high-accuracy and stiff-sensitive problems.

```python
solver = DOP853Solver(function, y0, params, rtol=1e-9, atol=1e-9, n_max_steps=10000)
t, y = solver.solve(t_max, dt_init)
```

| Parameter | Description |
|---|---|
| `rtol` | Relative error tolerance (default `1e-9`) |
| `atol` | Absolute error tolerance (default `1e-9`) |
| `n_max_steps` | Maximum number of accepted steps (default `10000`) |
| `dt_init` | Initial step size guess |

---

## Common API

All solvers share the same RHS signature convention and output contract.

### Right-hand side signature

```python
@njit
def f(t: float, y: np.ndarray, params: np.ndarray) -> np.ndarray:
    ...
```

Decorating `f` with `@njit` is optional but strongly recommended — it eliminates the Python call overhead inside the inner loop and yields the best performance.

### Output

`solver.solve(...)` returns a tuple `(t, y)`:

| Name | Shape | Description |
|---|---|---|
| `t` | `(N,)` | Time grid |
| `y` | `(N, n_vars)` | Solution at each time point |

Both are also available as `.t` and `.y` properties after the call.

---

## Performance tips

- Always decorate the RHS with `@njit`. The first call triggers compilation (warm-up); subsequent calls run at native speed.
- If you benchmark solvers against each other or against SciPy, run a short warm-up integration first to exclude JIT compilation time from the measurement.
- For adaptive solvers, tighter tolerances (`1e-12` — `1e-14`) with DOP853 are often faster than the same accuracy with RK45, because DOP853 accepts far fewer steps.
