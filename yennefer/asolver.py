import numpy as np
from numpy.typing import NDArray
from typing import Callable

from .dop853 import DOP853Solver
from .radau import RADAUSolver, _estimate_jacobian

F = Callable[[float, NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]


class ASolver:
    """
    Automatic ODE Solver for the Yennefer library.

    Evaluates the initial stiffness of the system using the local Jacobian.
    Routes to RADAUSolver if the system is stiff, and DOP853Solver if it is non-stiff.
    """

    def __init__(
            self,
            function: F,
            y0: NDArray[np.float64] | list[float],
            params: NDArray[np.float64] | list[float],
            rtol: float = 1e-8,
            atol: float = 1e-8,
            n_max_steps: int = 100000,
            stiffness_threshold: float = 10000.0,
            verbose: bool = True
    ) -> None:
        self._f = function
        self._y0 = np.asarray(y0, dtype=np.float64)
        self._params = np.asarray(params, dtype=np.float64)

        self.rtol = rtol
        self.atol = atol
        self.n_max_steps = n_max_steps
        self.stiffness_threshold = stiffness_threshold
        self.verbose = verbose

        self._solver = None

    def _detect_stiffness(self, t_init: float, t_max: float) -> tuple[bool, float]:
        """
        Estimates the 'Work Index' of the problem.
        Uses max_eig * time_span to decide if explicit methods will fail
        over the requested duration.
        """
        f_now = self._f(t_init, self._y0, self._params)
        jac = _estimate_jacobian(self._f, t_init, self._y0, self._params, f_now)

        eigenvalues = np.linalg.eigvals(jac)
        max_eig = np.max(np.abs(eigenvalues))

        # Reverting to time-span scaling as it proved more robust for your benchmarks
        stiffness_index = float(max_eig * abs(t_max - t_init))

        is_stiff = stiffness_index > self.stiffness_threshold
        return is_stiff, stiffness_index

    def solve(self, t_max: float, dt_init: float, t_init: float = 0.0) -> tuple[
        NDArray[np.float64], NDArray[np.float64]]:
        """
        Detects stiffness, instantiates the proper solver, and integrates.
        """
        is_stiff, stiffness_index = self._detect_stiffness(t_init, t_max)

        if is_stiff:
            if self.verbose:
                print(f"[ASolver] Stiffness index {stiffness_index:.2f} > {self.stiffness_threshold}.")
                print("[ASolver] System is STIFF. Routing to RADAU (Order 5 Implicit).")

            self._solver = RADAUSolver(
                self._f,
                self._y0,
                self._params,
                rtol=self.rtol,
                atol=self.atol,
                n_max_steps=self.n_max_steps
            )
        else:
            if self.verbose:
                print(f"[ASolver] Stiffness index {stiffness_index:.2f} <= {self.stiffness_threshold}.")
                print("[ASolver] System is NON-STIFF. Routing to DOP853 (Order 8 Explicit).")

            self._solver = DOP853Solver(
                self._f,
                self._y0,
                self._params,
                rtol=self.rtol,
                atol=self.atol,
                n_max_steps=self.n_max_steps
            )

        return self._solver.solve(t_max, dt_init)

    @property
    def t(self) -> NDArray[np.float64]:
        if self._solver is None:
            raise RuntimeError("Call solve() first.")
        return self._solver.t

    @property
    def y(self) -> NDArray[np.float64]:
        if self._solver is None:
            raise RuntimeError("Call solve() first.")
        return self._solver.y