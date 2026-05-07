"""
RADAU — Radau IIA implicit Runge-Kutta method (order 5).

Source: Hairer E., Wanner G.
        "Solving Ordinary Differential Equations II: Stiff and
        Differential-Algebraic Problems", 2nd ed., Springer, 1996.
        Section IV.8.
"""
import numpy as np

SQ6 = 2.449489742783178098197284074705

C1 = (4.0 - SQ6) / 10.0
C2 = (4.0 + SQ6) / 10.0
C3 = 1.0

A11 = (88.0 - 7.0 * SQ6) / 360.0
A12 = (296.0 - 169.0 * SQ6) / 1800.0
A13 = (-2.0 + 3.0 * SQ6) / 225.0

A21 = (296.0 + 169.0 * SQ6) / 1800.0
A22 = (88.0 + 7.0 * SQ6) / 360.0
A23 = (-2.0 - 3.0 * SQ6) / 225.0

A31 = (16.0 - SQ6) / 36.0
A32 = (16.0 + SQ6) / 36.0
A33 = 1.0 / 9.0

B1 = A31
B2 = A32
B3 = A33

_SQ6 = 6.0 ** 0.5
E1 = (-13.0 - 7.0 * _SQ6) / 3.0
E2 = (-13.0 + 7.0 * _SQ6) / 3.0
E3 = -1.0 / 3.0

STEP_EXPO = 1.0 / 6.0

STEP_SAFE    = 0.9
STEP_MIN_FAC = 0.2
STEP_MAX_FAC = 10.0

NEWTON_MAXITER = 10

JAC_REUSE_RATE = 0.5
