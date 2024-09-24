import numpy as np
from sympy.solvers import solve
from sympy import Symbol
from displacement import Displacement


class CalculationResonance(Displacement):
    def __init__(self):
        self.x = Symbol("x")

    # calculation of resonance points
    def SimpleCompositeResonancePoints(self):
        return solve(
            self.A(self.x) ** 4
            - self.A(self.x) ** 2 * self.x / self.s**2
            - 3 * (1 - self.nu) / 2 * self.x**2
            == 0,
            self.x,
        )

    def RefinedCompositeResonancePoints(self):
        return solve(
            (
                self.A(self.x) ** 4
                - 3
                * (1 - self.nu)
                / 2
                * (1 - (7 * self.nu - 17) / (15 * (1 - self.nu)) * self.A(self.x) ** 2)
                * self.x**2
                - 1 / self.s**2 * (1 / self.s**2 + (7 * self.nu - 17) / 10) * self.x**2
            )
            == 0,
            self.x,
        )
