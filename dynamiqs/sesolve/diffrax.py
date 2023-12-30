from __future__ import annotations

import diffrax
from jaxtyping import PyTree

from ..solvers.base import SESolver
from ..solvers.diffrax import DiffraxSolver, tocomplex, toreal


class SEDiffrax(SESolver, DiffraxSolver):
    @property
    def terms(self):
        def f(t: float, y: PyTree, args) -> PyTree:
            H = args
            y = tocomplex(y)
            H = H(t)
            ynext = -1j * H @ y
            return toreal(ynext)

        return diffrax.ODETerm(f)

    @property
    def args(self):
        return self.H


class SEEuler(SEDiffrax):
    @property
    def solver(self) -> diffrax.AbstractSolver:
        return diffrax.Euler()

    @property
    def dt0(self) -> float | None:
        return self.options.dt


class SEDormandPrince5(SEDiffrax):
    @property
    def solver(self) -> diffrax.AbstractSolver:
        return diffrax.Dopri5()

    @property
    def stepsize_controller(self) -> diffrax.AbstractAdaptiveStepSizeController:
        return diffrax.PIDController(rtol=self.options.rtol, atol=self.options.atol)
