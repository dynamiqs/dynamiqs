from __future__ import annotations

from jaxtyping import PyTree

from ...result import Saved, SEResult
from ...time_array import Shape
from ...utils.operators import eye
from .._utils import _cartesian_vectorize
from ..apis.sesolve import _sesolve
from ..core.abstract_integrator import SEPropagatorIntegrator


class SEPropagatorDynamiqsIntegrator(SEPropagatorIntegrator):
    def run(self) -> PyTree:
        # for technical reasons, matrix-matrix product (for square matrices) is about
        # 20% faster than matrix-vector product. So rather than directly calling
        # sesolve, here we call _sesolve instead and specify the initial state as
        # a square matrix (the identity matrix)
        initial_states = eye(self.H.shape[-1])
        n_batch = (self.H.in_axes, Shape(), Shape(), Shape(), Shape(), Shape(), Shape())
        out_axes = SEResult(False, False, False, False, 0, 0)
        f = _cartesian_vectorize(_sesolve, n_batch, out_axes)
        seresult = f(
            self.H,
            initial_states,
            self.ts,
            None,
            self.solver,
            self.gradient,
            self.options,
        )
        saved = Saved(seresult.states, None, None)
        return self.result(saved, seresult.infos)
