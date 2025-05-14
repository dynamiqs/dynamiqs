import equinox as eqx
from jaxtyping import Array

from ...result import Result, SolveSaved
from ..apis.jssesolve import _vectorized_jssesolve
from .abstract_integrator import BaseIntegrator
from .interfaces import MEInterface, SolveInterface


class JumpMonteCarloExtra(eqx.Module):
    clicktimes: Array
    final_state_norm: Array


class MESolveJumpMonteCarloIntegrator(BaseIntegrator, MEInterface, SolveInterface):
    def run(self) -> Result:
        # call _vectorized_jssesolve to compute the jump SSE results
        jsse_result = _vectorized_jssesolve(
            self.H,
            self.Ls,
            self.y0,
            self.ts,
            self.method.keys,
            self.Es,
            self.method.jsse_method,
            self.gradient,
            self.method.jsse_options,
        )

        # compute saved with the mean results
        saved = SolveSaved(
            ysave=jsse_result.mean_states,
            Esave=jsse_result.mean_expects,
            extra=JumpMonteCarloExtra(
                clicktimes=jsse_result.clicktimes,
                final_state_norm=jsse_result.final_state_norm,
            ),
        )

        # return
        return self.result(saved, infos=None)


mesolve_jumpmontecarlo_integrator_constructor = MESolveJumpMonteCarloIntegrator
