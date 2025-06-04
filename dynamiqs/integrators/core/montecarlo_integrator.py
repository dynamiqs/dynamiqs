import warnings
from dataclasses import replace

from ...result import Result, SolveSaved
from ..apis.jssesolve import _vectorized_jssesolve
from .abstract_integrator import BaseIntegrator
from .interfaces import MEInterface, SolveInterface


class MESolveJumpMonteCarloIntegrator(BaseIntegrator, MEInterface, SolveInterface):
    def run(self) -> Result:
        # modify nmaxclick in the options passed to jssesolve
        jsse_options = replace(self.options, nmaxclick=self.method.jsse_nmaxclick)

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
            jsse_options,
        )

        # compute mean states and expectation values
        ysave = jsse_result.mean_states()
        Esave = jsse_result.mean_expects()

        # compute save_extra if requested
        extra = None
        if self.options.save_extra and self.options.save_states:
            extra = self.options.save_extra(ysave)
        elif self.options.save_extra:
            # this is enforced because `save_extra` is not necessarily a linear
            # function of the states, and thus save_extra(mean(states)) may not
            # be equal to mean(save_extra(states)).
            warnings.warn(
                '`save_extra` is not supported with the `JumpMonteCarlo` method'
                'unless `save_states` is also set to True.',
                stacklevel=2,
            )

        # return
        saved = SolveSaved(ysave=ysave, extra=extra, Esave=Esave)
        return self.result(saved, infos=None)


mesolve_jumpmontecarlo_integrator_constructor = MESolveJumpMonteCarloIntegrator
