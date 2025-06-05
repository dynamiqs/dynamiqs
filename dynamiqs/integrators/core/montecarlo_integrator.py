import warnings
from dataclasses import replace

import jax.numpy as jnp

from ...result import Result, SolveSaved
from ..apis.jssesolve import _vectorized_jssesolve
from .abstract_integrator import BaseIntegrator
from .interfaces import MEInterface, SolveInterface


class MESolveJumpMonteCarloIntegrator(BaseIntegrator, MEInterface, SolveInterface):
    def __post_init__(self):
        # check that y0 is a ket
        if not self.y0.isket():
            raise ValueError(
                'For the `JumpMonteCarlo` method, `rho0` must be a ket, '
                f'but has shape {self.y0.shape}.'
            )

    def run(self) -> Result:
        # modify nmaxclick in the options passed to jssesolve
        jsse_options = replace(
            self.options, nmaxclick=self.method.jsse_nmaxclick, save_extra=None
        )

        keys = jnp.asarray(self.method.keys)

        # call _vectorized_jssesolve to compute the jump SSE results
        jsse_result = _vectorized_jssesolve(
            self.H,
            self.Ls,
            self.y0,
            self.ts,
            keys,
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
        if self.options.save_extra:
            if self.options.save_states:
                extra = self.options.save_extra(ysave)
            else:
                # this is enforced because `save_extra` is not necessarily a linear
                # function of the states, and thus save_extra(mean(states)) may not
                # be equal to mean(save_extra(states)).
                warnings.warn(
                    'The `save_extra` option is not supported with the `JumpMonteCarlo`'
                    ' method unless the `save_states` option is also set to True.'
                    ' Falling back to extra=None.',
                    stacklevel=2,
                )

        # return
        saved = SolveSaved(ysave=ysave, extra=extra, Esave=Esave)
        return self.result(saved, infos=jsse_result.infos)


mesolve_jumpmontecarlo_integrator_constructor = MESolveJumpMonteCarloIntegrator
