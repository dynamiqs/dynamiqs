import warnings
from abc import abstractmethod
from dataclasses import replace

import jax.numpy as jnp

from ...result import Result, SolveSaved, StochasticSolveResult
from ..apis.dssesolve import _vectorized_dssesolve
from ..apis.jssesolve import _vectorized_jssesolve
from .abstract_integrator import BaseIntegrator
from .interfaces import MEInterface, SolveInterface


class MESolveMonteCarloIntegrator(BaseIntegrator, MEInterface, SolveInterface):
    def __post_init__(self):
        # check that y0 is a ket
        if not self.y0.isket():
            raise ValueError(
                'For the Monte Carlo methods, `rho0` must be a ket, '
                f'but has shape {self.y0.shape}.'
            )

    def run(self) -> Result:
        stochastic_result = self._run_stochastic()

        # compute mean states and expectation values
        ysave = stochastic_result.mean_states()
        Esave = stochastic_result.mean_expects()

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
                    'The `save_extra` option is not supported with the Monte Carlo'
                    ' methods unless the `save_states` option is also set to True.'
                    ' Falling back to extra=None.',
                    stacklevel=2,
                )

        # return
        saved = SolveSaved(ysave=ysave, extra=extra, Esave=Esave)
        return self.result(saved, infos=stochastic_result.infos)

    @abstractmethod
    def _run_stochastic(self) -> StochasticSolveResult:
        pass


class MESolveJumpMonteCarloIntegrator(MESolveMonteCarloIntegrator):
    def _run_stochastic(self) -> StochasticSolveResult:
        # modify nmaxclick in the options passed to jssesolve
        jsse_options = replace(
            self.options, nmaxclick=self.method.jsse_nmaxclick, save_extra=None
        )

        keys = jnp.asarray(self.method.keys)

        # call _vectorized_jssesolve to compute the jump SSE results
        return _vectorized_jssesolve(
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


mesolve_jumpmontecarlo_integrator_constructor = MESolveJumpMonteCarloIntegrator


class MESolveDiffusiveMonteCarloIntegrator(MESolveMonteCarloIntegrator):
    def _run_stochastic(self) -> StochasticSolveResult:
        keys = jnp.asarray(self.method.keys)

        # call _vectorized_jssesolve to compute the diffusive SSE results
        return _vectorized_dssesolve(
            self.H,
            self.Ls,
            self.y0,
            self.ts,
            keys,
            self.Es,
            self.method.dsse_method,
            self.gradient,
            self.options,
        )


mesolve_diffusivemontecarlo_integrator_constructor = (
    MESolveDiffusiveMonteCarloIntegrator
)
