from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jaxtyping import PyTree

import dynamiqs as dq
from dynamiqs import dense
from dynamiqs.gradient import Gradient
from dynamiqs.method import Method
from dynamiqs.options import Options
from dynamiqs.result import Result

from ..system import OCavity, OpenSystem, OTDQubit


class OpenSystemSteadyState(OpenSystem):
    def run(
        self,
        method: Method,
        *,
        gradient: Gradient | None = None,
        options: Options = Options(),  # noqa: B008
        params: PyTree | None = None,
    ) -> Result:
        params = self.params_default if params is None else params
        H = self.H(params)
        Ls = self.Ls(params)
        y0 = self.y0(params)
        Es = self.Es(params)
        return dq.steadystate(
            H, Ls, y0, exp_ops=Es, method=method, gradient=gradient, options=options
        )


class OCavitySteadyState(OCavity, OpenSystemSteadyState):
    def __init__(
        self, *, n: int, delta: float, alpha0: float, kappa: float, layout: dq.Layout
    ):
        super().__init__(
            n=n,
            delta=delta,
            alpha0=alpha0,
            kappa=kappa,
            tsave=np.array([jnp.inf]),
            layout=layout,
        )


class OTDQubitSteadyState(OTDQubit, OpenSystemSteadyState):
    def __init__(self, *, eps: float, omega: float, gamma: float):
        super().__init__(eps=eps, omega=omega, gamma=gamma, tsave=np.array([jnp.inf]))


# # we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# # gradients
Hz = 2 * jnp.pi
dense_ocavity = OCavitySteadyState(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, layout=dense
)
dia_ocavity = OCavitySteadyState(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, layout=dq.dia
)

otdqubit = OTDQubitSteadyState(eps=3.0, omega=10.0, gamma=1.0)

# steady state solutions
dense_ocavity_steady = OCavitySteadyState(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, layout=dense
)
