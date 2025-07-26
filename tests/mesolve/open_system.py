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


class OpenSystemMESolve(OpenSystem):
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
        return dq.mesolve(
            H,
            Ls,
            y0,
            self.tsave,
            exp_ops=Es,
            method=method,
            gradient=gradient,
            options=options,
        )


class OCavityMESolve(OCavity, OpenSystemMESolve):
    pass


class OTDQubitMESolve(OTDQubit, OpenSystemMESolve):
    pass


# # we choose `t_end` not coinciding with a full period (`t_end=1.0`) to avoid null
# # gradients
Hz = 2 * jnp.pi
tsave = np.linspace(0.0, 0.3, 11)
dense_ocavity = OCavityMESolve(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, tsave=tsave, layout=dense
)
dia_ocavity = OCavityMESolve(
    n=8, delta=1.0 * Hz, alpha0=0.5, kappa=1.0 * Hz, tsave=tsave, layout=dq.dia
)

tsave = np.linspace(0.0, 1.0, 11)
otdqubit = OTDQubitMESolve(eps=3.0, omega=10.0, gamma=1.0, tsave=tsave)
