from __future__ import annotations

from typing import Any

import diffrax as dx
from jax import numpy as jnp
from jaxtyping import ArrayLike

from .._utils import bexpect, merge_complex, split_complex
from ..gradient import Adjoint, Autograd, Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Solver, _stepsize_controller
from ..time_array import totime


def sesolve(
    H: ArrayLike,
    psi0: ArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: Solver | None = None,
    gradient: Gradient | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    # === default solver
    if solver is None:
        solver = Dopri5()

    # === options
    options = Options(solver=solver, gradient=gradient, options=options)

    # === solver class
    solvers = {Dopri5: dx.Dopri5}
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]

    # === gradient class
    gradients = {
        Autograd: dx.RecursiveCheckpointAdjoint,
        Adjoint: dx.BacksolveAdjoint,
    }
    if gradient is None:
        pass
    elif not isinstance(gradient, tuple(gradients.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in gradients.keys())
        raise ValueError(
            f'Gradient of type `{type(gradient).__name__}` is not supported'
            f' (supported gradient types: {supported_str}).'
        )
    elif not solver.supports_gradient(gradient):
        support_str = ', '.join(f'`{x.__name__}`' for x in solver.SUPPORTED_GRADIENT)
        raise ValueError(
            f'Solver `{type(solver).__name__}` does not support gradient'
            f' `{type(gradient).__name__}` (supported gradient types: {support_str}).'
        )

    if gradient is not None:
        gradient_class = gradients[type(gradient)]
    else:
        gradient_class = None

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    H = totime(H)

    def f(t, psi, _args):
        psi = merge_complex(psi)
        res = -1j * H(t) @ psi
        res = split_complex(res)
        return res

    def save(_t, psi, _args):
        res = {}
        if options.save_states:
            res['states'] = psi

        psi = merge_complex(psi)
        # TODO : use vmap ?
        res['expects'] = tuple([split_complex(bexpect(op, psi)) for op in exp_ops])
        return res

    solution = dx.diffeqsolve(
        dx.ODETerm(f),
        solver_class(),
        t0=tsave[0],
        t1=tsave[-1],
        dt0=dt,
        y0=split_complex(psi0),
        saveat=dx.SaveAt(ts=tsave, fn=save),
        stepsize_controller=stepsize_controller,
        adjoint=(
            gradient_class()
            if gradient_class is not None
            else dx.RecursiveCheckpointAdjoint()
        ),
    )

    ysave = None
    if options.save_states:
        ysave = solution.ys['states']
        ysave = merge_complex(ysave)

    Esave = None
    if len(exp_ops) > 0:
        Esave = solution.ys['expects']
        Esave = jnp.stack(Esave, axis=0)
        Esave = merge_complex(Esave)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
