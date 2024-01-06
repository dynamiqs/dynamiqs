from __future__ import annotations

from collections import namedtuple
from typing import Any

import diffrax as dx
from jax import numpy as jnp
from jaxtyping import ArrayLike

from .._utils import (
    bexpect,
    merge_complex,
    split_complex,
    _get_solver_class,
    _get_gradient_class,
)
from ..gradient import Gradient
from ..options import Options
from ..result import Result
from ..solver import Dopri5, Solver, _stepsize_controller
from ..time_array import (
    totime,
    ConstantTimeArray,
    CallableTimeArray,
    PWCTimeArray,
    ModulatedTimeArray,
)

Cache = namedtuple('Cache', ['H'])


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
    solver_class = _get_solver_class(solver, solvers)

    # === gradient class
    gradient_class = _get_gradient_class(gradient, solver)

    # === stepsize controller
    stepsize_controller, dt = _stepsize_controller(solver)

    # === solve differential equation with diffrax
    H = totime(H)

    def save(_t, psi, _args):
        res = {}
        if options.save_states:
            res['states'] = psi

        psi = merge_complex(psi)
        # TODO : use vmap ?
        if exp_ops is not None:
            res['expects'] = tuple([split_complex(bexpect(op, psi)) for op in exp_ops])
        return res

    def f(cache: Cache, psi):
        return -1j * cache.H @ psi

    if isinstance(H, ConstantTimeArray):
        cache = Cache(H=H(tsave[0]))

        def f_constant(_t, psi, _args):
            psi = merge_complex(psi)
            res = f(cache, psi)
            res = split_complex(res)
            return res

        ode_term = dx.ODETerm(f_constant)

    elif isinstance(H, (CallableTimeArray, ModulatedTimeArray)):

        def f_time_dependent(t, psi, _args):
            psi = merge_complex(psi)
            cache = Cache(H=H(t))
            res = f(cache, psi)
            res = split_complex(res)
            return res

        ode_term = dx.ODETerm(f_time_dependent)
    elif isinstance(H, PWCTimeArray):
        pass
    else:
        raise NotImplementedError(
            "H must be either a ConstantTimeArray, a PWCTimeArray, a ModulatedTimeArray"
            " or a CallableTimeArray"
        )

    if isinstance(H, (CallableTimeArray, ModulatedTimeArray, ConstantTimeArray)):
        solution = dx.diffeqsolve(
            ode_term,
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
    else:
        raise NotImplementedError

    ysave = None
    if options.save_states:
        ysave = solution.ys['states']
        ysave = merge_complex(ysave)

    Esave = None
    if exp_ops is not None and len(exp_ops) > 0:
        Esave = solution.ys['expects']
        Esave = jnp.stack(Esave, axis=0)
        Esave = merge_complex(Esave)

    return Result(
        options,
        ysave=ysave,
        Esave=Esave,
        tsave=solution.ts,
    )
