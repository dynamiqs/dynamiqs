from __future__ import annotations

from typing import NamedTuple

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from .gradient import Gradient
from .options import Options
from .solver import Solver

__all__ = ['Result']


def memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def memory_str(x: Array) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.2f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.2f} Mb'
    else:
        return f'{mem / 1024**3:.2f} Gb'


def array_str(x: Array) -> str:
    return f'Array {x.dtype} {tuple(x.shape)} | {memory_str(x)}'


class Saved(NamedTuple):
    ysave: Array
    Esave: Array | None
    extra: PyTree | None


class Result(eqx.Module):
    """Result of the integration.

    Attributes:
        states _(Array)_: Saved states.
        expects _(Array, optional)_: Saved expectation values.
        extra _(PyTree, optional)_: Extra data saved.
        infos _(PyTree, optional)_: Solver-dependent information on the resolution.
        tsave _(Array)_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
        final_time _(Array)_: final solution time
    """

    tsave: Array
    solver: Solver
    gradient: Gradient | None
    options: Options
    _saved: Saved
    final_time: Array
    infos: PyTree | None = None

    @property
    def states(self) -> Array:
        return self._saved.ysave

    @property
    def expects(self) -> Array | None:
        return self._saved.Esave

    @property
    def extra(self) -> PyTree | None:
        return self._saved.extra

    def __str__(self) -> str:
        parts = {
            'Solver  ': type(self.solver).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'States  ': array_str(self.states),
            'Expects ': array_str(self.expects) if self.expects is not None else None,
            'Extra   ': (
                eqx.tree_pformat(self.extra) if self.extra is not None else None
            ),
            'Infos   ': self.infos if self.infos is not None else None,
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        parts_str = '\n'.join(f'{k}: {v}' for k, v in parts.items())
        return '==== Result ====\n' + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError
