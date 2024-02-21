from __future__ import annotations

from typing import NamedTuple, Optional

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from .gradient import Gradient
from .options import Options
from .solver import Solver


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


Saved = NamedTuple('Saved', ysave=Array, Esave=Optional[Array], extra=Optional[PyTree])


class Result(eqx.Module):
    tsave: Array
    solver: Solver
    gradient: Gradient | None
    options: Options
    _saved: Saved

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
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        parts_str = '\n'.join(f'{k}: {v}' for k, v in parts.items())
        return '==== Result ====\n' + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError
