from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from ..gradient import Gradient
from ..options import Options
from ..solver import Solver
from ..time_array import TimeArray


class AbstractSolver(eqx.Module):
    @abstractmethod
    def run(self) -> PyTree:
        pass


class BaseSolver(AbstractSolver):
    ts: Array
    y0: Array
    H: TimeArray
    E: Array
    solver: Solver
    gradient: Gradient | None
    options: Options


SESolver = BaseSolver


class MESolver(BaseSolver):
    Ls: list[TimeArray]
