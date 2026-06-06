"""Implementation of a Schrödinger equation problem and its solving.

Classes
-------

* SESolveProblem
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from jax import Array
from typing_extensions import override

import dynamiqs as dq
from dynamiqs import QArray, TimeQArray
from dynamiqs.result import SESolveResult

from .problem import Problem


@dataclass(frozen=True, slots=True)
class SESolveProblem(Problem, ABC):
    """General structure of a Schrödinger equation problem."""

    @override
    @abstractmethod
    def build(self) -> tuple[TimeQArray, QArray, Array]:
        pass

    @override
    def solve(self) -> SESolveResult:
        hamiltonian, initial_state, tsave = self.build()

        return dq.sesolve(hamiltonian, initial_state, tsave, progress_meter=False)


__all__ = ['SESolveProblem']
