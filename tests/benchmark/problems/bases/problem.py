"""Generic definition of a solvable problem."""

from abc import ABC, abstractmethod
from collections.abc import Collection
from dataclasses import dataclass
from typing import Any, ClassVar

from dynamiqs.result import SolveResult


@dataclass(frozen=True, slots=True)
class Problem(ABC):
    """General structure of a benchmarkable problem handled by a solver.

    Problem can be any type of mathematical problem that can be solved. In the context
    of the benchmark, such examples could be Schrodinger and Linbad equations.
    """

    problem_name: ClassVar[str]

    def get_name(self) -> str:
        """Return the name of the problem.

        The name is expected to be in snakecase. This is to make the name
        conventional inside benchmarking result tables.

        Returns:
            A snakecase version of the problem name.
        """
        return self.__class__.problem_name

    @abstractmethod
    def build(self) -> Collection[Any]:
        """Build the parameters of the problem to solve.

        More information about the parameters of the problem should be
        specified on its documentation.

        Returns:
            An arbitrary array of parameters representing the problem.
        """

    @abstractmethod
    def solve(self) -> SolveResult:
        """Solve the problem and return the result.

        Returns:
            The solution found by the problem's solver.
        """


__all__ = ['Problem']
