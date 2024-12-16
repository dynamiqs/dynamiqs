import equinox as eqx
from jaxtyping import Scalar
from jax import Array
from optimistix import AbstractRootFinder

from ...options import Options
from ...qarrays.qarray import QArray
from ...time_array import TimeArray


class OptionsInterface(eqx.Module):
    options: Options


class SEInterface(eqx.Module):
    """Interface for the Schrödinger equation."""

    H: TimeArray


class _MInterface(eqx.Module):

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)


class MEInterface(_MInterface):
    """Interface for the Lindblad master equation."""

    H: TimeArray
    Ls: list[TimeArray]


class MCInterface(_MInterface):
    """Interface for the Monte-Carlo jump unraveling of the master equation."""

    H: TimeArray
    Ls: list[TimeArray]
    keys: Array
    root_finder: AbstractRootFinder | None


class SolveInterface(eqx.Module):
    Es: QArray
