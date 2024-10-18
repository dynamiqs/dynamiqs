import equinox as eqx
from jax import Array

from ...options import Options
from ...time_array import TimeArray


class OptionsInterface(eqx.Module):
    options: Options


class SEInterface(eqx.Module):
    """Interface for the Schrödinger equation."""

    H: TimeArray


class MEInterface(eqx.Module):
    """Interface for the Lindblad master equation."""

    H: TimeArray
    Ls: list[TimeArray]


class SMEInterface(eqx.Module):
    """Interface for the diffusive SME."""

    H: TimeArray
    Lcs: list[TimeArray]  # (nLc, n, n)
    Lms: list[TimeArray]  # (nLm, n, n)
    etas: Array  # (nLm,)

    @property
    def Ls(self) -> list[TimeArray]:
        return self.Lcs + self.Lms  # (nLc + nLm, n, n)


class SolveInterface(eqx.Module):
    Es: Array
