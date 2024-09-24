import equinox as eqx

from ...options import Options
from ...qarrays.qarray import QArray
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


class SolveInterface(eqx.Module):
    Es: QArray
