import equinox as eqx
from jaxtyping import Scalar

from ...options import Options
from ...qarrays.qarray import QArray
from ...time_qarray import TimeQArray


class OptionsInterface(eqx.Module):
    options: Options


class SEInterface(eqx.Module):
    """Interface for the Schrödinger equation."""

    H: TimeQArray


class MEInterface(eqx.Module):
    """Interface for the Lindblad master equation."""

    H: TimeQArray
    Ls: list[TimeQArray]

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)


class SolveInterface(eqx.Module):
    Es: list[QArray]
