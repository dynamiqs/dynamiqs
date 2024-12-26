import equinox as eqx
from jax import Array
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


class DSMEInterface(eqx.Module):
    """Interface for the diffusive SME."""

    H: TimeQArray
    Lcs: list[TimeQArray]  # (nLc, n, n)
    Lms: list[TimeQArray]  # (nLm, n, n)
    etas: Array  # (nLm,)

    @property
    def Ls(self) -> list[TimeQArray]:
        return self.Lcs + self.Lms  # (nLc + nLm, n, n)

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)

    def Lm(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Lms]  # (nLm, n, n)


class SolveInterface(eqx.Module):
    Es: QArray
