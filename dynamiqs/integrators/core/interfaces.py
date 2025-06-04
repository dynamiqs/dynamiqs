from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import Scalar

from ..._utils import concatenate_sort
from ...options import Options
from ...qarrays.qarray import QArray
from ...time_qarray import TimeQArray


class OptionsInterface(eqx.Module):
    options: Options


class AbstractTimeInterface(eqx.Module):
    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array:
        pass


class SEInterface(AbstractTimeInterface):
    """Interface for the Schrödinger equation."""

    H: TimeQArray

    @property
    def discontinuity_ts(self) -> Array:
        return self.H.discontinuity_ts


class MEInterface(AbstractTimeInterface):
    """Interface for the Lindblad master equation."""

    H: TimeQArray
    Ls: list[TimeQArray]

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)

    @property
    def discontinuity_ts(self) -> Array:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return concatenate_sort(*ts)


class JSSEInterface(MEInterface):
    """Interface for the jump SSE."""


class DSSEInterface(MEInterface):
    """Interface for the diffusive SSE."""


class SMEInterface(AbstractTimeInterface):
    """Interface for the jump or diffusive SME."""

    H: TimeQArray
    Lcs: list[TimeQArray]  # (nLc, n, n)
    Lms: list[TimeQArray]  # (nLm, n, n)

    @property
    def Ls(self) -> list[TimeQArray]:
        return self.Lcs + self.Lms  # (nLc + nLm, n, n)

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)

    def Lc(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Lcs]  # (nLc, n, n)

    def Lm(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Lms]  # (nLm, n, n)

    @property
    def discontinuity_ts(self) -> Array:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return concatenate_sort(*ts)


class JSMEInterface(SMEInterface):
    """Interface for the jump SME."""

    thetas: Array  # (nLm,)
    etas: Array  # (nLm,)


class DSMEInterface(SMEInterface):
    """Interface for the diffusive SME."""

    etas: Array  # (nLm,)


class SolveInterface(eqx.Module):
    Es: list[QArray] | None
