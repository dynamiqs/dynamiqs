from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import Scalar

from ..._utils import _concatenate_sort
from ...options import Options
from ...qarrays.qarray import QArray
from ...time_qarray import TimeQArray


class OptionsInterface(eqx.Module):
    options: Options


class TimeInterface(eqx.Module):
    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array | None:
        pass


class SEInterface(TimeInterface):
    """Interface for the Schrödinger equation."""

    H: TimeQArray

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.H.discontinuity_ts


class MEInterface(TimeInterface):
    """Interface for the Lindblad master equation."""

    H: TimeQArray
    Ls: list[TimeQArray]

    def L(self, t: Scalar) -> list[QArray]:
        return [_L(t) for _L in self.Ls]  # (nLs, n, n)

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [x.discontinuity_ts for x in [self.H, *self.Ls]]
        return _concatenate_sort(*ts)


class SolveInterface(eqx.Module):
    Es: QArray
