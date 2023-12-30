from __future__ import annotations

from abc import abstractmethod

from jax import Array

from .types import Scalar


class TimeArray:
    @abstractmethod
    def __call__(self, t: Scalar) -> Array:
        pass


class ConstantTimeArray(TimeArray):
    def __init__(self, array: Array):
        self.array = array

    def __call__(self, t: Scalar) -> Array:
        return self.array


class CallableTimeArray(TimeArray):
    def __init__(self, f: callable[[Scalar], Array]):
        self.f = f

    def __call__(self, t: Scalar) -> Array:
        return self.f(t)
