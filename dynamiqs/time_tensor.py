from __future__ import annotations

from abc import abstractmethod, abstractproperty
from typing import get_args

import torch
from torch import Tensor

from ._utils import obj_type_str, type_str
from .utils.tensor_types import ArrayLike, Number, get_cdtype, to_device, to_tensor

__all__ = ['tt']


def to_time_tensor(
    x: ArrayLike | callable[[float], Tensor],
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> TimeTensor:
    dtype = dtype or get_cdtype(dtype)  # assume complex by default
    device = to_device(device)

    # constant time tensor
    if isinstance(x, get_args(ArrayLike)):
        x = to_tensor(x, dtype=dtype, device=device)
        return ConstantTimeTensor(x)
    # callable time tensor
    elif callable(x):
        f0 = x(0.0)

        # check type, dtype and device match
        if not isinstance(f0, Tensor):
            raise TypeError(
                f'The time-dependent operator must be a {type_str(Tensor)}, but has'
                f' type {obj_type_str(f0)}. The provided callable must return a tensor,'
                ' to avoid costly type conversion at each time solver step.'
            )
        elif f0.dtype != dtype:
            raise TypeError(
                f'The time-dependent operator must have dtype `{dtype}`, but has dtype'
                f' `{f0.dtype}`. The provided callable must return a tensor with the'
                ' same `dtype` as provided to the solver, to avoid costly dtype'
                ' conversion at each solver time step.'
            )
        elif f0.device != device:
            raise TypeError(
                f'The time-dependent operator must be on device `{device}`, but is on'
                f' device `{f0.device}`. The provided callable must return a tensor on'
                ' the same device as provided to the solver, to avoid costly device'
                ' transfer at each solver time step.'
            )

        return CallableTimeTensor(x, f0)
    else:
        raise TypeError(
            'Argument `x` must be an array-like object or a callable with signature'
            f' (t: float) -> Tensor, but has type {obj_type_str(x)}.'
        )


tt = to_time_tensor


class TimeTensor:
    # Subclasses should implement:
    # - the properties: dtype, device, shape
    # - the methods: __call__, view, adjoint, __neg__, __mul__
    # - adapt `time_tensor_add` to support their type (no need to support addition
    #   with all the time tensor types, just support addition with `Tensor`,
    #   `ConstantTimeTensor` and the subclass type itself)

    @abstractproperty
    def dtype(self) -> torch.dtype:
        """Data type."""
        pass

    @abstractproperty
    def device(self) -> torch.device:
        """Device."""
        pass

    @abstractproperty
    def shape(self) -> torch.Size:
        """Shape."""
        pass

    @abstractmethod
    def __call__(self, t: float) -> Tensor:
        """Evaluate at a given time"""
        pass

    @abstractmethod
    def view(self, *shape: int) -> TimeTensor:
        """Returns a new tensor with the same data but of a different shape."""
        pass

    @abstractmethod
    def adjoint(self) -> TimeTensor:
        pass

    @property
    def mH(self) -> TimeTensor:
        return self.adjoint()

    @abstractmethod
    def __neg__(self) -> TimeTensor:
        pass

    @abstractmethod
    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        if not isinstance(other, get_args(Number) + (Tensor,)):
            return NotImplemented

    def __rmul__(self, other: Number | Tensor) -> TimeTensor:
        return self * other

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if not isinstance(other, (Tensor, TimeTensor)):
            return NotImplemented
        return time_tensor_add(self, other)

    def __radd__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if not isinstance(other, (Tensor, TimeTensor)):
            return NotImplemented
        return self + other

    def __sub__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if not isinstance(other, (Tensor, TimeTensor)):
            return NotImplemented
        return self + (-other)

    def __rsub__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if not isinstance(other, (Tensor, TimeTensor)):
            return NotImplemented
        return other + (-self)

    def __repr__(self) -> str:
        return f'<{obj_type_str(self)}>'

    def __str__(self) -> str:
        return self.__repr__()

    def size(self, dim: int) -> int:
        """Size along a given dimension."""
        return self.shape[dim]

    def dim(self) -> int:
        """Get the number of dimensions."""
        return len(self.shape)

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.dim()


class ConstantTimeTensor(TimeTensor):
    def __init__(self, tensor: Tensor):
        self.tensor = tensor

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def shape(self) -> torch.Size:
        return self.tensor.shape

    def __call__(self, t: float) -> Tensor:
        return self.tensor

    def view(self, *shape: int) -> TimeTensor:
        return ConstantTimeTensor(self.tensor.view(*shape))

    def adjoint(self) -> TimeTensor:
        return ConstantTimeTensor(self.tensor.adjoint())

    def __neg__(self) -> TimeTensor:
        return ConstantTimeTensor(-self.tensor)

    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        super().__mul__(other)
        return ConstantTimeTensor(self.tensor * other)


class CallableTimeTensor(TimeTensor):
    def __init__(self, f: callable[[float], Tensor], f0: Tensor):
        # f0 carries all the transformation on the shape
        self.f = f
        self.f0 = f0

    @property
    def dtype(self) -> torch.dtype:
        return self.f0.dtype

    @property
    def device(self) -> torch.device:
        return self.f0.device

    @property
    def shape(self) -> torch.Size:
        return self.f0.shape

    def __call__(self, t: float) -> Tensor:
        return self.f(t).view(self.shape)

    def view(self, *shape: int) -> TimeTensor:
        f = self.f
        f0 = self.f0.view(*shape)
        return CallableTimeTensor(f, f0)

    @abstractmethod
    def adjoint(self) -> TimeTensor:
        f = lambda t: self.f(t).adjoint()
        f0 = self.f0.adjoint()
        return CallableTimeTensor(f, f0)

    def __neg__(self) -> TimeTensor:
        f = lambda t: -self.f(t)
        f0 = -self.f0
        return CallableTimeTensor(f, f0)

    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        super().__mul__(other)
        f = lambda t: self.f(t) * other
        f0 = self.f0 * other
        return CallableTimeTensor(f, f0)


def time_tensor_add(x: TimeTensor, y: Tensor | TimeTensor) -> TimeTensor:
    if isinstance(x, ConstantTimeTensor):
        if isinstance(y, Tensor):
            return ConstantTimeTensor(x.tensor + y)
        elif isinstance(y, ConstantTimeTensor):
            return ConstantTimeTensor(x.tensor + y.tensor)
        elif isinstance(y, CallableTimeTensor):
            f = lambda t: y.f(t) + x.tensor
            f0 = y.f0 + x.tensor
            return CallableTimeTensor(f, f0)
    elif isinstance(x, CallableTimeTensor):
        if isinstance(y, Tensor):
            f = lambda t: x.f(t) + y
            f0 = x.f0 + y
            return CallableTimeTensor(f, f0)
        elif isinstance(y, ConstantTimeTensor):
            f = lambda t: x.f(t) + y.tensor
            f0 = x.f0 + y.tensor
            return CallableTimeTensor(f, f0)
        elif isinstance(y, CallableTimeTensor):
            f = lambda t: x.f(t) + y.f(t)
            f0 = x.f0 + y.f0
            return CallableTimeTensor(f, f0)
    raise TypeError(
        f'Unsupported operand type(s) for +: {obj_type_str(x)} and {obj_type_str(y)}.'
    )
