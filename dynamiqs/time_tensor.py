from __future__ import annotations

from abc import abstractmethod, abstractproperty
from typing import get_args

import torch
from torch import Tensor

from ._utils import obj_type_str, type_str
from .solvers.utils.utils import cache
from .utils.tensor_types import (
    ArrayLike,
    Number,
    dtype_complex_to_real,
    get_cdtype,
    to_device,
    to_tensor,
)

__all__ = ['totime']


def totime(
    x: ArrayLike | callable[[float], Tensor],
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> TimeTensor:
    dtype = dtype or get_cdtype(dtype)  # assume complex by default
    device = to_device(device)

    # pwc time tensor
    if isinstance(x, tuple) and len(x) == 3:
        times, values, tensor = x
        if dtype in (torch.complex64, torch.complex128):
            rdtype = dtype_complex_to_real(dtype)
        else:
            rdtype = dtype
        times = to_tensor(times, dtype=rdtype, device=device)
        values = to_tensor(values, dtype=dtype, device=device)
        tensor = to_tensor(tensor, dtype=dtype, device=device)
        pwc = _PWCTimeTensor(times, values, tensor)
        return PWCTimeTensor([pwc])
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


class TimeTensor:
    # Subclasses should implement:
    # - the properties: dtype, device, shape
    # - the methods: __call__, view, adjoint, __neg__, __mul__, __add__

    # Special care should be taken when implementing `__call__` for caching to work
    # properly. The `@cache` decorator checks the tensor `__hash__`, which is
    # implemented as its address in memory. Thus, when two consecutive calls to a
    # `TimeTensor` should return a tensor with the same values, these two tensors must
    # not only be equal, they should be the same object in memory.

    # Note that a subclass implementation of `__add__` only need to support addition
    # with `Tensor`, `ConstantTimeTensor` and the subclass type itself.

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
        pass

    def __rmul__(self, other: Number | Tensor) -> TimeTensor:
        return self * other

    @abstractmethod
    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        pass

    def __radd__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return self + other

    def __sub__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return self + (-other)

    def __rsub__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return other + (-self)

    def __repr__(self) -> str:
        return str(type(self).__name__)

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
        return ConstantTimeTensor(self.tensor * other)

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if isinstance(other, Tensor):
            return ConstantTimeTensor(self.tensor + other)
        elif isinstance(other, ConstantTimeTensor):
            return self + other.tensor
        else:
            return NotImplemented


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

    @cache
    def __call__(self, t: float) -> Tensor:
        # cached if called twice with the same time, otherwise we recompute `f(t)`
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
        f = lambda t: self.f(t) * other
        f0 = self.f0 * other
        return CallableTimeTensor(f, f0)

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if isinstance(other, Tensor):
            f = lambda t: self.f(t) + other
            f0 = self.f0 + other
            return CallableTimeTensor(f, f0)
        elif isinstance(other, ConstantTimeTensor):
            return self + other.tensor
        elif isinstance(other, CallableTimeTensor):
            f = lambda t: self.f(t) + other.f(t)
            f0 = self.f0 + other.f0
            return CallableTimeTensor(f, f0)
        else:
            return NotImplemented


class _PWCTimeTensor(TimeTensor):
    # Crucially, this implementation of a PWC tensor doesn't handle addition with a
    # constant tensor or another PWC tensor. To support this we need additional
    # structure, which is achieved by the `PWCTimeTensor` class below. This class is
    # just an intermediate helper class.

    # Supports batching on `values`:
    # >>> times = torch.linspace(0, 1.0, 11)  # (nt+1) with nt = 10
    # >>> nseeds = 30
    # >>> eps = dq.rand_complex((10, nseeds))  # (nt, 30)
    # >>> a = dq.destroy(3)  #  (3, 3)
    # >>> H = PWCTimeTensor(times, values, tensor)
    # >>> H(0.0)  # (30, 3, 3)

    def __init__(self, times: Tensor, values: Tensor, tensor: Tensor) -> None:
        # values carries all the transformation on the shape
        self.times = times  # (nt+1)
        self.values = values  # (nt, ...v)
        self.tensor = tensor  # (n, n)

        self.nt = self.values.shape[0]
        self.n = tensor.shape[-1]

    @property
    def dtype(self) -> torch.dtype:
        return self.tensor.dtype

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @property
    def shape(self) -> torch.Size:
        return torch.Size((*self.values.shape[1:], self.n, self.n))  # (...v, n, n)

    def __call__(self, t: float) -> Tensor:
        if t < self.times[0] or t >= self.times[-1]:
            return torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        else:
            # find the index $k$ such that $t \in [t_k, t_{k+1})$
            idx = torch.searchsorted(self.times, t, side='right') - 1
            v = self.values[idx, ...]  # (...v)
            v = v.view(*v.shape, 1, 1)  # (...v, n, n)
            return v * self.tensor  # (...v, n, n)

    def view(self, *shape: int) -> TimeTensor:
        values = self.values.view(self.nt, *shape[:-2])
        return _PWCTimeTensor(self.times, values, self.tensor)

    def adjoint(self) -> TimeTensor:
        return _PWCTimeTensor(self.times, self.values.conj(), self.tensor.mH)

    def __neg__(self) -> TimeTensor:
        return _PWCTimeTensor(self.times, self.values, -self.tensor)

    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        return _PWCTimeTensor(self.times, self.values, self.tensor * other)

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        return NotImplemented


class PWCTimeTensor(TimeTensor):
    def __init__(
        self,
        pwc: list[_PWCTimeTensor],
        static: Tensor | None = None,  # constant part
    ):
        # argument pwc must be a non-empty list of compatible shapes values and tensor
        self.pwc = pwc
        self.static = torch.zeros_like(self.pwc[0].tensor) if static is None else static

        # merge times tensors
        times = [x.times for x in pwc]
        self.times = torch.cat(times).unique().sort()[0]

    @property
    def dtype(self) -> torch.dtype:
        return self.pwc[0].dtype

    @property
    def device(self) -> torch.device:
        return self.pwc[0].device

    @property
    def shape(self) -> torch.Size:
        return self.pwc[0].shape

    def __call__(self, t: float) -> Tensor:
        static_part = self.static.expand(*self.shape)
        pwc_part = torch.stack([pwc(t) for pwc in self.pwc]).sum(0)
        return static_part + pwc_part

    def view(self, *shape: int) -> TimeTensor:
        pwc = [p.view(*shape) for p in self.pwc]
        return PWCTimeTensor(pwc, static=self.static)

    def adjoint(self) -> TimeTensor:
        static = self.static.mH
        pwc = [p.adjoint() for p in self.pwc]
        return PWCTimeTensor(pwc, static=static)

    def __neg__(self) -> TimeTensor:
        static = -self.static
        pwc = [-p for p in self.pwc]
        return PWCTimeTensor(pwc, static=static)

    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        static = self.static * other
        pwc = [p * other for p in self.pwc]
        return PWCTimeTensor(pwc, static=static)

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if isinstance(other, Tensor):
            static = self.static + other
            return PWCTimeTensor(self.pwc, static=static)
        elif isinstance(other, ConstantTimeTensor):
            return self + other.tensor
        elif isinstance(other, PWCTimeTensor):
            static = self.static + other.static
            pwc = [*self.pwc, *other.pwc]
            return PWCTimeTensor(pwc, static=static)
        else:
            return NotImplemented
