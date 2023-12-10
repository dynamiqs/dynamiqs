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
    x: ArrayLike | callable[[float], Tensor] | tuple[ArrayLike, ArrayLike, ArrayLike],
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
        values = values.unsqueeze(0)  # npwc = 1
        tensor = tensor.unsqueeze(0)  # npwc = 1
        return PWCTimeTensor(times, values, tensor)
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


class PWCTimeTensor(TimeTensor):
    # Arbitrary sum of PWC tensors.
    #
    # A single PWC tensor is defined by a tuple of 3 tensors (times, values, tensor),
    # where
    # - times: (nv+1) are the time points between which the PWC tensor take constant
    #          values, where nv is the number of time intervals
    # - values: (nv, ...) are the constant complex values for each time interval, where
    #           (...) is an arbitrary batching size
    # - tensor: (n, n) is the tensor value
    #
    # To support additions of PWC tensors with non-aligned `times` values we augment
    # the `values` and `tensor` objects with one dimension `npwc`, which is the number
    # of summed PWC tensors. Upon addition between two PWC tensors, we merge the
    # `times` and `values` tensors appropriately, and concatenate the `tensor` objects.
    # In addition, we keep track of a `static` tensor for addition with a constant
    # tensor.

    def __init__(
        self,
        times: Tensor,
        values: Tensor,
        tensors: Tensor,
        static: Tensor | None = None,  # constant part
    ):
        # Dimensions:
        # - nv: number of time intervals
        # - ...: values batching (`self.values` carries all the shape transformation)
        # - npwc: number of piecewise constant tensors
        # - n: Hilbert space dimension

        self.times = times  # (nv+1)

        self.values = values  # (npwc, nv, ...)
        self.npwc = values.shape[0]
        self.nv = values.shape[1]
        self.dots = values.shape[2:]  # (...)

        self.tensors = tensors  # (npwc, n, n)
        self.n = tensors.shape[-1]

        if static is None:
            self.static = torch.zeros(
                self.n, self.n, dtype=self.dtype, device=self.device
            )
        else:
            self.static = static

    @property
    def dtype(self) -> torch.dtype:
        return self.tensors.dtype

    @property
    def device(self) -> torch.device:
        return self.tensors.device

    @property
    def shape(self) -> torch.Size:
        return torch.Size((*self.dots, self.n, self.n))  # (..., n, n)

    def __call__(self, t: float) -> Tensor:
        static = self.static.expand(*self.shape)  # (..., n, n)

        if t < self.times[0] or t >= self.times[-1]:
            return static  # (..., n, n)
        else:
            # find the index $k$ such that $t \in [t_k, t_{k+1})$
            idx = torch.searchsorted(self.times, t, side='right') - 1
            v = self.values[:, idx, ...]  # (npwc, ...)
            v = v.permute(*range(1, v.ndim), 0)  # (..., npwc)
            v = v.view(*v.shape, 1, 1)  # (..., npwc, n, n)
            return (v * self.tensors).sum(-3) + static  # (..., n, n)

    def view(self, *shape: int) -> TimeTensor:
        # shape: (..., n, n)
        values = self.values.view(self.npwc, self.nv, *shape[:-2])
        return PWCTimeTensor(self.times, values, self.tensors, static=self.static)

    def adjoint(self) -> TimeTensor:
        return PWCTimeTensor(
            self.times, self.values.conj(), self.tensors.mH, static=self.static.mH
        )

    def __neg__(self) -> TimeTensor:
        return PWCTimeTensor(
            self.times, self.values, -self.tensors, static=-self.static
        )

    def __mul__(self, other: Number | Tensor) -> TimeTensor:
        return PWCTimeTensor(
            self.times, self.values, self.tensors * other, static=self.static * other
        )

    def __add__(self, other: Tensor | TimeTensor) -> TimeTensor:
        if isinstance(other, Tensor):
            static = self.static + other
            return PWCTimeTensor(self.times, self.values, self.tensors, static=static)
        elif isinstance(other, ConstantTimeTensor):
            return self + other.tensor
        elif isinstance(other, PWCTimeTensor):
            # merge times and values -> times: (nv+1), values: (npwc1 + npwc2, nv, ...)
            t1, t2 = self.times, other.times
            v1, v2 = self.values, other.values
            times, values = _merge_pwc_times_values(t1, t2, v1, v2)

            # merge tensors -> tensors: (npwc1 + npwc2, n, n)
            tensors = torch.cat((self.tensors, other.tensors))

            # merge static part -> static: (n, n)
            static = self.static + other.static

            return PWCTimeTensor(times, values, tensors, static=static)
        else:
            return NotImplemented


def _merge_pwc_times_values(t1, t2, v1, v2):
    # t1: (nv1+1)
    # t2: (nv2+1)
    # v1: (npwc1, nv1, ...)
    # v2: (npwc1, nv2, ...)
    # --> (nv+1), (npwc1 + npwc2, nv, ...)

    # Example with (...) = () and npwc1 = npwc2 = 1:
    # t1 = [0, 10, 20]
    # v1 = [[1, 2]]
    # t2 = [0, 5, 10, 30]
    # v2 = [[1, 2, 3]]
    # times: 0     5     10        20        30 = [0, 5, 10, 20, 30]
    #        |-----|-----|---------|---------|
    # v1n:   |  1  |  1  |    2    |    0    |  = [1, 1, 2, 0]
    # v2n:   |  1  |  2  |    3    |    3    |  = [1, 2, 3, 3]
    # t = [0, 5, 10, 20, 30]
    # v = [[1, 1, 2, 0], [1, 2, 3, 3]]

    # merge times
    t = torch.cat((t1, t2)).sort()[0].unique(sorted=True)  # (nv+1)

    # shapes
    nv1 = len(t1) - 1
    nv2 = len(t2) - 1
    nv = len(t) - 1
    dots = v1.shape[2:]  # (...)

    # format v1 and v2 for broadcasting
    v1 = v1[:, :, None, ...]  # (npwc1, nv1, 1, ...)
    v2 = v2[:, :, None, ...]  # (npwc2, nv2, 1, ...)

    # for each interval of t1, find whether it overlaps with each interval of t
    mask1 = (t1[:-1, None] <= t[:-1]) & (t1[1:, None] > t[:-1])  # (nv1, nv)
    mask1 = mask1.view(1, nv1, nv, *[1 for _ in dots])  # (1, nv1, nv, ...)

    # for each interval of t2, find whether it overlaps with each interval of t
    mask2 = (t2[:-1, None] <= t[:-1]) & (t2[1:, None] > t[:-1])  # (nv2, nv)
    mask2 = mask2.view(1, nv2, nv, *[1 for _ in dots])  # (1, nv2, nv, ...)

    # apply masks
    zero = torch.tensor(0.0)
    v1n = torch.where(mask1, v1, zero).sum(dim=1)  # (npwc1, nv, ...)
    v2n = torch.where(mask2, v2, zero).sum(dim=1)  # (npwc2, nv, ...)

    # concatenate results
    v = torch.cat((v1n, v2n))  # (npwc1 + npwc2, nv, ...)

    return t, v
