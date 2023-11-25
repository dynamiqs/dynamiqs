from __future__ import annotations

from abc import ABC, abstractmethod
from typing import get_args

import torch
from torch import Tensor

from ..._utils import obj_type_str, to_device, type_str
from ...utils.tensor_types import ArrayLike, TDArrayLike, get_rdtype, to_tensor
from .utils import cache


def to_td_tensor(
    x: TDArrayLike,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> TDTensor:
    """Convert a `TDArrayLike` object to a `TDTensor` object."""
    device = to_device(device)

    if isinstance(x, get_args(ArrayLike)):  # constant tensor
        x = to_tensor(x, dtype=dtype, device=device)
        return ConstantTDTensor(x)
    elif callable(x):  # time-dependent tensor
        dtype = get_rdtype(dtype) if dtype is None else dtype  # assume real by default
        return CallableTDTensor(x, f0=x(0.0), dtype=dtype, device=device)


class TDTensor(ABC):
    @property
    @abstractmethod
    def is_constant(self) -> bool:
        """Whether the tensor is constant in time."""
        pass

    @abstractmethod
    def __call__(self, t: float) -> Tensor:
        """Evaluate at a given time"""
        pass

    @abstractmethod
    def size(self, dim: int) -> int:
        """Size along a given dimension."""
        pass

    @abstractmethod
    def dim(self) -> int:
        """Get the number of dimensions."""
        pass

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self.dim()

    @property
    @abstractmethod
    def shape(self) -> torch.Size:
        """Shape."""
        pass

    @abstractmethod
    def view(self, *shape: int) -> TDTensor:
        """Returns a new tensor with the same data but of a different shape."""
        pass


class ConstantTDTensor(TDTensor):
    def __init__(self, tensor: Tensor):
        self._tensor = tensor
        self.dtype = tensor.dtype
        self.device = tensor.device

    @property
    def is_constant(self) -> bool:
        return True

    def __call__(self, t: float) -> Tensor:
        return self._tensor

    def size(self, dim: int) -> int:
        return self._tensor.size(dim)

    def dim(self) -> int:
        return self._tensor.dim()

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    def view(self, *shape: int) -> TDTensor:
        return ConstantTDTensor(self._tensor.view(*shape))


class CallableTDTensor(TDTensor):
    def __init__(
        self,
        f: callable[[float], Tensor],
        f0: Tensor,
        dtype: torch.dtype,
        device: torch.device,
    ):
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

        self._callable = f
        self._f0 = f0
        self._shape = f0.shape
        self.dtype = dtype
        self.device = device

    @property
    def is_constant(self) -> bool:
        return False

    @cache
    def __call__(self, t: float) -> Tensor:
        return self._callable(t).view(self._shape)

    def size(self, dim: int) -> int:
        return self._shape[dim]

    def dim(self) -> int:
        return len(self._shape)

    @property
    def shape(self) -> torch.Size:
        return self._shape

    def view(self, *shape: int) -> TDTensor:
        f0 = self._f0.view(*shape)
        return CallableTDTensor(
            f=self._callable, f0=f0, dtype=self.dtype, device=self.device
        )
