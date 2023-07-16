from __future__ import annotations

from abc import ABC, abstractmethod
from typing import get_args

import torch
from torch import Tensor

from ...utils.tensor_types import (
    OperatorLike,
    TDOperatorLike,
    to_device,
    to_rdtype,
    to_tensor,
)
from ...utils.utils import obj_type_str, type_str
from .utils import cache


def to_td_tensor(
    x: TDOperatorLike,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> TDTensor:
    """Convert a `TDOperatorLike` object to a `TDTensor` object."""
    device = to_device(device)

    if isinstance(x, get_args(OperatorLike)):
        # convert to tensor
        x = to_tensor(x, dtype=dtype, device=device)
        return ConstantTDTensor(x)
    elif callable(x):
        dtype = to_rdtype(dtype) if dtype is None else dtype  # assume real by default

        # compute initial value of the callable
        x0 = x(0.0)

        # check callable
        check_callable(x0, dtype, device)

        return CallableTDTensor(x, shape=x0.shape, dtype=dtype, device=device)


def check_callable(
    x0: Tensor,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
):
    # check type, dtype and device match

    if not isinstance(x0, Tensor):
        raise TypeError(
            f'Time-dependent operator must be a {type_str(Tensor)}, but has type'
            f' {obj_type_str(x0)}. The function must return a tensor, to avoid costly'
            ' type conversion at each time solver step.'
        )
    elif x0.dtype != expected_dtype:
        raise TypeError(
            f'Time-dependent operator must have dtype `{expected_dtype}`, but has'
            f' dtype `{x0.dtype}`. The function must return a tensor with the same'
            ' `dtype` as provided to the solver, to avoid costly dtype conversion at'
            ' each solver time step.'
        )
    elif x0.device != expected_device:
        raise TypeError(
            f'Time-dependent operator must be on device `{expected_device}`, but is'
            f' on device `{x0.device}`. The function must return a tensor on the same'
            ' device as provided to the solver, to avoid costly device transfer at'
            ' each solver time step.'
        )


class TDTensor(ABC):
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

    @abstractmethod
    def unsqueeze(self, dim: int) -> TDTensor:
        """Unsqueeze at position `dim`."""
        pass


class ConstantTDTensor(TDTensor):
    def __init__(self, tensor: Tensor):
        self._tensor = tensor
        self.dtype = tensor.dtype
        self.device = tensor.device

    def __call__(self, t: float) -> Tensor:
        return self._tensor

    def size(self, dim: int) -> int:
        return self._tensor.size(dim)

    def dim(self) -> int:
        return self._tensor.dim()

    def unsqueeze(self, dim: int) -> ConstantTDTensor:
        return ConstantTDTensor(self._tensor.unsqueeze(dim))


class CallableTDTensor(TDTensor):
    def __init__(
        self,
        f: callable[[float], Tensor],
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._callable = f
        self._shape = shape
        self.dtype = dtype
        self.device = device

    @cache
    def __call__(self, t: float) -> Tensor:
        return self._callable(t).view(self._shape)

    def size(self, dim: int) -> int:
        return self._shape[dim]

    def dim(self) -> int:
        return len(self._shape)

    def unsqueeze(self, dim: int) -> CallableTDTensor:
        new_shape = list(self._shape)
        new_shape.insert(dim, 1)
        new_shape = torch.Size(new_shape)
        return CallableTDTensor(
            f=self._callable, shape=new_shape, dtype=self.dtype, device=self.device
        )
