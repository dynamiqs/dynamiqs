from __future__ import annotations

from abc import ABC, abstractmethod
from typing import get_args

import torch
from torch import Tensor

from .tensor_types import OperatorLike, TDOperatorLike, cdtype, rdtype, to_tensor


def to_td_tensor(
    x: TDOperatorLike,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    is_complex: bool = False,
) -> TDTensor:
    """Convert a `TDOperatorLike` object to a `TDTensor` object."""
    if isinstance(x, get_args(OperatorLike)):
        # convert to tensor
        x = to_tensor(x, dtype=dtype, device=device, is_complex=is_complex)

        return ConstantTDTensor(x)
    elif callable(x):
        # get default dtype and device
        if dtype is None:
            if is_complex:
                dtype = cdtype(dtype)
            else:
                dtype = rdtype(dtype)
        if device is None:
            device = get_default_device()

        # check callable
        x0 = check_callable(x, dtype, device)

        return CallableTDTensor(x, shape=x0.shape, dtype=dtype, device=device)


def get_default_device() -> torch.device:
    """Get the default device."""
    return torch.ones(1).device


def check_callable(
    f: callable,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> Tensor:
    # compute initial value of the callable
    tensor = f(0.0)

    # check type, dtype and device match
    prefix = (
        'Time-dependent operators in the `callable` format should always'
        ' return a `torch.Tensor` with the same dtype and device as provided'
        ' to the solver. This avoids type conversion or device transfer at'
        ' every time step that would slow down the solver.'
    )
    if not isinstance(tensor, Tensor):
        raise TypeError(
            f'{prefix} The provided operator is currently of type'
            f' {type(tensor)} instead of {Tensor}.'
        )
    elif tensor.dtype != expected_dtype:
        raise TypeError(
            f'{prefix} The provided operator is currently of dtype'
            f' {tensor.dtype} instead of {expected_dtype}.'
        )
    elif tensor.device != expected_device:
        raise TypeError(
            f'{prefix} The provided operator is currently on device'
            f' {tensor.device} instead of {expected_device}.'
        )

    return tensor


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

    @abstractmethod
    def requires_updates(self, t: float) -> bool:
        """Whether the cache needs to be updated for new time values."""
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

    def requires_updates(self, t: float) -> bool:
        return False


class CallableTDTensor(TDTensor):
    def __init__(
        self,
        f: callable,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._callable = f
        self.dtype = dtype
        self.device = device
        self._shape = shape
        self._cached_tensor = None
        self._cached_t = None

    def __call__(self, t: float) -> Tensor:
        if t != self._cached_t:
            self._cached_tensor = self._callable(t).view(self._shape)
            self._cached_t = t
        return self._cached_tensor

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

    def requires_updates(self, t: float) -> bool:
        return True
