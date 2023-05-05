from __future__ import annotations

from abc import ABC, abstractmethod
from typing import get_args

import torch
from torch import Tensor

from .tensor_types import OperatorLike, TDOperatorLike, cdtype, rdtype, to_tensor


def to_tdtensor(
    x: TDOperatorLike,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    is_complex: bool = False,
) -> TDTensor:
    """Convert a `TDOperatorLike` object to a `TDTensor` object."""
    if is_complex:
        dtype = cdtype(dtype)

    if isinstance(x, get_args(OperatorLike)):
        x = to_tensor(x, dtype=dtype, device=device, is_complex=is_complex)
        return ConstantTDTensor(x, dtype=dtype, device=device)
    elif callable(x):
        if dtype is None and not is_complex:
            dtype = rdtype(dtype)
        if device is None:
            device = torch.device('cpu')
        return CallableTDTensor(x, dtype=dtype, device=device)


class TDTensor(ABC):
    @abstractmethod
    def __call__(self, t: float) -> Tensor:
        """Evaluate a `TDTensor` at a given time"""
        pass

    @abstractmethod
    def size(self, dim: int) -> int:
        """Size of a `TDTensor` along a given dimension."""
        pass

    @abstractmethod
    def dim(self) -> int:
        """Get the number of dimensions of a `TDTensor`."""
        pass

    @abstractmethod
    def unsqueeze(self, dim: int) -> TDTensor:
        """Unsqueeze a `TDTensor` at position `dim`."""
        pass

    @abstractmethod
    def has_changed(self, t: float) -> bool:
        """Checks whether the `TDTensor` has changed since the last call."""
        pass


class ConstantTDTensor(TDTensor):
    def __init__(
        self, tensor: Tensor, *, dtype: torch.dtype, device: torch.device | None
    ):
        self._tensor = tensor
        self.dtype = dtype or tensor.dtype
        self.device = device or tensor.device

    def __call__(self, t: float) -> Tensor:
        return self._tensor

    def size(self, dim: int) -> int:
        return self._tensor.size(dim)

    def dim(self) -> int:
        return self._tensor.dim()

    def unsqueeze(self, dim: int) -> ConstantTDTensor:
        return ConstantTDTensor(
            self._tensor.unsqueeze(dim), dtype=self.dtype, device=self.device
        )

    def has_changed(self, t: float) -> bool:
        return False


class CallableTDTensor(TDTensor):
    def __init__(
        self,
        f: callable,
        *,
        dtype: torch.dtype,
        device: torch.device,
        shape: torch.Size | None = None,
        check_input: bool = True,
    ):
        self._callable = f
        self.dtype = dtype
        self.device = device
        if check_input:
            tensor = self._check_callable()
        self._shape = shape or tensor.shape
        self._cached_tensor = None
        self._last_t = None

    def __call__(self, t: float) -> Tensor:
        if t != self._last_t:
            self._cached_tensor = self._callable(t).view(self._shape)
            self._last_t = t
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
            self._callable,
            dtype=self.dtype,
            device=self.device,
            shape=new_shape,
            check_input=False,
        )

    def has_changed(self, t: float) -> bool:
        return True

    def _check_callable(self) -> Tensor:
        # check number of arguments and compute at initial time
        try:
            tensor = self._callable(0.0)
        except TypeError as e:
            raise TypeError(
                'Time-dependent operators in the `callable` format should only accept a'
                ' single argument for time `t`.'
            ) from e

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
        elif tensor.dtype != self.dtype:
            raise TypeError(
                f'{prefix} The provided operator is currently of dtype'
                f' {tensor.dtype} instead of {self.dtype}.'
            )
        elif tensor.device != self.device:
            raise TypeError(
                f'{prefix} The provided operator is currently on device'
                f' {tensor.device} instead of {self.device}.'
            )

        return tensor
