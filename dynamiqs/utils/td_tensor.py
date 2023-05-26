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

        # compute initial value of the callable
        x0 = x(0.0)

        # check callable
        check_callable(x0, dtype, device)

        return CallableTDTensor(x, shape=x0.shape, dtype=dtype, device=device)


def get_default_device() -> torch.device:
    """Get the default device."""
    return torch.ones(1).device


def check_callable(
    x0: Tensor,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
):
    # check type, dtype and device match
    prefix = (
        'Time-dependent operators in the `callable` format should always'
        ' return a `torch.Tensor` with the same dtype and device as provided'
        ' to the solver. This avoids type conversion or device transfer at'
        ' every time step that would slow down the solver.'
    )
    if not isinstance(x0, Tensor):
        raise TypeError(
            f'{prefix} The provided operator is currently of type'
            f' {type(x0)} instead of {Tensor}.'
        )
    elif x0.dtype != expected_dtype:
        raise TypeError(
            f'{prefix} The provided operator is currently of dtype'
            f' {x0.dtype} instead of {expected_dtype}.'
        )
    elif x0.device != expected_device:
        raise TypeError(
            f'{prefix} The provided operator is currently on device'
            f' {x0.device} instead of {expected_device}.'
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

    @abstractmethod
    def adjoint(self) -> TDTensor:
        """Compute the adjoint."""
        pass

    @abstractmethod
    def __add__(self, other: TDTensor | Tensor) -> TDTensor:
        """Add another `Tensor` or `TDTensor`."""
        pass

    @abstractmethod
    def __sub__(self, other: TDTensor | Tensor) -> TDTensor:
        """Subtract another `Tensor` or `TDTensor`."""
        pass

    def __pos__(self) -> TDTensor:
        """Positive."""
        return self

    @abstractmethod
    def __neg__(self) -> TDTensor:
        """Negative."""
        pass

    @abstractmethod
    def __mul__(self, other: float | complex) -> TDTensor:
        """Multiply by a scalar."""
        pass

    @abstractmethod
    def clear_cache(self):
        """Clear the cache of the instance."""
        pass

    def _check_operation_types(self, other: TDTensor | Tensor) -> TDTensor:
        """Checks the input types for a tensor operation."""
        # check types
        if not isinstance(other, (TDTensor, Tensor)):
            raise TypeError(
                f'Operations between a TDTensor and a {type(other)} are not supported.'
            )

        # check dtype
        if other.dtype != self.dtype:
            raise ValueError(
                f'The dtype of a TDTensor ({self.dtype}) must match the dtype of'
                f' {type(other)} ({other.dtype}).'
            )

        # check device
        if other.device != self.device:
            raise ValueError(
                f'The device of a TDTensor ({self.device}) must match the device of'
                f' {type(other)} ({other.device}).'
            )

    def _check_shape(self, other: TDTensor | Tensor) -> torch.Size:
        """Checks the input shapes for a tensor operation and return the new shape
        after broadcasting."""
        try:
            shape = torch.broadcast_shapes(self.shape, other.shape)
        except RuntimeError:
            raise ValueError(
                f'The shapes of a TDTensor ({self.shape}) and {type(other)}'
                f' ({other.shape}) are not compatible.'
            )

        return shape


class ConstantTDTensor(TDTensor):
    def __init__(self, tensor: Tensor):
        self._tensor = tensor
        self.shape = tensor.shape
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

    def adjoint(self) -> ConstantTDTensor:
        return ConstantTDTensor(self._tensor.adjoint())

    def __add__(self, other: TDTensor | Tensor) -> TDTensor:
        # check types and shapes
        self._check_operation_types(other)
        shape = self._check_shape(other)

        # return a new tensor
        if isinstance(other, Tensor):
            return ConstantTDTensor(self._tensor + other)
        elif isinstance(other, ConstantTDTensor):
            return ConstantTDTensor(self._tensor + other._tensor)
        elif isinstance(other, CallableTDTensor):
            return CallableTDTensor(
                f=lambda t: self._tensor + other(t),
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )

    def __sub__(self, other: TDTensor | Tensor) -> TDTensor:
        # check types and shapes
        self._check_operation_types(other)
        shape = self._check_shape(other)

        # return a new tensor
        if isinstance(other, Tensor):
            return ConstantTDTensor(self._tensor - other)
        elif isinstance(other, ConstantTDTensor):
            return ConstantTDTensor(self._tensor - other._tensor)
        elif isinstance(other, CallableTDTensor):
            return CallableTDTensor(
                f=lambda t: self._tensor - other(t),
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )

    def __neg__(self) -> ConstantTDTensor:
        return ConstantTDTensor(-self._tensor)

    def __mul__(self, other: float | complex) -> ConstantTDTensor:
        if isinstance(other, (float, complex)):
            return ConstantTDTensor(other * self._tensor)
        else:
            raise TypeError(
                f'Operations between a TDTensor and a {type(other)} are not supported.'
            )

    def clear_cache(self):
        pass


class CallableTDTensor(TDTensor):
    def __init__(
        self,
        f: callable,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self._callable = f
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self._cached_tensor = None
        self._cached_t = None

    def __call__(self, t: float) -> Tensor:
        if t != self._cached_t:
            self._cached_tensor = self._callable(t).view(self.shape)
            self._cached_t = t
        return self._cached_tensor

    def size(self, dim: int) -> int:
        return self.shape[dim]

    def dim(self) -> int:
        return len(self.shape)

    def unsqueeze(self, dim: int) -> CallableTDTensor:
        new_shape = list(self.shape)
        new_shape.insert(dim, 1)
        new_shape = torch.Size(new_shape)
        return CallableTDTensor(
            f=self._callable, shape=new_shape, dtype=self.dtype, device=self.device
        )

    def adjoint(self) -> CallableTDTensor:
        return CallableTDTensor(
            f=lambda t: self(t).adjoint(),
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def __add__(self, other: TDTensor | Tensor) -> TDTensor:
        # check types and shapes
        self._check_operation_types(other)
        shape = self._check_shape(other)

        # return a new tensor
        if isinstance(other, Tensor):
            return CallableTDTensor(
                f=lambda t: self(t) + other,
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )
        elif isinstance(other, ConstantTDTensor):
            return CallableTDTensor(
                f=lambda t: self(t) + other._tensor,
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )
        elif isinstance(other, CallableTDTensor):
            return CallableTDTensor(
                f=lambda t: self(t) + other(t),
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )

    def __sub__(self, other: TDTensor | Tensor) -> TDTensor:
        # check types
        self._check_operation_types(other)
        shape = self._check_shape(other)

        # return a new tensor
        if isinstance(other, Tensor):
            return CallableTDTensor(
                f=lambda t: self(t) - other,
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )
        elif isinstance(other, ConstantTDTensor):
            return CallableTDTensor(
                f=lambda t: self(t) - other._tensor,
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )
        elif isinstance(other, CallableTDTensor):
            return CallableTDTensor(
                f=lambda t: self(t) - other(t),
                shape=shape,
                dtype=self.dtype,
                device=self.device,
            )

    def __neg__(self) -> CallableTDTensor:
        return CallableTDTensor(
            f=lambda t: -self(t),
            shape=self.shape,
            dtype=self.dtype,
            device=self.device,
        )

    def __mul__(self, other: float | complex) -> CallableTDTensor:
        if isinstance(other, (float, complex)):
            return CallableTDTensor(
                f=lambda t: other * self(t),
                shape=self.shape,
                dtype=self.dtype,
                device=self.device,
            )
        else:
            raise TypeError(
                f'Operations between a TDTensor and a {type(other)} are not supported.'
            )

    def clear_cache(self):
        self._cached_tensor = None
        self._cached_t = None
