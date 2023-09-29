from __future__ import annotations

from abc import ABC, abstractmethod
from itertools import compress
from typing import get_args

import numpy as np
import torch
from torch import Tensor

from ..._utils import obj_type_str, to_device, type_str
from ...utils.tensor_types import (
    ArrayLike,
    TDArrayLike,
    dtype_complex_to_real,
    get_cdtype,
    to_tensor,
)
from .utils import cache, merge_tensors


def to_td_tensor(
    x: TDArrayLike,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> TDTensor:
    """Convert a `TDArrayLike` object to a `TDTensor` object."""
    cdtype = get_cdtype(dtype)
    rdtype = dtype_complex_to_real(cdtype)
    device = to_device(device)

    if isinstance(x, tuple):
        if len(x) == 0:
            raise ValueError(
                'The piecewise-constant operator must be a non-empty tuple.'
            )

        # split tuple elements between static and pwc elements using a mask
        is_static = np.array([isinstance(_x, get_args(ArrayLike)) for _x in x])
        if not any(is_static):
            x_static = None
        else:
            # sum static parts to a single tensor
            x_static = list(compress(x, is_static))
            x_static = [to_tensor(_x, dtype=cdtype, device=device) for _x in x_static]
            x_static = torch.sum(torch.stack(x_static), dim=0)

        # if there is only static parts, convert to constant tensor
        if all(is_static):
            return to_td_tensor(x_static, dtype=dtype, device=device)

        # convert all pwc parts to tensors
        x_pwc = list(compress(x, ~is_static))
        x_pwc = [
            (
                to_tensor(tensor, dtype=cdtype, device=device),
                to_tensor(times, dtype=rdtype, device=device),
                to_tensor(values, dtype=cdtype, device=device),
            )
            for tensor, times, values in x_pwc
        ]

        return PWCTDTensor(x_static, x_pwc)
    elif isinstance(x, get_args(ArrayLike)):
        # convert to tensor
        x = to_tensor(x, dtype=cdtype, device=device)
        return ConstantTDTensor(x)
    elif callable(x):
        # compute initial value of the callable
        x0 = x(0.0)

        # check callable
        check_callable(x0, cdtype, device)

        return CallableTDTensor(x, shape=x0.shape, dtype=cdtype, device=device)


def check_callable(
    x0: Tensor,
    expected_dtype: torch.dtype,
    expected_device: torch.device,
):
    # check type, dtype and device match

    if not isinstance(x0, Tensor):
        raise TypeError(
            f'The time-dependent operator must be a {type_str(Tensor)}, but has type'
            f' {obj_type_str(x0)}. The provided callable must return a tensor, to avoid'
            ' costly type conversion at each time solver step.'
        )
    elif x0.dtype != expected_dtype:
        raise TypeError(
            f'The time-dependent operator must have dtype `{expected_dtype}`, but has'
            f' dtype `{x0.dtype}`. The provided callable must return a tensor with the'
            ' same `dtype` as provided to the solver, to avoid costly dtype conversion'
            ' at each solver time step.'
        )
    elif x0.device != expected_device:
        raise TypeError(
            f'The time-dependent operator must be on device `{expected_device}`, but is'
            f' on device `{x0.device}`. The provided callable must return a tensor on'
            ' the same device as provided to the solver, to avoid costly device'
            ' transfer at each solver time step.'
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


class PWCTDTensor(TDTensor):
    def __init__(self, static: Tensor | None, pwc: list[tuple[Tensor, Tensor, Tensor]]):
        # argument pwc must be a non-empty list
        tensor0, _, values0 = pwc[0]
        self._dtype = tensor0.dtype
        self._device = tensor0.device
        self._shape = torch.Size((*values0.shape[:-1], *tensor0.shape))  # (..., n, n)

        self._static = static  # (n, n)
        self._pwc = pwc

        self.times = merge_tensors(*[times for _, times, _ in pwc])

    def __call__(self, t: float) -> Tensor:
        total_tensor = torch.zeros(self._shape, dtype=self._dtype, device=self._device)
        if self._static is not None:
            total_tensor += self._static
        for x in self._pwc:
            tensor, times, values = x
            if t < times[0] or t >= times[-1]:
                continue
            else:
                idx = torch.searchsorted(times, t, side='right') - 1
                v = values[..., idx]  # (...)
                total_tensor += v[..., None, None] * tensor  # (..., n, n)
        return total_tensor

    def size(self, dim: int) -> int:
        return self._shape[dim]

    def dim(self) -> int:
        return len(self._shape)

    def unsqueeze(self, dim: int) -> TDTensor:
        pwc = [
            (tensor, times, values.unsqueeze(dim))
            for tensor, times, values in self._pwc
        ]
        return PWCTDTensor(self._static, pwc)
