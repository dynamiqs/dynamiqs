from __future__ import annotations

from typing import Callable, Union, get_args

import numpy as np
import torch
from qutip import Qobj
from torch import Tensor

from .._utils import obj_type_str

__all__ = [
    'to_tensor',
    'from_qutip',
    'to_qutip',
]

# type for objects convertible to a torch.Tensor using `torch.as_tensor`
TensorLike = Union[list, np.ndarray, Tensor]

# type for objects convertible to a torch.Tensor using `to_tensor`
OperatorLike = Union[TensorLike, Qobj]

# TODO add typing for Hamiltonian with piecewise-constant factor
# type for time-dependent objects
TDOperatorLike = Union[OperatorLike, Callable[[float], Tensor]]


def to_tensor(
    x: OperatorLike | list[OperatorLike] | None,
    *,
    dtype: torch.dtype | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Convert an array-like object (or a list of array-like objects) to a tensor.

    Args:
        x: QuTiP quantum object or NumPy array or Python list or PyTorch tensor or
            list of these types. If `None` or empty list, returns an empty tensor of
            shape _(0)_.
        dtype: Data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        Output tensor.
    """
    if x is None:
        return torch.tensor([], dtype=dtype, device=device)
    if isinstance(x, list):
        if len(x) == 0:
            return torch.tensor([], dtype=dtype, device=device)
        return torch.stack([to_tensor(y, dtype=dtype, device=device) for y in x])
    if isinstance(x, Qobj):
        return from_qutip(x, dtype=get_cdtype(dtype), device=device)
    elif isinstance(x, get_args(TensorLike)):
        return torch.as_tensor(x, dtype=dtype, device=device)
    else:
        raise TypeError(
            'Argument `x` must be an array-like object, but has type'
            f' {obj_type_str(x)}.'
        )


def get_cdtype(
    dtype: torch.complex64 | torch.complex128 | None = None,
) -> torch.complex64 | torch.complex128:
    if dtype is None:
        # the default dtype for complex tensors is determined by the default
        # floating point dtype (torch.complex128 if default is torch.float64,
        # otherwise torch.complex64)
        if torch.get_default_dtype() is torch.float64:
            return torch.complex128
        else:
            return torch.complex64
    elif dtype not in (torch.complex64, torch.complex128):
        raise TypeError(
            'Argument `dtype` must be `torch.complex64`, `torch.complex128` or `None`'
            f' for a complex tensor, but is `{dtype}`.'
        )
    return dtype


def get_rdtype(
    dtype: torch.float32 | torch.float64 | None = None,
) -> torch.float32 | torch.float64:
    if dtype is None:
        return torch.get_default_dtype()
    elif dtype not in (torch.float32, torch.float64):
        raise TypeError(
            'Argument `dtype` must be `torch.float32`, `torch.float64` or `None` for'
            f' a real-valued tensor, but is `{dtype}`.'
        )
    return dtype


DTYPE_TO_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}
DTYPE_TO_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def dtype_complex_to_real(
    dtype: torch.complex64 | torch.complex128,
) -> torch.float32 | torch.float64:
    return DTYPE_TO_REAL[dtype]


def dtype_real_to_complex(
    dtype: torch.float32 | torch.float64,
) -> torch.complex64 | torch.complex128:
    return DTYPE_TO_COMPLEX[dtype]


def from_qutip(
    x: Qobj,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    """Convert a QuTiP quantum object to a PyTorch tensor.

    Args:
        x: Input quantum object.
        dtype: Complex data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        Output tensor.
    """
    return torch.from_numpy(x.full()).to(dtype=get_cdtype(dtype), device=device)


def to_qutip(x: Tensor, dims: list | None = None) -> Qobj:
    """Convert a PyTorch tensor to a QuTiP quantum object.

    Args:
        x: PyTorch tensor.
        dims: QuTiP object dimensions, with size _(2, n)_ where _n_ is the number of
            modes in the tensor product.

    Returns:
        QuTiP quantum object.
    """
    return Qobj(x.numpy(force=True), dims=dims)


def to_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.ones(1).device  # default device
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(
            f'Argument `device` ({device}) must be a string, a `torch.device` object or'
            ' `None`.'
        )
