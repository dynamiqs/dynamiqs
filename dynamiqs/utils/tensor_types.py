from __future__ import annotations

import functools
from typing import Callable, Union, get_args

import numpy as np
import torch
from qutip import Qobj
from torch import Tensor

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
    device: torch.device | None = None,
    is_complex: bool = False,
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
    if is_complex:
        dtype = cdtype(dtype)

    if x is None:
        return torch.tensor([], dtype=dtype, device=device)
    if isinstance(x, list):
        if len(x) == 0:
            return torch.tensor([], dtype=dtype, device=device)
        return torch.stack([to_tensor(y, dtype=dtype, device=device) for y in x])
    if isinstance(x, Qobj):
        return from_qutip(x, dtype=dtype, device=device)
    elif isinstance(x, get_args(TensorLike)):
        return torch.as_tensor(x, dtype=dtype, device=device)
    else:
        raise TypeError(
            f'Input of type {type(x)} is not supported. `to_tensor` only '
            'supports QuTiP quantum object, NumPy array, Python list or PyTorch tensor '
            'or list of these types.'
        )


def cdtype(
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
            f'Argument `dtype` ({dtype}) must be `torch.complex64`,'
            ' `torch.complex128` or `None` for a complex tensor.'
        )
    return dtype


def rdtype(
    dtype: torch.float32 | torch.float64 | None = None,
) -> torch.float32 | torch.float64:
    if dtype is None:
        return torch.get_default_dtype()
    elif dtype not in (torch.float32, torch.float64):
        raise TypeError(
            f'Argument `dtype` ({dtype}) must be `torch.float32`,'
            ' `torch.float64` or `None` for a real-valued tensor.'
        )
    return dtype


def complex_tensor(func):
    @functools.wraps(func)
    def wrapper(
        *args,
        dtype: torch.complex64 | torch.complex128 | None = None,
        device: torch.device | None = None,
        **kwargs,
    ):
        return func(*args, dtype=cdtype(dtype), device=device, **kwargs)

    return wrapper


DTYPE_TO_REAL = {torch.complex64: torch.float32, torch.complex128: torch.float64}
DTYPE_TO_COMPLEX = {torch.float32: torch.complex64, torch.float64: torch.complex128}


def dtype_complex_to_real(
    dtype: torch.complex64 | torch.complex128 | None = None,
) -> torch.float32 | torch.float64:
    dtype = cdtype(dtype)
    return DTYPE_TO_REAL[dtype]


def dtype_real_to_complex(
    dtype: torch.float32 | torch.float64 | None = None,
) -> torch.complex64 | torch.complex128:
    dtype = rdtype(dtype)
    return DTYPE_TO_COMPLEX[dtype]


@complex_tensor
def from_qutip(
    x: Qobj, *, dtype: torch.dtype | None = None, device: torch.device | None = None
) -> Tensor:
    """Convert a QuTiP quantum object to a PyTorch tensor.

    Args:
        x: Input quantum object.
        dtype: Data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        Output tensor.
    """
    return torch.from_numpy(x.full()).to(dtype=dtype, device=device)


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
