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
TDTensor = Union[Tensor, Callable[[float], Tensor]]
TDOperatorLike = Union[OperatorLike, Callable[[float], Tensor]]


def to_tensor(
    x: OperatorLike | list[OperatorLike] | None,
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    is_complex: bool = False,
) -> Tensor:
    """Convert a `OperatorLike` object or a list of `OperatorLike` object to a PyTorch
    tensor.

    Args:
        x: QuTiP quantum object or NumPy array or Python list or PyTorch tensor or list
           of these types. If `None` or empty list, returns an empty tensor of size (0).
        dtype: Data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        PyTorch tensor.
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
        x = to_tensor(x, dtype=dtype, device=device, is_complex=True)
    elif isinstance(x, callable):
        # check number of arguments and compute at initial time
        try:
            x0 = x(0.0)
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
        if not isinstance(x0, Tensor):
            raise TypeError(
                f'{prefix} The provided operator is currently of type'
                f' {type(x0)} instead of {Tensor}.'
            )
        elif x0.dtype != dtype:
            raise TypeError(
                f'{prefix} The provided operator is currently of dtype'
                f' {x0.dtype} instead of {dtype}.'
            )
        elif x0.device != device:
            raise TypeError(
                f'{prefix} The provided operator is currently on device'
                f' {x0.device} instead of {device}.'
            )

    return x


def tdtensor_get_ndim(x: TDTensor) -> int:
    """Get the number of dimensions of a `TDTensor`."""
    if isinstance(x, Tensor):
        return x.ndim
    elif isinstance(x, callable):
        return x(0.0).ndim


def tdtensor_get_shape(x: TDTensor) -> torch.Size:
    """Get the shape of a `TDTensor`."""
    if isinstance(x, Tensor):
        return x.shape
    elif isinstance(x, callable):
        return x(0.0).shape


def tdtensor_get_size(x: TDTensor, dim: int) -> int:
    """Get the size of a given dimension of a `TDTensor`."""
    if isinstance(x, Tensor):
        return x.size(dim)
    elif isinstance(x, callable):
        return x(0.0).size(dim)


def tdtensor_unsqueeze(x: TDTensor, dims: tuple[int]) -> TDTensor:
    """Unsqueeze a `TDTensor` at each position provided by `dims`."""
    # compute output shape
    shape = torch.ones(len(dims), dtype=torch.int)
    x_shape = torch.as_tensor(tdtensor_get_shape(x))
    out_shape = torch.cat([shape, x_shape])

    # compute output dims
    dims = torch.as_tensor(dims)
    x_dims = torch.arange(0, len(x_shape))
    out_dims = torch.cat([dims - 0.5, x_dims])

    # compute sorted output shape
    sorted_indices = torch.argsort(out_dims)
    out_shape = torch.Size(out_shape[sorted_indices].tolist())

    if isinstance(x, Tensor):
        return x.view(out_shape)
    elif isinstance(x, callable):
        return lambda t: x(t).view(out_shape)


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
def from_qutip(x: Qobj, *, dtype=None, device=None) -> Tensor:
    """Convert a QuTiP quantum object to a PyTorch tensor.

    Args:
        x: QuTiP quantum object.

    Returns:
        PyTorch tensor.
    """
    return torch.from_numpy(x.full()).to(dtype=dtype, device=device)


def to_qutip(x: Tensor, dims: list | None = None) -> Qobj:
    """Convert a PyTorch tensor to a QuTiP quantum object.

    Args:
        x: PyTorch tensor.
        dims: QuTiP object dimensions.

    Returns:
        QuTiP quantum object.
    """
    return Qobj(x.numpy(force=True), dims=dims)
