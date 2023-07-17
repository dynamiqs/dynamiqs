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
    r"""Convert an array-like object (or a list of array-like objects) to a tensor.

    Args:
        x _(array-like or list of array-like)_: QuTiP quantum object or NumPy array or
            Python list or PyTorch tensor or list of these types. If `None` or empty
            list, returns an empty tensor of shape _(0)_.
        dtype: Data type of the returned tensor.
        device: Device on which the returned tensor is stored.
        is_complex: Whether the returned tensor is complex-valued.

    Returns:
        Output tensor.

    Examples:
        >>> import numpy as np
        >>> import qutip as qt
        >>> dq.to_tensor(qt.fock(3, 1))
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.to_tensor(np.array([[1, 2, 3], [4, 5, 6]]))
        tensor([[1, 2, 3],
                [4, 5, 6]])
        >>> dq.to_tensor([qt.fock(3, 1), qt.fock(3, 2)])
        tensor([[[0.+0.j],
                 [1.+0.j],
                 [0.+0.j]],
        <BLANKLINE>
                [[0.+0.j],
                 [0.+0.j],
                 [1.+0.j]]])
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
    x: Qobj,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: torch.device | None = None,
) -> Tensor:
    r"""Convert a QuTiP quantum object to a PyTorch tensor.

    Args:
        x _(QuTiP quantum object)_: Input quantum object.
        dtype: Complex data type of the returned tensor.
        device: Device on which the returned tensor is stored.

    Returns:
        Output tensor.

    Examples:
        >>> import qutip as qt
        >>> omega = 2.0
        >>> a = qt.destroy(4)
        >>> H = omega * a.dag() * a
        >>> H
        Quantum object: dims = [[4], [4]], shape = (4, 4), type = oper, isherm = True
        Qobj data =
        [[0. 0. 0. 0.]
         [0. 2. 0. 0.]
         [0. 0. 4. 0.]
         [0. 0. 0. 6.]]
        >>> dq.from_qutip(H)
        tensor([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 2.+0.j, 0.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 4.+0.j, 0.+0.j],
                [0.+0.j, 0.+0.j, 0.+0.j, 6.+0.j]])
    """
    return torch.from_numpy(x.full()).to(dtype=dtype, device=device)


def to_qutip(x: Tensor, dims: list[list[int]] | None = None) -> Qobj:
    r"""Convert a PyTorch tensor to a QuTiP quantum object.

    Args:
        x: PyTorch tensor.
        dims _(list of list of int or None)_: QuTiP object dimensions, with shape
            _(2, n)_ where _n_ is the number of modes in the tensor product.

    Returns:
        QuTiP quantum object.

    Examples:
        >>> psi = dq.fock(3, 1)
        >>> psi
        tensor([[0.+0.j],
                [1.+0.j],
                [0.+0.j]])
        >>> dq.to_qutip(psi)
        Quantum object: dims = [[3], [1]], shape = (3, 1), type = ket
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        Note that the tensor product structure is not inferred automatically, it must be
        specified with the `dims` argument using QuTiP dimensions format:
        >>> I = dq.eye(3, 2)
        >>> dq.to_qutip(I)
        Quantum object: dims = [[6], [6]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
        >>> dq.to_qutip(I, [[3, 2], [3, 2]])
        Quantum object: dims = [[3, 2], [3, 2]], shape = (6, 6), type = oper, isherm = True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    return Qobj(x.numpy(force=True), dims=dims)
