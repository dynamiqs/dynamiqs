from typing import Callable, List, Optional, Union, get_args

import numpy as np
import torch
from qutip import Qobj
from torch import Tensor

from .utils import from_qutip

# TODO add typing for Hamiltonian with piecewise-constant factor
TDOperator = Union[Tensor, Callable[[float], Tensor]]

# type for objects convertible to a torch tensor using `torch.as_tensor`
TensorLike = Union[List, np.ndarray, Tensor]

# type for objects convertible to a torch tensor using `to_tensor`
OperatorLike = Union[TensorLike, Qobj]

# type for objects convertible to a `TDOperator` using `time_dependent_to_tensor`
TDOperatorLike = Union[OperatorLike, Callable[[float], OperatorLike]]


def to_tensor(x: Optional[Union[OperatorLike, List[OperatorLike]]]) -> Tensor:
    """Convert a `OperatorLike` object or a list of `OperatorLike` object to a PyTorch
    tensor.

    Args:
        x: QuTiP quantum object or NumPy array or Python list or PyTorch tensor or list
           of these types. If `None` or empty list, returns an empty tensor of size (0).

    Returns:
        PyTorch tensor.
    """
    if x is None:
        return torch.tensor([])
    if isinstance(x, list):
        if len(x) == 0:
            return torch.tensor([])
        return torch.stack([to_tensor(y) for y in x])
    if isinstance(x, Qobj):
        return from_qutip(x)
    elif isinstance(x, get_args(TensorLike)):
        return torch.as_tensor(x)
    else:
        raise TypeError(
            f'Input of type {type(x)} is not supported. `to_tensor` only '
            'supports QuTiP quantum object, NumPy array, Python list or PyTorch tensor '
            'or list of these types.'
        )
