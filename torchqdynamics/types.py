from typing import Callable, List, Union, get_args

import numpy as np
import torch
from qutip import Qobj

from .utils import from_qutip

# TODO: add typing for Hamiltonian with piecewise-constant factor
TimeDependentOperator = Union[torch.Tensor, Callable[[float], torch.Tensor]]

# type for objects convertible to a torch tensor using `torch.as_tensor`
TensorLike = Union[List, np.ndarray, torch.Tensor]

# type for objects convertible to a torch tensor using `torchqdynamics.to_tensor`
QTensorLike = Union[TensorLike, Qobj]


def to_torch(x: QTensorLike) -> torch.Tensor:
    """Convert a `QTensorLike` object to a PyTorch tensor.

    Args:
        x: QuTiP quantum object or NumPy array or Python list or PyTorch tensor.

    Returns:
        PyTorch tensor.
    """
    if isinstance(x, Qobj):
        return from_qutip(x)
    elif isinstance(x, get_args(TensorLike)):
        return torch.as_tensor(x)
    else:
        raise TypeError(
            'invalid type (supported: QuTiP quantum object, NumPy array, '
            'Python list or PyTorch tensor)'
        )
