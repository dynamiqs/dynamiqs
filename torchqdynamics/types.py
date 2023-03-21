from typing import Callable, List, Union

import numpy as np
import torch

# TODO: add typing for Hamiltonian with piecewise-constant factor
TimeDependentOperator = Union[torch.Tensor, Callable[[float], torch.Tensor]]

# type for objects convertible to a torch tensor using `torch.as_tensor`
TensorLike = Union[List, np.ndarray, torch.Tensor]
