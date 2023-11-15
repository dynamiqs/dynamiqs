from __future__ import annotations

from typing import Any

import torch
from torch import Tensor

from .utils.utils import isket


def type_str(type: Any) -> str:
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def to_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        torch_device = torch.ones(1).device  # default device
    elif isinstance(device, str):
        torch_device = torch.device(device)
    elif isinstance(device, torch.device):
        torch_device = device
    else:
        raise TypeError(
            'Argument `device` must be a string, a `torch.device` or `None` but has'
            f' type {obj_type_str(device)}.'
        )

    if torch_device.type == 'cuda' and not torch_device.index:
        return torch.device(torch_device.type, index=0)  # default cuda index to 0
    else:
        return torch_device


def hdim(x: Tensor) -> int:
    if isket(x):
        return x.size(-2)
    else:
        return x.size(-1)


def check_time_tensor(x: Tensor, arg_name: str, allow_empty=False):
    # check that a time tensor is valid (it must be a 1D tensor sorted in strictly
    # ascending order and containing only positive values)
    if x.ndim != 1:
        raise ValueError(
            f'Argument `{arg_name}` must be a 1D tensor, but is a {x.ndim}D tensor.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument `{arg_name}` must contain at least one element.')
    if not torch.all(torch.diff(x) > 0):
        raise ValueError(
            f'Argument `{arg_name}` must be sorted in strictly ascending order.'
        )
    if not torch.all(x >= 0):
        raise ValueError(f'Argument `{arg_name}` must contain positive values only.')
