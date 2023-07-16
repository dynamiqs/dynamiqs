from __future__ import annotations

from typing import Any

import torch


def type_str(type: Any) -> str:
    return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def to_device(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.ones(1).device  # default device
    elif isinstance(device, str):
        return torch.device(device)
    elif isinstance(device, torch.device):
        return device
    else:
        raise TypeError(
            'Argument `device` must be a string, a `torch.device` or `None` but has'
            f' type {obj_type_str(device)}.'
        )
