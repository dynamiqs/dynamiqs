from __future__ import annotations

from functools import partial, wraps
from typing import Any

import torch
from methodtools import lru_cache
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

    if torch_device.type == 'cuda' and torch_device.index is None:
        torch_device = torch.device(
            torch_device.type, index=torch.cuda.current_device()
        )

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


def cache(func=None, *, maxsize: int = 1):
    """Cache a function returning a tensor by memoizing its most recent calls.

    This decorator extends `methodtools.lru_cache` to also cache a function on
    PyTorch grad mode status (enabled or disabled). This prevents cached tensors
    detached from the graph (for example computed within a `with torch.no_grad()`
    block) from being used by mistake by later code which requires tensors attached
    to the graph.

    By default, the cache size is `1`, which means that only the most recent call is
    cached. Use the `maxsize` keyword argument to change the maximum cache size.

    Warning:
        This decorator should only be used for PyTorch tensors.

    Example:
        >>> @cache
        ... def square(x: Tensor) -> Tensor:
        ...     print('compute square')
        ...     return x**2
        ...
        >>> x = torch.tensor([1, 2, 3])
        >>> square(x)
        compute square
        tensor([1, 4, 9])
        >>> square(x)
        tensor([1, 4, 9])
        >>> with torch.no_grad():
        ...     print(square(x))
        ...     print(square(x))
        ...
        compute square
        tensor([1, 4, 9])
        tensor([1, 4, 9])

        Increasing the maximum cache size:
        >>> @cache(maxsize=2)
        ... def square(x):
        ...     print('compute square')
        ...     return x**2
        ...
        >>> square(1)
        compute square
        1
        >>> square(2)
        compute square
        4
        >>> square(1)
        1
        >>> square(2)
        4
        >>> square(3)
        compute square
        9
        >>> square(2)
        4
        >>> square(1)
        compute square
        1

    Args:
        func: Function returning a tensor, can take any number of arguments.

    Returns:
        Cached function.
    """
    if func is None:
        return partial(cache, maxsize=maxsize)

    # define a function cached on its arguments and also on PyTorch grad mode status
    @lru_cache(maxsize=maxsize)
    def grad_cached_func(*args, grad_enabled, **kwargs):
        return func(*args, **kwargs)

    # wrap `func` to call its modified cached version
    @wraps(func)
    def wrapper(*args, **kwargs):
        return grad_cached_func(*args, grad_enabled=torch.is_grad_enabled(), **kwargs)

    return wrapper
