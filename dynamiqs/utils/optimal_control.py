from __future__ import annotations

import torch
from torch import Tensor

from .._utils import linmap
from .tensor_types import dtype_complex_to_real, get_cdtype, to_device

__all__ = [
    'rand_complex',
    'pwc_pulse',
]


def rand_complex(
    size: int | tuple[int, ...],
    *,
    requires_grad: bool = False,
    seed: int | None = None,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns a tensor filled with random complex numbers uniformly distributed in the
    complex plane unit disk.

    Each element of the returned tensor has a random magnitude between 0 and 1 and a
    random phase. Formally, each element is defined by

    $$
        x = re^{i\theta}\ \text{with}\
        \left\{\begin{aligned}
        r      &= \sqrt{\texttt{rand(0,1)}} \\
        \theta &= 2\pi \cdot \texttt{rand(0,1)}
        \end{aligned}\right.
    $$

    where $\texttt{rand(0,1)}$ is a random number uniformly distributed between 0 and 1.

    Note:
        The square root in the definition of the magnitude $r$ ensures that the
        resulting complex numbers are uniformly distributed in the complex plane unit
        disk.

    Args:
        size _(int or tuple of ints)_: Size of the returned tensor.
        requires_grad: Whether gradients need to be computed with respect to the
            returned tensor.
        seed: Seed for the random number generator.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(*size)_ Tensor filled with random complex numbers.

    Examples:
        >>> x = dq.rand_complex((2, 5), seed=42)
        >>> x
        tensor([[ 0.861-0.375j, -0.796-0.531j,  0.422-0.453j, -0.892-0.404j,
                 -0.035-0.624j],
                [-0.700+0.333j,  0.381-0.334j, -0.797-0.399j, -0.101+0.965j,
                 -0.254-0.262j]])
    """
    # Note: We need to manually fetch the default device, because if `device` is `None`
    # `torch.Generator` picks "cpu" as the default device, and not the device set by
    # `torch.set_default_device`.
    device = to_device(device)

    # define random number generator from seed
    generator = torch.Generator(device=device)
    generator.seed() if seed is None else generator.manual_seed(seed)

    rdtype = dtype_complex_to_real(get_cdtype(dtype))

    rand = lambda: torch.rand(size, generator=generator, dtype=rdtype, device=device)

    # generate random magnitude with values in [0, 1[, the sqrt ensures that the
    # resulting complex numbers are uniformly distributed in the complex plane
    r = rand().sqrt()
    # generate random phase with values in [0, 2pi[
    theta = 2 * torch.pi * rand()
    x = r * torch.exp(1j * theta)

    x.requires_grad = requires_grad

    return x


def pwc_pulse(
    t_start: float, t_end: float, values: Tensor
) -> callable[[float], Tensor]:
    """Returns a piecewise-constant (PWC) pulse. TODO: a callable

    TODO: explain better what a PWC pulse is and how it is defined from `x`. Also it
    returns 0.0 outside of `[t_start, t_end]`.

    Note:
        You can use [rand_complex()][dynamiqs.rand_complex] to generate a tensor
        filled with random complex numbers for the parameter `x`.

    Args:
        t_start: Start time of the pulse.
        t_end: End time of the pulse.
        values _(..., nbins)_: Pulse complex values where `nbins` is the number of
            different values between `t_start` and `t_end`.

    Returns:
        Function that takes a time `t` and returns the pulse value at time `t` (a
            tensor of shape _(...)_).

    Examples:
        >>> x = dq.rand_complex((2, 5), seed=42)
        >>> pulse = dq.pwc_pulse(0.0, 1.0, x)
        >>> type(pulse)
        <class 'function'>
        >>> pulse(0.5)
        tensor([ 0.422-0.453j, -0.797-0.399j])
        >>> pulse(1.2)
        tensor([0.+0.j, 0.+0.j])
    """

    def pulse(t):
        if t < t_start or t > t_end:
            # return a null tensor of appropriate shape
            batch_sizes = values.shape[:-1]
            return torch.zeros(batch_sizes, dtype=values.dtype, device=values.device)
        else:
            # find the index corresponding to time `t`
            nbins = values.size(-1)
            idx = int(linmap(t, t_start, t_end, 0, nbins - 1))
            return values[..., idx]

    return pulse
