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
    rmax: float = 1.0,
    requires_grad: bool = False,
    seed: int | None = None,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns a tensor filled with random complex numbers uniformly distributed in the
    complex plane.

    Each element of the returned tensor has a random magnitude between 0 and `rmax` and
    a random phase. Formally, each element is defined by

    $$
        x = re^{i\theta}\ \text{with}\
        \left\{\begin{aligned}
        r      &= \texttt{rmax} \cdot \sqrt{\texttt{rand(0,1)}} \\
        \theta &= 2\pi \cdot \texttt{rand(0,1)}
        \end{aligned}\right.
    $$

    where $\texttt{rand(0,1)}$ is a random number uniformly distributed between 0 and 1.

    Note:
        The square root in the definition of the magnitude $r$ ensures that the
        resulting complex numbers are uniformly distributed in the complex plane.

    Args:
        size _(int or tuple of ints)_: Size of the returned tensor.
        rmax: Maximum magnitude.
        requires_grad: Whether gradients need to be computed with respect to the
            returned tensor.
        seed: Seed for the random number generator.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(*size)_ Tensor filled with random complex numbers.

    Examples:
        >>> x = dq.rand_complex((2, 5), rmax=2.0, seed=42)
        >>> x
        tensor([[ 1.722-0.750j, -1.592-1.061j,  0.844-0.905j, -1.784-0.809j,
                 -0.070-1.248j],
                [-1.400+0.665j,  0.762-0.668j, -1.593-0.798j, -0.202+1.929j,
                 -0.508-0.524j]])
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

    # generate random magnitude with values in [0, rmax], the sqrt ensures that the
    # resulting complex numbers are uniformly distributed in the complex plane
    r = rmax * rand().sqrt()
    # generate random phase with values in [0, 2pi]
    theta = 2 * torch.pi * rand()
    x = r * torch.exp(1j * theta)

    x.requires_grad = requires_grad

    return x


def pwc_pulse(t_start: float, t_end: float, x: Tensor) -> callable[[float], Tensor]:
    """Returns a piecewise-constant (PWC) pulse.

    TODO: explain better what a PWC pulse is and how it is defined from `x`. Also it
    returns 0.0 outside of `[t_start, t_end]`.

    Note:
        You can use [rand_complex()][dynamiqs.rand_complex] to generate a tensor
        filled with random complex numbers for the parameter `x`.

    Args:
        t_start: Start time of the pulse.
        t_end: End time of the pulse.
        x _(..., nbins)_: Pulse complex values where `nbins` is the number of different
            values between `t_start` and `t_end`.

    Returns:
        Function that takes a time `t` and returns the pulse value at time `t` (a
            tensor of shape _(...)_).

    Examples:
        >>> x = dq.rand_complex((2, 5), rmax=2.0, seed=42)
        >>> pulse = dq.pwc_pulse(0.0, 1.0, x)
        >>> type(pulse)
        <class 'function'>
        >>> pulse(0.5)
        tensor([ 0.844-0.905j, -1.593-0.798j])
        >>> pulse(1.2)
        tensor([0.+0.j, 0.+0.j])
    """

    def pulse(t):
        if t < t_start or t > t_end:
            # return a null tensor of appropriate shape
            batch_sizes = x.shape[:-1]
            return torch.zeros(batch_sizes, dtype=x.dtype, device=x.device)
        else:
            # find the index corresponding to time `t`
            nbins = x.size(-1)
            idx = int(linmap(t, t_start, t_end, 0, nbins - 1))
            return x[..., idx]

    return pulse
