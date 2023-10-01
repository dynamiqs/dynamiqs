from __future__ import annotations

import torch
from torch import Tensor

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
    r"""Returns a tensor filled with random complex numbers uniformly distributed in
    the complex plane.

    Each element of the returned tensor has a random magnitude in the interval
    $[0, \texttt{rmax})$ and a random phase in the interval $[0, 2\pi)$. Formally, each
    element is defined by

    $$
        x = re^{i\theta}\ \text{with}\
        \begin{cases}
            r = \texttt{rmax} \cdot \sqrt{\texttt{rand(0,1)}} \\
            \theta = 2\pi \cdot \texttt{rand(0,1)}
        \end{cases}
    $$

    where $\texttt{rand(0,1)}$ is a random number uniformly distributed between 0 and 1.

    Notes:
        The square root in the definition of the magnitude $r$ ensures that the
        resulting complex numbers are uniformly distributed in the disc of the complex
        plane with a radius of `rmax`.

    Args:
        size _(int or tuple of ints)_: Size of the returned tensor.
        rmax: Maximum magnitude of the random complex numbers.
        requires_grad: Whether gradients need to be computed with respect to the
            returned tensor.
        seed: Seed for the random number generator.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(*size)_ Tensor filled with random complex numbers.

    Examples:
        >>> x = dq.rand_complex((2, 5), rmax=5.0, seed=42)
        >>> x
        tensor([[ 4.305-1.876j, -3.980-2.653j,  2.109-2.263j, -4.461-2.021j,
                 -0.175-3.119j],
                [-3.501+1.663j,  1.904-1.670j, -3.983-1.995j, -0.504+4.823j,
                 -1.270-1.310j]])
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

    # generate random magnitude with values in [0, rmax[, the sqrt ensures that the
    # resulting complex numbers are uniformly distributed in the complex plane
    r = rmax * rand().sqrt()
    # generate random phase with values in [0, 2pi[
    theta = 2 * torch.pi * rand()
    x = r * torch.exp(1j * theta)

    x.requires_grad = requires_grad

    return x


def pwc_pulse(
    t_start: float, t_end: float, values: Tensor
) -> callable[[float], Tensor]:
    r"""Returns a function that takes a time $t$ and returns the piecewise-constant
    (PWC) pulse value at this time.

    The time interval $[t_\text{start}, t_\text{end}]$ is split into $n$ bins, where
    $n$ is the last dimension of `values` with shape _(..., n)_. The pulse value at
    time $t$ is defined by

    - `torch.zeros(...)` if $t$ is not in $[t_\text{start}, t_\text{end}]$,
    - `values[..., k]` if $t$ is in the $k$-th bin of $[t_\text{start}, t_\text{end}]$.

    Notes:
        You can use [rand_complex()][dynamiqs.rand_complex] to generate a tensor
        filled with random complex numbers for the parameter `values`.

    Args:
        t_start: Start time of the pulse.
        t_end: End time of the pulse.
        values _(..., n)_: Pulse complex values where `n` is the number of different
            values between `t_start` and `t_end`.

    Returns:
        Function that takes a time $t$ and returns the pulse value at time $t$ (a
            tensor of shape _(...)_).

    Examples:
        >>> values = dq.rand_complex((2, 5), rmax=5.0, seed=42)
        >>> pulse = dq.pwc_pulse(0.0, 1.0, values)
        >>> type(pulse)
        <class 'function'>
        >>> pulse(0.5)
        tensor([ 2.109-2.263j, -3.983-1.995j])
        >>> pulse(1.2)
        tensor([0.+0.j, 0.+0.j])
    """

    def pulse(t):
        if t < t_start or t > t_end:
            # return a null tensor of appropriate shape
            batch_sizes = values.shape[:-1]
            return torch.zeros(batch_sizes, dtype=values.dtype, device=values.device)
        elif t == t_end:
            # return the last value, this case if necessary to avoid an out of bounds
            # index error when computing `idx` below
            return values[..., -1]
        else:
            # find the index corresponding to time t for t in [t_start, t_end[
            n = values.size(-1)
            tbin = (t_end - t_start) / n
            idx = int(t / tbin)
            return values[..., idx]

    return pulse
