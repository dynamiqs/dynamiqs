from __future__ import annotations

import torch
from torch import Tensor

from .tensor_types import dtype_complex_to_real, get_cdtype, to_device

__all__ = [
    'rand_complex',
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
        size: Size of the returned tensor.
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
