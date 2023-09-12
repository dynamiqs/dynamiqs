from __future__ import annotations

import torch
from torch import Tensor

from .tensor_types import dtype_complex_to_real, get_cdtype, to_device

__all__ = [
    'rand_pwc',
]


def _rand_complex_uniform(
    size: list[int],
    rmax: float,
    generator: torch.Generator,
    dtype: torch.complex64 | torch.complex128,
    device: str | torch.device | None,
):
    rdtype = dtype_complex_to_real(get_cdtype(dtype))
    rand = lambda: torch.rand(size, generator=generator, dtype=rdtype, device=device)

    # generate random magnitude with values in [0, rmax], the sqrt ensures that the
    # resulting complex numbers are uniformly distributed in the complex plane
    r = rmax * rand().sqrt()
    # generate random phase with values in [0, 2pi]
    theta = 2 * torch.pi * rand()

    return r * torch.exp(1j * theta)


def rand_pwc(
    *size: int,
    amp_max: float = 1.0,
    requires_grad: bool = False,
    seed: int | None = None,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns a random piecewise-constant (PWC) pulse.

    Each element of the returned tensor is a complex number uniformly distributed in the
    complex plane with a magnitude between 0 and `amp_max` and a random phase.
    Formally, each element is defined by

    $$
        x = re^{i\theta}\ \text{with}\
        \left\{\begin{aligned}
        r      &= \texttt{amp_max} \cdot \sqrt{\texttt{rand(0,1)}} \\
        \theta &= 2\pi \cdot \texttt{rand(0,1)}
        \end{aligned}\right.
    $$

    where $\texttt{rand(0,1)}$ is a random number uniformly distributed between 0 and 1.

    Note:
        The square root in the definition of the magnitude $r$ ensures that the
        resulting complex numbers are uniformly distributed in the complex plane.

    Args:
        *size: Returned tensor dimensions.
        amp_max: Maximum pulse amplitude.
        requires_grad: Whether gradients need to be computed with respect to the
            returned tensor.
        seed: Seed for the random number generator.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(*size)_ Random PWC pulse.

    Examples:
        >>> pulse = dq.rand_pwc(10, amp_max=8.0, seed=42)
        >>> pulse
        tensor([ 6.889-3.002j, -6.367-4.245j,  3.375-3.621j, -7.137-3.234j,
                -0.280-4.991j, -5.601+2.661j,  3.047-2.671j, -6.372-3.192j,
                -0.807+7.717j, -2.032-2.096j])
    """
    # Note: We need to manually fetch the default device, because if `device` is `None`
    # `torch.Generator` picks "cpu" as the default device, and not the device set by
    # `torch.set_default_device`.
    device = to_device(device)

    # define random number generator from seed
    generator = torch.Generator(device=device)
    generator.seed() if seed is None else generator.manual_seed(seed)

    pwc_pulse = _rand_complex_uniform(size, amp_max, generator, dtype, device)
    pwc_pulse.requires_grad = requires_grad
    return pwc_pulse
