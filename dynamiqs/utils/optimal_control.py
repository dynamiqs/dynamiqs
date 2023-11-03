from __future__ import annotations

import torch
from torch import Tensor

from .._utils import check_time_tensor
from ..utils.operators import displace
from ..utils.states import fock
from ..utils.utils import tensprod, tobra
from .tensor_types import dtype_complex_to_real, get_cdtype, to_device

__all__ = ['rand_complex', 'pwc_pulse', 'snap_gate', 'cd_gate']


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


def pwc_pulse(times: Tensor, values: Tensor) -> callable[[float], Tensor]:
    r"""Returns a function that takes a time $t$ and returns the corresponding
    piecewise-constant pulse value.

    The `times` tensor $(t_0,\dots,t_n)$ of shape _(n+1)_ defines the time interval for
    each element of the `values` tensor of shape _(..., n)_. The pulse value at time
    $t$ is defined by

    - `torch.zeros(...)` if $t<t_0$ or $t>=t_n$,
    - `values[..., k]` if $t\in[t_k, t_{k+1})$.

    Notes:
        You can use [rand_complex()][dynamiqs.rand_complex] to generate a tensor
        filled with random complex numbers for the parameter `values`.

    Args:
        times _(n+1)_: Time points between which the pulse takes constant values.
        values _(..., n)_: Pulse complex values.

    Returns:
        Function with signature `float -> Tensor` that takes a time $t$ and returns the
            corresponding pulse value, a tensor of shape _(...)_.

    Examples:
        >>> times = torch.linspace(0.0, 1.0, 11)
        >>> values = dq.rand_complex((2, 10), rmax=5.0, seed=42)
        >>> pulse = dq.pwc_pulse(times, values)
        >>> type(pulse)
        <class 'function'>
        >>> pulse(0.5)
        tensor([-0.474+3.847j,  1.186-3.054j])
        >>> pulse(1.2)
        tensor([0.+0.j, 0.+0.j])
    """

    check_time_tensor(times, 'times')

    def pulse(t):
        if t < times[0] or t >= times[-1]:
            # return a null tensor of appropriate shape
            batch_sizes = values.shape[:-1]
            return torch.zeros(batch_sizes, dtype=values.dtype, device=values.device)
        else:
            # find the index $k$ such that $t \in [t_k, t_{k+1})$
            idx = torch.searchsorted(times, t)
            return values[..., idx]

    return pulse


def snap_gate(
    phase: Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns a SNAP gate.

    The *selective number-dependent arbitrary phase* (SNAP) gate imparts a different
    phase $\theta_k$ to each Fock state $\ket{k}\bra{k}$. It is defined by
    $$
        \mathrm{SNAP}(\theta_0,\dots,\theta_{n-1}) =
        \sum_{k=0}^{n-1} e^{i\theta_k} \ket{k}\bra{k}.
    $$

    Args:
        phase _(..., n)_: Phase for each Fock state. The size of the last tensor
            dimension _n_ defines the Hilbert space dimension.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(..., n, n)_ SNAP gate operator.

    Examples:
        >>> dq.snap_gate(torch.tensor([0, 1, 2]))
        tensor([[ 1.000+0.000j,  0.000+0.000j,  0.000+0.000j],
                [ 0.000+0.000j,  0.540+0.841j,  0.000+0.000j],
                [ 0.000+0.000j,  0.000+0.000j, -0.416+0.909j]])
        >>> dq.snap_gate(torch.tensor([[0, 1, 2], [2, 3, 4]])).shape
        torch.Size([2, 3, 3])
    """
    cdtype = get_cdtype(dtype)
    rdtype = dtype_complex_to_real(cdtype)
    phase = phase.to(dtype=rdtype, device=device)
    return torch.diag_embed(torch.exp(1j * phase))


def cd_gate(
    dim: int,
    alpha: Tensor,
    *,
    dtype: torch.complex64 | torch.complex128 | None = None,
    device: str | torch.device | None = None,
) -> Tensor:
    r"""Returns a conditional displacement gate.

    The *conditional displacement* (CD) gate displaces an oscillator conditioned on
    the state of a coupled two-level system (TLS) state. It is defined by
    $$
       \mathrm{CD}(\alpha) = D(\alpha/2)\ket{g}\bra{g} + D(-\alpha/2)\ket{e}\bra{e},
    $$
    where $\ket{g}=\ket0$ and $\ket{e}=\ket1$ are the ground and excited states of the
    TLS, respectively.

    Args:
        dim: Dimension of the oscillator Hilbert space.
        alpha _(...)_: Displacement amplitude.
        dtype: Complex data type of the returned tensor.
        device: Device of the returned tensor.

    Returns:
        _(..., n, n)_ CD gate operator (acting on the oscillator + TLS system of
            dimension _n = 2 x dim_).

    Examples:
        >>> dq.cd_gate(3, torch.tensor([0.1]))
        tensor([[[ 0.999+0.j,  0.000+0.j, -0.050+0.j,  0.000+0.j,  0.002+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.999+0.j,  0.000+0.j,  0.050+0.j,  0.000+0.j,  0.002+0.j],
                 [ 0.050+0.j,  0.000+0.j,  0.996+0.j,  0.000+0.j, -0.071+0.j,  0.000+0.j],
                 [ 0.000+0.j, -0.050+0.j,  0.000+0.j,  0.996+0.j,  0.000+0.j,  0.071+0.j],
                 [ 0.002+0.j,  0.000+0.j,  0.071+0.j,  0.000+0.j,  0.998+0.j,  0.000+0.j],
                 [ 0.000+0.j,  0.002+0.j,  0.000+0.j, -0.071+0.j,  0.000+0.j,  0.998+0.j]]])
        >>> dq.cd_gate(3, torch.tensor([0.1, 0.2])).shape
        torch.Size([2, 6, 6])
    """  # noqa: E501
    g = fock(2, 0, dtype=dtype, device=device).repeat(len(alpha), 1, 1)  # (n, 2, 1)
    e = fock(2, 1, dtype=dtype, device=device).repeat(len(alpha), 1, 1)  # (n, 2, 1)
    disp_plus = displace(dim, alpha / 2, dtype=dtype, device=device)  # (n, dim, dim)
    disp_minus = displace(dim, -alpha / 2, dtype=dtype, device=device)  # (n, dim, dim)
    return tensprod(disp_plus, g @ tobra(g)) + tensprod(disp_minus, e @ tobra(e))
