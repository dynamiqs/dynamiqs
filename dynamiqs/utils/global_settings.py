from __future__ import annotations

from typing import Literal

import jax

from ..progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter
from ..qarrays.layout import dense, dia, set_global_layout

__all__ = [
    'set_device',
    'set_layout',
    'set_matmul_precision',
    'set_precision',
    'set_progress_meter',
]

_DEFAULT_PROGRESS_METER: AbstractProgressMeter = TqdmProgressMeter()


def set_device(device: Literal['cpu', 'gpu', 'tpu'], index: int = 0):
    """Configure the default device.

    See [JAX documentation on devices](https://docs.jax.dev/en/latest/faq.html#faq-data-placement).

    Args:
        device (string 'cpu', 'gpu', or 'tpu'): Default device.
        index: Index of the device to use, defaults to 0.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        jax.config.update('jax_default_device', jax.devices(device)[index])
        ```

    Examples:
        >>> dq.set_device('cpu')
        >>> jnp.ones(2).device
        CpuDevice(id=0)
        >>> jax.config.update('jax_default_device', None)  # back to default device
    """
    jax.config.update('jax_default_device', jax.devices(device)[index])


def set_precision(precision: Literal['single', 'double']):
    """Configure the default floating point precision.

    Two options are available:

    - `'single'` sets default precision to `float32` and `complex64` (default setting),
    - `'double'` sets default precision to `float64` and `complex128`.

    Args:
        precision (string 'single' or 'double'): Default precision.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        if precision == 'single':
            jax.config.update('jax_enable_x64', False)
        elif precision == 'double':
            jax.config.update('jax_enable_x64', True)
        ```
        See [JAX documentation on double precision](https://docs.jax.dev/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

    Examples:
        >>> dq.eye(2).dtype
        dtype('complex64')
        >>> dq.set_precision('double')
        >>> dq.eye(2).dtype
        dtype('complex128')
        >>> dq.set_precision('single')  # back to default precision
    """
    if precision == 'single':
        jax.config.update('jax_enable_x64', False)
    elif precision == 'double':
        jax.config.update('jax_enable_x64', True)
    else:
        raise ValueError(
            f"Argument `precision` must be a string 'single' or 'double', but is"
            f" '{precision}'."
        )


def set_matmul_precision(matmul_precision: Literal['low', 'high', 'highest']):
    """Configure the default precision for matrix multiplications on GPUs and TPUs.

    Some devices allow trading off accuracy for speed when performing matrix
    multiplications (matmul). Three options are available:

    - `'low'` reduces matmul precision to `bfloat16` (fastest but least accurate),
    - `'high'` reduces matmul precision to `bfloat16_3x` or `tensorfloat32` if available
        (faster but less accurate),
    - `'highest'` keeps matmul precision at `float32` or `float64` as applicable
        (slowest but most accurate, default setting).

    Args:
        matmul_precision (string 'low', 'high', or 'highest'): Default precision
            for matrix multiplications on GPUs and TPUs.

    Note:
        This setting applies only to single precision matrices (`float32` or
        `complex64`).

    Note-: Equivalent JAX syntax
        This function is equivalent to setting `jax_default_matmul_precision` in
        `jax.config`. See [JAX documentation on matmul precision](https://docs.jax.dev/en/latest/_autosummary/jax.default_matmul_precision.html)
        and [JAX documentation on the different available options](https://docs.jax.dev/en/latest/jax.lax.html#jax.lax.Precision).

    Examples:
        >>> dq.set_matmul_precision('low')
        >>> dq.set_matmul_precision('highest')  # back to default precision
    """
    if matmul_precision == 'low':
        jax.config.update('jax_default_matmul_precision', 'bfloat16')
    elif matmul_precision == 'high':
        jax.config.update('jax_default_matmul_precision', 'high')
    elif matmul_precision == 'highest':
        jax.config.update('jax_default_matmul_precision', 'highest')
    else:
        raise ValueError(
            f"Argument `matmul_precision` must be a string 'low', 'high', or"
            f" 'highest', but is '{matmul_precision}'."
        )


def set_layout(layout: Literal['dense', 'dia']):
    """Configure the default matrix layout for operators supporting this option.

    Two layouts are supported by most operators (see the list of available operators in
    the [Python API](../../index.md#operators)):

    - `'dense'`: JAX native dense layout,
    - `'dia'`: Dynamiqs sparse diagonal layout, only non-zero diagonals are stored.

    Args:
        layout (string 'dense' or 'dia'): Default matrix layout for operators.

    Note:
        The default layout upon importing Dynamiqs is `'dia'`.

    Examples:
        >>> dq.eye(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]
        >>> dq.set_layout('dense')
        >>> dq.eye(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]
        >>> dq.set_layout('dia')  # back to default layout
    """
    layouts = {'dense': dense, 'dia': dia}
    if layout not in layouts:
        raise ValueError(
            f"Argument `layout` must be a string 'dense' or 'dia', but is {layout}."
        )

    set_global_layout(layouts[layout])


def set_progress_meter(progress_meter: AbstractProgressMeter | bool):
    """Configure the default progress meter.

    Args:
        progress_meter: Default progress meter. Set to `True` for a
            [`dq.TqdmProgressMeter`][dynamiqs.TqdmProgressMeter], and `False` for a
            [`dq.NoProgressMeter`][dynamiqs.NoProgressMeter]. Any other progress meter
            can also be passed, such as
            [`dq.TextProgressMeter`][dynamiqs.TextProgressMeter].

    Examples:
        >>> dq.set_progress_meter(False)
        >>> dq.set_progress_meter(True)  # back to default progress meter
    """
    global _DEFAULT_PROGRESS_METER  # noqa: PLW0603

    if progress_meter is True:
        progress_meter = TqdmProgressMeter()
    elif progress_meter is False:
        progress_meter = NoProgressMeter()

    _DEFAULT_PROGRESS_METER = progress_meter


def get_progress_meter(
    progress_meter: AbstractProgressMeter | bool | None,
) -> AbstractProgressMeter:
    if progress_meter is None:
        return _DEFAULT_PROGRESS_METER
    elif progress_meter is True:
        return TqdmProgressMeter()
    elif progress_meter is False:
        return NoProgressMeter()
    return progress_meter
