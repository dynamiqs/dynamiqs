from __future__ import annotations

from typing import Literal

import jax

from .._utils import obj_type_str
from ..progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter
from ..qarrays.composite_qarray import set_global_composite
from ..qarrays.layout import dense, dia, set_global_layout

__all__ = [
    'set_composite',
    'set_device',
    'set_layout',
    'set_matmul_precision',
    'set_precision',
    'set_progress_meter',
]

_DEFAULT_PROGRESS_METER: AbstractProgressMeter = TqdmProgressMeter()


def set_device(device: Literal['cpu', 'gpu', 'tpu'], index: int = 0):
    """Configure the default device.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        jax.config.update('jax_default_device', jax.devices(device)[index])
        ```

    See [JAX documentation on devices](https://jax.readthedocs.io/en/latest/faq.html#faq-data-placement).

    Args:
        device (string 'cpu', 'gpu', or 'tpu'): Default device.
        index: Index of the device to use, defaults to 0.
    """
    jax.config.update('jax_default_device', jax.devices(device)[index])


def set_precision(precision: Literal['single', 'double']):
    """Configure the default floating point precision.

    Two options are available:

    - `'single'` sets default precision to `float32` and `complex64` (default setting),
    - `'double'` sets default precision to `float64` and `complex128`.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        ```
        if precision == 'single':
            jax.config.update('jax_enable_x64', False)
        elif precision == 'double':
            jax.config.update('jax_enable_x64', True)
        ```
         See [JAX documentation on double precision](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision).

    Args:
        precision (string 'single' or 'double'): Default precision.
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
    - `'highest'` keeps matmul precision to `float32` or `float64` as applicable
        (slowest but most accurate, default setting).

    Note:
        This setting applies only to single precision matrices (`float32` or
        `complex64`).

    Note-: Equivalent JAX syntax
        This function is equivalent to setting `jax_default_matmul_precision` in
        `jax.config`. See [JAX documentation on matmul precision](https://jax.readthedocs.io/en/latest/_autosummary/jax.default_matmul_precision.html)
        and [JAX documentation on the different available options](https://jax.readthedocs.io/en/latest/jax.lax.html#jax.lax.Precision).

    Args:
        matmul_precision (string 'low', 'high', or 'highest'): Default precision
            for matrix multiplications on GPUs and TPUs.
    """
    if matmul_precision == 'low':
        jax.config.update('jax_default_matmul_precision', 'fastest')
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
    the [Python API](/python_api/index.html#operators))):

    - `'dense'`: JAX native dense layout,
    - `'dia'`: dynamiqs sparse diagonal layout, only non-zero diagonals are stored.

    Note:
        The default layout upon importing dynamiqs is `'dia'`.

    Args:
        layout (string 'dense' or 'dia'): Default matrix layout for operators.

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


def set_composite(composite: bool):
    r"""Configure whether tensor products of operators are stored as their factors.

    If `True`, [`dq.tensor()`][dynamiqs.tensor] stores a tensor product of operators
    $A_1\otimes\dots\otimes A_N$ as its factors, without building the $n\times n$
    matrix, and so do the multi-mode operators built from it, such as
    `dq.destroy(*dims)`. Such a composite qarray behaves exactly like the full matrix,
    and the solvers apply it to the state mode by mode. Kets, bras and the states
    returned by the state functions are always stored as full matrices, and the
    solvers convert a composite initial density matrix to a full matrix.

    Composite qarrays pay off when the operators acting on each subsystem are dense:
    [`dq.mesolve()`][dynamiqs.mesolve] then runs up to 10 times faster than with the
    full dense matrix. On sparse operators, such as bosonic modes or qubit chains, the
    full matrix in the `dq.dia` layout is faster. The full matrix is built anyway with
    `vectorized=True`, the `Expm` method, [`dq.mepropagator()`][dynamiqs.mepropagator]
    and the jump solvers.

    Note:
        The default upon importing dynamiqs is `False`.

    Args:
        composite: Whether to store tensor products of operators as their factors.

    Examples:
        >>> dq.set_composite(True)
        >>> a, b = dq.destroy(3, 4)
        >>> H = a.dag() @ a + b.dag() @ b
        >>> H
        CompositeQArray: shape=(12, 12), dims=(3, 4), dtype=complex64, layout=dia, n_terms=2
          term[0]: coeff=1.0, op_dims=((3,), (4,))
          term[1]: coeff=1.0, op_dims=((3,), (4,))
        >>> dq.set_composite(False)  # back to default
    """  # noqa: E501
    if not isinstance(composite, bool):
        raise TypeError(
            'Argument `composite` must be `True` or `False`, but has type'
            f' {obj_type_str(composite)}.'
        )

    set_global_composite(composite)


def set_progress_meter(progress_meter: AbstractProgressMeter | bool):
    """Configure the default progress meter.

    Args:
        progress_meter: Default progress meter. Set to `True` for a [tqdm](https://github.com/tqdm/tqdm)
            progress meter, and `False` for no output. See other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
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
