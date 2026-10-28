from __future__ import annotations

from enum import Enum

__all__ = ['dense', 'dia']


class Layout(Enum):
    """Data layout of a qarray.

    The two layouts are [`dq.dense`][dynamiqs.dense] and [`dq.dia`][dynamiqs.dia].
    Operator constructors such as [`dq.eye()`][dynamiqs.eye] take a `layout`
    argument, whose default is set globally with
    [`dq.set_layout()`][dynamiqs.set_layout].
    """

    DENSE = 'dense'
    DIA = 'dia'

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return repr(self)


dense = Layout.DENSE
"""Dense layout, storing all the matrix elements of a qarray in a JAX array."""

dia = Layout.DIA
"""Sparse diagonal layout, storing only the non-zero diagonals of a qarray.

Efficient for operators with few non-zero diagonals, such as the bosonic annihilation
and creation operators. This is the default layout of the operator constructors, see
[`dq.set_layout()`][dynamiqs.set_layout].
"""

_DEFAULT_LAYOUT = dia


def set_global_layout(layout: Layout):
    global _DEFAULT_LAYOUT  # noqa: PLW0603
    _DEFAULT_LAYOUT = layout


def get_layout(layout: Layout | None = None) -> Layout:
    if layout is None:
        return _DEFAULT_LAYOUT
    elif isinstance(layout, Layout):
        return layout
    else:
        raise TypeError(
            'Argument `layout` must be `dq.dense`, `dq.dia` or `None`, but is'
            f' `{layout}`.'
        )


def promote_layouts(layout1: Layout, layout2: Layout) -> Layout:
    if layout1 is dia and layout2 is dia:
        return dia
    else:
        return dense
