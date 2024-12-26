from __future__ import annotations

from enum import Enum


class Layout(Enum):
    DENSE = 'dense'
    DIA = 'dia'

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return repr(self)


dense = Layout.DENSE
dia = Layout.DIA

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
