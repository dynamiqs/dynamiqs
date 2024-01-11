from __future__ import annotations

from typing import Callable, Union

import diffrax as dx
from jaxtyping import PyTree, Scalar
from jax.experimental import host_callback
from tqdm import tqdm


class ProgressBarTerm(dx.ODETerm):
    update_progressbar: Union[Callable, None]

    def __init__(self, update_progressbar: Union[Callable, None]):
        self.update_progressbar = update_progressbar

    def vector_field(self, t: Scalar, _psi: PyTree, _args: PyTree):
        if self.update_progressbar is not None:
            host_callback.id_tap(self.update_progressbar, t)


def make_progressbar(progress_bar: bool, tfinal: float):
    if progress_bar:
        bar = tqdm(total=100)

        def update_progressbar(t, _):
            bar.n = int(t / tfinal * 100)
            bar.refresh()

        return bar, update_progressbar
    else:
        return None, None


def close_progressbar(bar: tqdm | None):
    if bar is not None:
        bar.close()
