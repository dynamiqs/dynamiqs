from __future__ import annotations


import diffrax as dx
from jaxtyping import PyTree, Scalar
from jax.experimental import host_callback
from tqdm import tqdm


class ProgressBarTerm(dx.ODETerm):
    update_progressbar: callable | None

    def __init__(self, update_progressbar: callable | None):
        self.update_progressbar = update_progressbar

    def vector_field(self, t: Scalar, _psi: PyTree, _args: PyTree):
        if self.update_progressbar is not None:
            host_callback.id_tap(self.update_progressbar, t)


def make_progressbar(
    verbose: bool, tinitial: float, tfinal: float
) -> (tqdm | None, callable | None):
    if verbose:
        bar = tqdm(total=100)

        def update_progressbar(t, _):
            bar.n = int((t - tinitial) / tfinal * 100)
            bar.refresh()

        return bar, update_progressbar
    else:
        return None, None


def close_progressbar(bar: tqdm | None):
    if bar is not None:
        bar.close()
