from __future__ import annotations

import equinox as eqx
import jax
from jax import Array
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter

__all__ = ['Options']


class Options(eqx.Module):
    """Generic options for the quantum solvers.

    Args:
        save_states: If `True`, the state is saved at every time in `tsave`,
            otherwise only the final state is returned.
        verbose: If `True`, print information about the integration, otherwise
            nothing is printed.
        cartesian_batching: If `True`, batched arguments are treated as separated
            batch dimensions, otherwise the batching is performed over a single
            shared batched dimension.
        progress_meter: Progress meter indicating how far the solve has progressed.
            Defaults to a [tqdm](https://github.com/tqdm/tqdm) progress meter. Pass
            `None` for no output, see other options in
            [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
            If gradients are computed, the progress meter only displays during the
            forward pass.
        t0: Initial time. If `None`, defaults to the first time in `tsave`.
        t1: Final time. If `None`, defaults to the last time in `tsave`.
        save_extra _(function, optional)_: A function with signature
            `f(Array) -> PyTree` that takes a state as input and returns a PyTree.
            This can be used to save additional arbitrary data during the
            integration.
        target_fidelity: if this fidelity is reached, stop grape optimization
        epochs: number of epochs to loop over in grape
        coherent: If true, use coherent definition of the fidelity which
                  accounts for relative phases. If not, use incoherent
                  definition
        ntraj: number of trajectories for mcsolve
        one_jump_only: should we do mcsolve with only a single jump
    """

    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter()
    t0: ScalarLike | None = None
    t1: ScalarLike | None
    save_extra: callable[[Array], PyTree] | None = None
    target_fidelity: float
    epochs: int
    coherent: bool
    ntraj: int
    one_jump_only: bool

    def __init__(
        self,
        save_states: bool = True,
        verbose: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),  # noqa: B008
        t0: ScalarLike | None = None,
        t1: ScalarLike | None = None,
        save_extra: callable[[Array], PyTree] | None = None,
        target_fidelity: float = 0.9995,
        epochs: int = 1000,
        coherent: bool = True,
        ntraj: int = 10,
        one_jump_only: bool = True
    ):
        if progress_meter is None:
            progress_meter = NoProgressMeter()

        self.save_states = save_states
        self.verbose = verbose
        self.cartesian_batching = cartesian_batching
        self.progress_meter = progress_meter
        self.t0 = t0
        self.t1 = t1
        self.target_fidelity = target_fidelity
        self.epochs = epochs
        self.coherent = coherent
        self.ntraj = ntraj
        self.one_jump_only = one_jump_only

        # make `save_extra` a valid Pytree with `jax.tree_util.Partial`
        if save_extra is not None:
            save_extra = jax.tree_util.Partial(save_extra)
        self.save_extra = save_extra

    def __str__(self) -> str:
        return tree_str_inline(self)
