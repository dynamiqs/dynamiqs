from __future__ import annotations

import equinox as eqx

from ._utils import tree_str_inline

__all__ = ['Autograd', 'CheckpointAutograd']


class Gradient(eqx.Module):
    def __str__(self) -> str:
        return tree_str_inline(self)


class Autograd(Gradient):
    """Standard automatic differentiation of JAX.

    With this option, the gradient is computed by automatically differentiating
    through the internals of the solver.

    Note:
        For Diffrax-based solvers, this falls back to the
        [`diffrax.DirectAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.DirectAdjoint)
        option.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass


class CheckpointAutograd(Gradient):
    """Checkpointed automatic differentiation.

    With this option, the gradient is computed by automatically differentiating
    through the internals of the solver. The difference with the standard automatic
    differentiation (see [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd]) is
    that a checkpointing scheme is used to reduce the memory usage of the
    backpropagation.

    Note:
        For most problems this is the preferred technique for backpropagating
        through the quantum solvers.

    Warning:
        This cannot be forward-mode autodifferentiated (e.g. using
        [`jax.jvp`](https://jax.readthedocs.io/en/latest/_autosummary/jax.jvp.html)
        ). Try using [`dq.gradient.Autograd`][dynamiqs.gradient.Autograd] if that
        is something you need.


    Note:
        For Diffrax-based solvers, this falls back to the
        [`diffrax.RecursiveCheckpointAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.RecursiveCheckpointAdjoint)
        option.

    Args:
        ncheckpoints: Number of checkpoints to use. The amount of memory used by
            the differential equation solve will be roughly equal to the number of
            checkpoints multiplied by the size of the state. You can speed up
            backpropagation by allocating more checkpoints, so it makes sense to
            set as many checkpoints as you have memory for. This value is set to
            `None` by default, in which case it will be set to `log(max_steps)`,
            for which a theoretical result is available guaranteeing that
            backpropagation will take `O(n_steps log(n_steps))` time in the number
            of steps `n_steps <= max_steps`.
    """

    ncheckpoints: int | None = None

    # dummy init to have the signature in the documentation
    def __init__(self, ncheckpoints: int | None = None):
        self.ncheckpoints = ncheckpoints
