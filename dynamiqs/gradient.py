from __future__ import annotations

import equinox as eqx

__all__ = ['Autograd', 'CheckpointAutograd']

class Gradient(eqx.Module):
    pass


class Autograd(Gradient):
    def __init__(self):
        """JAX's default automatic differentiation.

        For more information, see the [JAX documentation](https://jax.readthedocs.io/en/
        latest/notebooks/quickstart.html#taking-derivatives-with-grad). For Diffrax
        solvers, this falls back to `diffrax.DirectAdjoint` differentation.

        Note: _From the Diffrax documentation:_
            A variant of `diffrax.RecursiveCheckpointAdjoint`. The differences are that
            `DirectAdjoint`:

            - Is less time+memory efficient at reverse-mode autodifferentiation
                (specifically, these will increase every time max_steps increases
                passes a power of 16);
            - Cannot be reverse-mode autodifferentated if max_steps is None;
                Supports forward-mode autodifferentiation.

            So unless you need forward-mode autodifferentiation then `diffrax.
            RecursiveCheckpointAdjoint` should be preferred.
        """
        super().__init__()


class CheckpointAutograd(Gradient):
    ncheckpoints: int | None = None

    def __init__(self, ncheckpoints: int | None = None):
        """Diffrax's `RecursiveCheckpointAdjoint` automatic differentation.

        Note: _From the Diffrax documentation:_
            Backpropagate through `diffrax.diffeqsolve` by differentiating the numerical
            solution directly. This is sometimes known as "discretise-then-optimise", or
            described as "backpropagation through the solver".

            Uses a binomial checkpointing scheme to keep memory usage low.

            For most problems this is the preferred technique for backpropagating
            through a differential equation.

            Note that this cannot be forward-mode autodifferentiated (e.g. using
                [`jax.jvp`](https://jax.readthedocs.io/en/latest/_autosummary/
                jax.jvp.html)). Try using `diffrax.DirectAdjoint` if that is something
                you need.

        Args:
            ncheckpoints: Number of checkpoints to use. The amount of memory used by
                the differential equation solve will be roughly equal to the number of
                checkpoints multiplied by the size of the initial state.
                You can speed up backpropagation by allocating more checkpoints, so it
                makes sense to set as many checkpoints as you have memory for. This
                value is set to `None` by default, in which case it will be set to
                log(max_steps), for which a theoretical result is available guaranteeing
                that backpropagation will take O(n log n) time in the number of steps n.
        """
        super().__init__()
