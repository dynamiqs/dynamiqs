from __future__ import annotations

import equinox as eqx

from ._utils import tree_str_inline

__all__ = [
    'Direct',
    'BackwardCheckpointed',
    'Forward',
    'HigherOrder',
    'Implicit',
    'Gradient',
]


class Gradient(eqx.Module):
    """Base class of all gradients.

    Every gradient of `dq.gradient` is an instance of this class, which is useful for
    type annotations.
    """

    def __str__(self) -> str:
        return tree_str_inline(self)


class Direct(Gradient):
    """Standard automatic differentiation of JAX.

    With this option, the gradient is computed by automatically differentiating
    through the internals of the solver.

    Note:
        For Diffrax-based methods, this falls back to the
        [`diffrax.DirectAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.DirectAdjoint)
        option.

    Note:
        For supported explicit Diffrax ODE methods, this option also supports
        higher-order automatic differentiation (e.g.
        [`jax.hessian`](https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html)).
        Prefer [`dq.gradient.HigherOrder`][dynamiqs.gradient.HigherOrder] to make that
        requirement explicit.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass


class BackwardCheckpointed(Gradient):
    """Checkpointed backward-mode automatic differentiation.

    With this option, the gradient is computed by automatically differentiating
    through the internals of the solver. The difference with the standard automatic
    differentiation (see [`dq.gradient.Direct`][dynamiqs.gradient.Direct]) is
    that a checkpointing scheme is used to reduce the memory usage of the
    backpropagation.

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

    Note:
        For most problems this is the preferred technique for backpropagating
        through the solvers.

    Warning:
        This cannot be forward-mode autodifferentiated (e.g. using
        [`jax.jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)).
        Try using [`dq.gradient.Direct`][dynamiqs.gradient.Direct] if that is
        something you need.

    Note:
        For Diffrax-based methods, this falls back to the
        [`diffrax.RecursiveCheckpointAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.RecursiveCheckpointAdjoint)
        option.
    """

    ncheckpoints: int | None = None

    # dummy init to have the signature in the documentation
    def __init__(self, ncheckpoints: int | None = None):
        self.ncheckpoints = ncheckpoints


class Forward(Gradient):
    """Forward-mode automatic differentiation.

    Enables support for forward-mode automatic differentiation
    (like [`jax.jvp`](https://docs.jax.dev/en/latest/_autosummary/jax.jvp.html)
    or [`jax.jacfwd`](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html)).

    Note:
        This is the most efficient when the function has more outputs than inputs. For
        instance, it's the preferred method when simulating a Lindbladian parameterized
        with a few values and computing the Jacobian of a function returning the
        expectation values of many observables (or the same observable at many different
        times).

    Warning:
        This cannot be backward-mode autodifferentiated (e.g. using
        [`jax.jacrev`](https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html)).
        Try using
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed] if
        that is something you need.

    Warning:
        By default
        [`jax.grad`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html)
        uses reverse mode. Use
        [`jax.jacfwd`](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html)
        to compute the gradient in forward mode.

    Note:
        For Diffrax-based methods, this falls back to the
        [`diffrax.ForwardMode`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.ForwardMode)
        option.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass


class HigherOrder(Gradient):
    """Higher-order automatic differentiation.

    Enables support for higher-order automatic differentiation, such as
    [`jax.hessian`](https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html).

    Note:
        This option is supported by the explicit Diffrax ODE methods
        [`dq.method.Euler`][dynamiqs.method.Euler],
        [`dq.method.Dopri5`][dynamiqs.method.Dopri5],
        [`dq.method.Dopri8`][dynamiqs.method.Dopri8] and
        [`dq.method.Tsit5`][dynamiqs.method.Tsit5]. Like
        [`dq.gradient.Direct`][dynamiqs.gradient.Direct], it uses
        [`diffrax.DirectAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.DirectAdjoint),
        which already configures the solver for higher-order differentiation. Use this
        option to make the higher-order differentiation requirement explicit and to
        restrict to the methods on which it is supported.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass


class Implicit(Gradient):
    r"""Implicit differentiation of the steady state.

    With this option, the gradient is computed from the equation satisfied by the
    steady state $\rho_\infty$, without differentiating through the time integration
    that reached it. By the implicit function theorem, the derivative of
    $\rho_\infty$ with respect to a parameter $\theta$ of $H$ or $L_k$ solves the
    linear system
    $$
        \tilde{\mathcal{L}}\left(\frac{\partial\rho_\infty}{\partial\theta}\right)
        = -\frac{\partial\mathcal{L}}{\partial\theta}(\rho_\infty),
    $$
    where $\tilde{\mathcal{L}}(X) = \mathcal{L}(X) - \tr{X} I/n$ is the Lindbladian
    deflated along the trace, which is invertible when the steady state is unique
    (the Lindbladian $\mathcal{L}$ itself is singular).

    Note:
        This is the most accurate and the cheapest option for
        [`dq.mesteadystate()`][dynamiqs.mesteadystate]: the cost does not depend on
        the number of integration steps, and both forward and reverse modes are
        supported.

    Warning:
        The gradient is only valid for a unique steady state. Otherwise, use a
        gradient supported by the time integration method, e.g.
        [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed].
        The linear system is solved by dense LU decomposition of an $n^2\times n^2$
        matrix, which scales as $\mathcal{O}(n^6)$.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass
