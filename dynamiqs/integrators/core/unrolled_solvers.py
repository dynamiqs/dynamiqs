"""Explicit Runge-Kutta solvers of Diffrax, with faster stages on GPU and CPU.

Diffrax evaluates the stages of a Runge-Kutta step in a `while_loop` and combines them
with a `tensordot` over all the stages. On GPU, the loop costs one device-to-host
synchronisation per stage (XLA copies the loop condition back to the host before each
iteration), which dominates the cost of a step for the small and medium systems dynamiqs
solves; on CPU, XLA runs the complex `tensordot` as a matrix-vector product on one
thread. The solvers below take the same steps with the same tableaus:
- on GPU, they write the stages out as straight-line code, which also lets XLA fuse the
  stage combinations into the vector field evaluations;
- on CPU, they loop over the stages (unrolled, XLA would recompute stages there), and
  each stage combines only the stages it reads, with elementwise sums;
- on other backends, they are Diffrax's solvers.

The choice follows `jax.default_backend()` when tracing, not the device a solve runs on:
with `dq.set_device('cpu')` on a GPU machine, the stages are unrolled.

The first stage of an FSAL method is the last stage of the previous step. Diffrax
evaluates it inside the stage loop on the first step and after a jump; here `init`
evaluates it once, and a step re-evaluates it only after a jump, under a `lax.cond`
that only exists when the solve has jump times.

Other terms (e.g. expensive vector fields or implicit tableaus) fall back to Diffrax's
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from functools import cache
from typing import ClassVar, cast

import diffrax as dx
import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from diffrax._custom_types import Args, BoolScalarLike, RealScalarLike, Y
from jax.experimental import layout
from jaxtyping import Array, ArrayLike, PyTree


def _combine(
    y0: PyTree, coefficients: Sequence[float] | np.ndarray, ks: list[PyTree]
) -> PyTree:
    # y0 + sum_j coefficients[j] * ks[j], skipping the zero coefficients
    def leaf(y: ArrayLike, *k: ArrayLike) -> ArrayLike:
        total = y
        for a, kj in zip(coefficients, k, strict=True):
            if a != 0:
                total = total + float(a) * kj
        return total

    return jtu.tree_map(leaf, y0, *ks)


def _stage_time(
    t0: RealScalarLike, t1: RealScalarLike, dt: RealScalarLike, c: float
) -> RealScalarLike:
    # as Diffrax: a stage at c = 1 is exactly at t1
    return t1 if c == 1 else t0 + float(c) * dt


def _dot(coefficients: Sequence[float] | np.ndarray, ks: list[PyTree]) -> PyTree:
    # sum_j coefficients[j] * ks[j], skipping the zero coefficients
    def leaf(*k: ArrayLike) -> ArrayLike:
        terms = [float(a) * kj for a, kj in zip(coefficients, k, strict=True) if a != 0]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    return jtu.tree_map(leaf, *ks)


def _row_major(x: PyTree) -> PyTree:
    # x, with each array laid out row-major (JAX >= 0.7; earlier versions named their
    # layout API differently)
    if jax.__version_info__ < (0, 7):
        return x
    return jtu.tree_map(
        lambda a: layout.with_layout_constraint(
            a, layout.Layout(tuple(range(jnp.ndim(a))))
        ),
        x,
    )


# The size from which an unrolled step writes its stages out with an odd number of
# elements (see `_odd_sized`), tuned on an A100: about the size from which XLA unrolls
# GPU kernels there, the number of threads it runs at once (108 multiprocessors x
# 2048).
_MIN_SIZE_ODD_STAGES = 2**18


def _flatten_odd(x: Array) -> Array:
    # x flattened, with a trailing zero if it has an even number of elements
    x = jnp.ravel(x)
    return jnp.pad(x, (0, 1 - x.size % 2))


def _odd_sized(fn: Callable, *xs: Array) -> tuple[Array, ...]:
    # fn(*xs) for an elementwise fn of arrays of one shape that returns a tuple of
    # arrays of that shape, with its outputs written out flattened to an odd number of
    # elements. XLA unrolls an elementwise GPU kernel by gcd(number of elements, 4),
    # and for complex arrays the unrolled kernels that evaluate a vector field reading
    # its input at many offsets (the fused DIA Lindbladian) and combine it into the
    # next stage input were 1.5-3x slower on an A100 (openxla/xla#50467). The barrier
    # keeps XLA from slicing the outputs back to their shape before writing them.
    # Under vmap, the whole batch is flattened, for an odd number of elements at any
    # batch size; a batch sharded across devices is then no longer split evenly, and
    # XLA moves data between them. The tangents are those of fn.

    @jax.custom_batching.custom_vmap
    def odd_sized(*xs: Array) -> tuple[Array, ...]:
        out = jax.lax.optimization_barrier(fn(*map(_flatten_odd, xs)))
        return tuple(x[: xs[0].size].reshape(xs[0].shape) for x in out)

    @odd_sized.def_vmap
    def odd_sized_vmap(
        axis_size: int, in_batched: list[bool], *xs: Array
    ) -> tuple[tuple[Array, ...], tuple[bool, ...]]:
        xs = tuple(
            x if batched else jnp.broadcast_to(x, (axis_size, *x.shape))
            for x, batched in zip(xs, in_batched, strict=True)
        )
        out = odd_sized(*xs)
        return out, (True,) * len(out)

    @jax.custom_jvp
    def odd_sized_jvp(*xs: Array) -> tuple[Array, ...]:
        return odd_sized(*xs)

    @odd_sized_jvp.defjvp
    def _(primals: tuple, tangents: tuple) -> tuple:
        return odd_sized_jvp(*primals), jax.jvp(fn, primals, tangents)[1]

    return odd_sized_jvp(*xs)


def _odd_sized_trees(
    fn: Callable, treedef: jtu.PyTreeDef, *trees: PyTree
) -> tuple[PyTree, ...]:
    # `_odd_sized` on each array of `trees` (each with structure `treedef`)
    arrays = [treedef.flatten_up_to(tree) for tree in trees]
    outs = [_odd_sized(fn, *xs) for xs in zip(*arrays, strict=True)]
    return tuple(treedef.unflatten(list(x)) for x in zip(*outs, strict=True))


class _UnrolledERK(dx.AbstractERK):
    # the degree of the solver's interpolation, a polynomial of time (`_nodes`)
    _interpolation_degree: ClassVar[int]

    def _supported(
        self,
        terms: dx.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> bool:
        # whether the step is a single explicit Butcher tableau with a cheap vector
        # field, on GPU or CPU, which the steps below support
        vf_expensive, _ = self._common(terms, t0, t1, y0, args)
        tableau = self.tableau
        return (
            jax.default_backend() in ('gpu', 'cpu')
            and not vf_expensive
            and isinstance(tableau, dx.ButcherTableau)
            and not tableau.implicit
        )

    @staticmethod
    def _unrolled() -> bool:
        # On CPU, XLA fuses each stage's output into the later stages that read it and
        # recomputes it there, which makes unrolled steps slower; there is also no
        # synchronisation to save. Unroll on GPU only (`_unrolled_stages`); on CPU,
        # loop over the stages (`_looped_stages`).
        return jax.default_backend() == 'gpu'

    def init(
        self,
        terms: dx.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> PyTree:
        if not self._supported(terms, t0, t1, y0, args):
            return super().init(terms, t0, t1, y0, args)
        _, fsal = self._common(terms, t0, t1, y0, args)
        if not fsal:
            return None
        # first_step=False: the stored f0 is the vector field at (t0, y0)
        return jnp.array(False), terms.vf(t0, y0, args)

    def step(
        self,
        terms: dx.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
        solver_state: PyTree,
        made_jump: BoolScalarLike,
    ) -> tuple:
        if not self._supported(terms, t0, t1, y0, args):
            return super().step(terms, t0, t1, y0, args, solver_state, made_jump)
        _, fsal = self._common(terms, t0, t1, y0, args)
        # a single explicit Butcher tableau (checked by `_supported`)
        tableau = cast(dx.ButcherTableau, self.tableau)
        control = terms.contr(t0, t1)
        dt = t1 - t0

        def f(t: RealScalarLike, y: Y) -> PyTree:
            return terms.vf(t, y, args)

        # === first stage
        t_first = t0 if tableau.c1 is None or tableau.c1 == 0 else t0 + tableau.c1 * dt
        if fsal:
            first_step, f_previous = solver_state
            if made_jump is False:
                f_first = f_previous
            else:
                # after a jump, the vector field changed at t0: re-evaluate it (for all
                # batch elements at once, as Diffrax does)
                evaluate = eqxi.unvmap_any(first_step | made_jump)
                f_first = jax.lax.cond(
                    evaluate, lambda: f(t_first, y0), lambda: f_previous
                )
        else:
            f_first = f(t_first, y0)
        k_first = terms.prod(f_first, control)

        # === other stages, solution, error estimate and dense output
        stages = self._unrolled_stages if self._unrolled() else self._looped_stages
        return stages(f, t0, t1, y0, k_first, f_first, control, terms, fsal)

    def _unrolled_stages(
        self,
        f: Callable[[RealScalarLike, Y], PyTree],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        k_first: PyTree,
        f_first: PyTree,
        control: PyTree,
        terms: dx.AbstractTerm,
        fsal: bool,
    ) -> tuple:
        # The stages after the first, as straight-line code (see the module docstring).
        tableau = cast(dx.ButcherTableau, self.tableau)
        dt = t1 - t0
        # The stages are kept apart for the dense output (see `_StagesInterpolation`).
        # Kept apart, XLA may lay some stages out transposed (e.g. a vector field
        # ending in x + x^dag), and every later read then transposes them (on an A100,
        # 7% slower solves): each stage is kept row-major.
        fs = [f_first]
        ks = [_row_major(k_first)]

        # === other stages
        # Large states: each stage input is written out with an odd number of
        # elements, by the kernel that evaluates the stage before it (see
        # `_odd_sized`). The size is that of one batch element: under vmap, XLA sees
        # the whole batch, so a batch of smaller states keeps the unrolled kernels.
        leaves, treedef = jtu.tree_flatten(y0)
        odd = all(x.size >= _MIN_SIZE_ODD_STAGES for x in leaves)

        def odd_sized_stages(fn: Callable, *trees: PyTree) -> tuple[PyTree, ...]:
            return _odd_sized_trees(fn, treedef, *trees) if odd else fn(*trees)

        def next_stage(k: PyTree, a: np.ndarray) -> tuple[PyTree, PyTree]:
            # the stage k, row-major, and the next stage input
            if not odd:
                k = _row_major(k)
                return k, _combine(y0, a, [*ks, k])
            # both written by one kernel
            k, yi = _odd_sized_trees(
                lambda y0, k, *ks: (k, _combine(y0, a, [*ks, k])), treedef, y0, k, *ks
            )
            return _row_major(k), yi

        # the tableaus unrolled here (Tsit5, Dopri5, Dopri8) have several stages
        stages = list(zip(tableau.c, tableau.a_lower, strict=True))
        a = stages[0][1]
        (yi,) = odd_sized_stages(lambda y0, k, a=a: (_combine(y0, a, [k]),), y0, ks[0])
        for i, (c, _) in enumerate(stages):
            ti = _stage_time(t0, t1, dt, c)
            fi = f(ti, yi)
            if i + 1 < len(stages):
                ki, yi = next_stage(terms.prod(fi, control), stages[i + 1][1])
            else:
                # the last stage, also the first stage of the next step if FSAL
                (fi,) = odd_sized_stages(lambda f: (f,), fi)
                ki = _row_major(terms.prod(fi, control))
            fs.append(fi)
            ks.append(ki)

        # === solution, error estimate and dense output
        y1 = yi if tableau.ssal else _combine(y0, tableau.b_sol, ks)
        y_error = _dot(tableau.b_error, ks)
        # The dense output keeps the stages apart (see `_StagesInterpolation`). At
        # every step, Diffrax passes it to its loop over the save times, and XLA
        # copies each odd-sized stage out of its padded buffer for that loop. So for
        # large states, the dense output is instead the values of the interpolation
        # at a few fixed times of the step (see `_NodesInterpolation`).
        if odd:
            degree = self._interpolation_degree
            dense = _nodes(self.interpolation_cls, degree, y0, y1, ks)
        else:
            dense = _Stages(tuple(ks))
        dense_info = dict(y0=y0, y1=y1, k=dense)
        new_solver_state = (jnp.array(False), fs[-1]) if fsal else None
        return y1, y_error, dense_info, new_solver_state, dx.RESULTS.successful

    def _looped_stages(
        self,
        f: Callable[[RealScalarLike, Y], PyTree],
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        k_first: PyTree,
        f_first: PyTree,
        control: PyTree,
        terms: dx.AbstractTerm,
        fsal: bool,
    ) -> tuple:
        # The stages after the first, in a loop as Diffrax does (unrolled, XLA would
        # recompute stages on CPU, see `_unrolled`), but each stage combines only the
        # stages it reads, as the unrolled step: Diffrax's `tensordot` over all the
        # stages runs on CPU as a complex matrix-vector product on one thread, several
        # times slower than these elementwise sums.
        tableau = cast(dx.ButcherTableau, self.tableau)
        num_stages = len(tableau.c) + 1
        dt = t1 - t0

        def unstack(ks: PyTree, count: int = num_stages) -> list[PyTree]:
            # the first `count` stages
            return [jtu.tree_map(lambda k, j=j: k[j], ks) for j in range(count)]

        def combination(a_lower: np.ndarray) -> Callable[[PyTree], Y]:
            return lambda ks: _combine(y0, a_lower, unstack(ks, len(a_lower)))

        combinations = [combination(a_lower) for a_lower in tableau.a_lower]
        times = jnp.stack([jnp.asarray(_stage_time(t0, t1, dt, c)) for c in tableau.c])

        def stage(i: Array, carry: tuple) -> tuple:
            ks, _, _ = carry
            yi = jax.lax.switch(i - 1, combinations, ks)
            ti = times[i - 1]
            fi = f(ti, yi)
            ki = terms.prod(fi, control)
            ks = jtu.tree_map(lambda buffer, k: buffer.at[i].set(k), ks, ki)
            return ks, yi, fi

        ks = jtu.tree_map(
            lambda k: jnp.zeros((num_stages, *k.shape), k.dtype).at[0].set(k), k_first
        )
        ks, y_last, f_last = jax.lax.fori_loop(1, num_stages, stage, (ks, y0, f_first))
        y1 = y_last if tableau.ssal else _combine(y0, tableau.b_sol, unstack(ks))
        y_error = _dot(tableau.b_error, unstack(ks))
        dense_info = dict(y0=y0, y1=y1, k=ks)
        new_solver_state = (jnp.array(False), f_last) if fsal else None
        return y1, y_error, dense_info, new_solver_state, dx.RESULTS.successful


class _Stages(eqx.Module):
    # the stages of an unrolled step, kept apart for `_StagesInterpolation`
    ks: tuple[PyTree, ...]


def _interpolation_weights(
    base: Callable[..., dx.AbstractLocalInterpolation],
    t0: RealScalarLike,
    t1: RealScalarLike,
    t: RealScalarLike,
    nk: int,
    dtype: np.dtype,
) -> list[Array]:
    # the weights of (y0, y1, *k) in `base` at t: the transpose of `base` on scalar
    # inputs, which XLA computes without the matrix products of `base` on unit vectors
    # (a vjp: JAX 0.4 cannot `linear_transpose` the polynomial of Dopri5's
    # interpolation)
    scalar = jnp.zeros((), dtype)

    def interpolate(y0: Array, y1: Array, k: Array) -> Array:
        return base(t0=t0, t1=t1, y0=y0, y1=y1, k=k).evaluate(t)

    _, transpose = jax.vjp(interpolate, scalar, scalar, jnp.zeros(nk, dtype))
    w_y0, w_y1, w_k = transpose(jnp.ones((), dtype))
    return [w_y0, w_y1, *w_k]


class _StagesInterpolation(dx.AbstractLocalInterpolation):
    # The dense output of an unrolled step, from its stages kept apart. Diffrax's
    # interpolations take the stages stacked in one array, which a step would write
    # at every step (on an A100, 12-14% of the time of a step of a mesolve of
    # dimension 2,304), while only the steps that contain a save time read it, often
    # only in part (the diagonals for the expectation value of a sparse operator).
    # These interpolations are linear in (y0, y1, k): the output is the combination of
    # the inputs with the weights of `base` at t, without the inputs whose weight is
    # always zero (`zero`, indices in (y0, y1, *k)).
    t0: RealScalarLike  # ty: ignore[invalid-attribute-override]
    t1: RealScalarLike  # ty: ignore[invalid-attribute-override]
    y0: Y
    y1: Y
    k: _Stages
    base: Callable[..., dx.AbstractLocalInterpolation] = eqx.field(static=True)
    zero: tuple[int, ...] = eqx.field(static=True)

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Y:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        nk = len(self.k.ks)
        dtype = jnp.result_type(t0, self.t0, float)
        weights = _interpolation_weights(self.base, self.t0, self.t1, t0, nk, dtype)
        used = [i for i in range(nk + 2) if i not in self.zero]

        def leaf(*x: ArrayLike) -> Array:
            total = weights[used[0]] * x[used[0]]
            for i in used[1:]:
                total = total + weights[i] * x[i]
            return total.astype(jnp.result_type(x[0]))

        return jtu.tree_map(leaf, self.y0, self.y1, *self.k.ks)


class _Nodes(eqx.Module):
    # the dense output of an unrolled step of a large state, for `_NodesInterpolation`
    k0: PyTree
    values: tuple[PyTree, ...]
    nodes: tuple[float, ...] = eqx.field(static=True)


def _linear_rescale(t0: RealScalarLike, t: RealScalarLike, t1: RealScalarLike) -> Array:
    # (t - t0) / (t1 - t0), or 0 if t0 == t1: Diffrax's private `linear_rescale`
    cond = t0 == t1
    numerator = jnp.where(cond, 0, t - t0)
    denominator = jnp.where(cond, 1, t1 - t0)
    return numerator / denominator


class _NodesInterpolation(dx.AbstractLocalInterpolation):
    # The dense output of an unrolled step of a large state. Let theta be the time in
    # the step, theta = (t - t0) / (t1 - t0). Diffrax's interpolations are
    # polynomials of theta, linear in (y0, y1, k), with value y0 and derivative k[0]
    # at theta = 0. So each one is y0 + theta k[0] + theta^2 g(theta), where g is a
    # polynomial. The step stores the values of g at a few fixed times of the step,
    # the nodes (`_nodes`). Here, g is evaluated from these values by Lagrange
    # interpolation. This gives the same polynomial as Diffrax, up to rounding.
    t0: RealScalarLike  # ty: ignore[invalid-attribute-override]
    t1: RealScalarLike  # ty: ignore[invalid-attribute-override]
    y0: Y
    k: _Nodes

    def evaluate(
        self, t0: RealScalarLike, t1: RealScalarLike | None = None, left: bool = True
    ) -> Y:
        del left
        if t1 is not None:
            return self.evaluate(t1) - self.evaluate(t0)
        theta = _linear_rescale(self.t0, t0, self.t1)
        nodes = self.k.nodes
        lagrange = []
        for node in nodes:
            weight = theta**2
            for other in nodes:
                if other != node:
                    weight = weight * (theta - other) / (node - other)
            lagrange.append(weight)

        def leaf(y0: Array, k0: Array, *values: Array) -> Array:
            total = y0 + theta * k0
            for weight, value in zip(lagrange, values, strict=True):
                total = total + weight * value
            return total.astype(y0.dtype)

        return jtu.tree_map(leaf, self.y0, self.k.k0, *self.k.values)


@cache
def _node_weights(
    base: Callable[..., dx.AbstractLocalInterpolation],
    nk: int,
    degree: int,
    dtype: np.dtype,
) -> tuple[tuple[float, ...], np.ndarray]:
    # The nodes theta_i of `_NodesInterpolation`, and w[i, j], the weight of the j-th
    # of (y0, y1, *k) in g(theta_i). `base` is Diffrax's interpolation, a polynomial
    # of theta of degree `degree`. So g has degree `degree` - 2, and `degree` - 1
    # nodes determine it. The nodes are the `degree` Chebyshev-Lobatto points of
    # [0, 1], without 0, where the formula of g below divides by zero.
    nodes = 0.5 - 0.5 * np.cos(np.pi * np.arange(1, degree) / (degree - 1))

    def weights(theta: Array) -> Array:
        return jnp.stack(_interpolation_weights(base, 0.0, 1.0, theta, nk, dtype))

    # Jitted: op by op, this took about 10x longer on the first call of a solve.
    with jax.ensure_compile_time_eval():
        values = jax.jit(jax.vmap(weights))(jnp.asarray(nodes, dtype))
    w = np.array(values, np.float64)
    # g(theta) = (y(theta) - y0 - theta k[0]) / theta^2
    w[:, 0] -= 1
    w[:, 2] -= nodes
    w /= nodes[:, None] ** 2
    # Some weights are zero in exact arithmetic (e.g. those of Dopri5's stages at
    # theta = 1), but come out as rounding residues. They are set to zero, so that
    # `_dot` skips their inputs.
    w[np.abs(w) < 1e3 * np.finfo(dtype).eps * np.abs(w).max(axis=1, keepdims=True)] = 0
    return tuple(nodes.tolist()), w


def _nodes(
    base: Callable[..., dx.AbstractLocalInterpolation],
    degree: int,
    y0: Y,
    y1: Y,
    ks: list[PyTree],
) -> _Nodes:
    # the dense output of a step of a large state: k[0] and the values of g at the
    # nodes, for the interpolation `base` of degree `degree` (`_NodesInterpolation`)
    dtype = np.finfo(jtu.tree_leaves(y0)[0].dtype).dtype
    nodes, w = _node_weights(base, len(ks), degree, dtype)
    values = tuple(_dot(row, [y0, y1, *ks]) for row in w)
    return _Nodes(ks[0], values, nodes)


def _interpolation(
    base: Callable[..., dx.AbstractLocalInterpolation], zero: tuple[int, ...]
) -> Callable[..., dx.AbstractLocalInterpolation]:
    # `base` for stacked stages (Diffrax's steps), `_StagesInterpolation` or
    # `_NodesInterpolation` for an unrolled step
    def interpolation(
        *, t0: RealScalarLike, t1: RealScalarLike, y0: Y, y1: Y, k: PyTree
    ) -> dx.AbstractLocalInterpolation:
        if type(k) is _Nodes:
            return _NodesInterpolation(t0, t1, y0, k)
        if type(k) is _Stages:
            return _StagesInterpolation(t0, t1, y0, y1, k, base, zero)
        return base(t0=t0, t1=t1, y0=y0, y1=y1, k=k)

    return interpolation


class UnrolledTsit5(_UnrolledERK, dx.Tsit5):
    """Diffrax's `Tsit5`, with its stages unrolled on GPU and looped over on CPU."""

    _interpolation_degree: ClassVar[int] = 4
    # Diffrax types this as its own interpolation class; y1 has a zero weight
    interpolation_cls = staticmethod(  # ty: ignore[invalid-assignment]
        _interpolation(dx.Tsit5.interpolation_cls, zero=(1,))
    )


class UnrolledDopri5(_UnrolledERK, dx.Dopri5):
    """Diffrax's `Dopri5`, with its stages unrolled on GPU and looped over on CPU."""

    _interpolation_degree: ClassVar[int] = 4
    # Diffrax types this as its own interpolation class; k[1] has a zero weight
    interpolation_cls = staticmethod(  # ty: ignore[invalid-assignment]
        _interpolation(dx.Dopri5.interpolation_cls, zero=(3,))
    )


class UnrolledDopri8(_UnrolledERK, dx.Dopri8):
    """Diffrax's `Dopri8`, with its stages unrolled on GPU and looped over on CPU."""

    _interpolation_degree: ClassVar[int] = 6
    # Diffrax types this as its own interpolation class; y1 and k[1:5] have zero
    # weights
    interpolation_cls = staticmethod(  # ty: ignore[invalid-assignment]
        _interpolation(dx.Dopri8.interpolation_cls, zero=(1, 3, 4, 5, 6))
    )
