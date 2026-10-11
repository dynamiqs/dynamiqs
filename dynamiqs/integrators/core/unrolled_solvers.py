"""Explicit Runge-Kutta solvers of Diffrax with their stages unrolled at trace time.

Diffrax evaluates the stages of a Runge-Kutta step in a `while_loop`, which keeps
compilation fast for any tableau but, on GPU, costs one device-to-host synchronisation
per stage: XLA copies the loop condition back to the host before each iteration. For
the small and medium systems dynamiqs solves, these synchronisations dominate the cost
of a step. The solvers below take the same steps with the same tableaus, but write the
stages out as straight-line code, which also lets XLA fuse the stage combinations into
the vector field evaluations.

The first stage of an FSAL method is the last stage of the previous step. Diffrax
evaluates it inside the stage loop on the first step and after a jump; here `init`
evaluates it once, and a step re-evaluates it only after a jump, under a `lax.cond`
that only exists when the solve has jump times.

Other terms (e.g. expensive vector fields or implicit tableaus) fall back to Diffrax's
implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

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


class _UnrolledERK(dx.AbstractERK):
    def _unrolled(
        self,
        terms: dx.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> bool:
        vf_expensive, _ = self._common(terms, t0, t1, y0, args)
        tableau = self.tableau
        # On CPU, XLA fuses each stage's output into the later stages that read it and
        # recomputes it there, which makes unrolled steps slower; there is also no
        # synchronisation to save. Unroll on GPU only.
        return (
            jax.default_backend() == 'gpu'
            and not vf_expensive
            and isinstance(tableau, dx.ButcherTableau)
            and not tableau.implicit
        )

    def init(
        self,
        terms: dx.AbstractTerm,
        t0: RealScalarLike,
        t1: RealScalarLike,
        y0: Y,
        args: Args,
    ) -> PyTree:
        if not self._unrolled(terms, t0, t1, y0, args):
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
        if not self._unrolled(terms, t0, t1, y0, args):
            return super().step(terms, t0, t1, y0, args, solver_state, made_jump)
        _, fsal = self._common(terms, t0, t1, y0, args)
        # a single explicit Butcher tableau (checked by `_unrolled`)
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
        # The stages are kept apart for the dense output (see `_StagesInterpolation`).
        # Kept apart, XLA may lay some stages out transposed (e.g. a vector field
        # ending in x + x^dag), and every later read then transposes them (on an A100,
        # 7% slower solves): each stage is kept row-major.
        fs = [f_first]
        ks = [_row_major(terms.prod(f_first, control))]

        # === other stages
        yi = y0
        for c, a_lower in zip(tableau.c, tableau.a_lower, strict=True):
            ti = t1 if c == 1 else t0 + float(c) * dt
            yi = _combine(y0, a_lower, ks)
            fi = f(ti, yi)
            fs.append(fi)
            ks.append(_row_major(terms.prod(fi, control)))

        # === solution, error estimate and dense output
        y1 = yi if tableau.ssal else _combine(y0, tableau.b_sol, ks)
        y_error = _dot(tableau.b_error, ks)
        # the stages are kept apart (see `_StagesInterpolation`)
        dense_info = dict(y0=y0, y1=y1, k=_Stages(tuple(ks)))
        new_solver_state = (jnp.array(False), fs[-1]) if fsal else None
        return y1, y_error, dense_info, new_solver_state, dx.RESULTS.successful


class _Stages(eqx.Module):
    # the stages of an unrolled step, kept apart for `_StagesInterpolation`
    ks: tuple[PyTree, ...]


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
        # the weights of (y0, y1, *k): the transpose of `base` on scalar inputs, which
        # XLA computes without the matrix products of `base` on unit vectors (a vjp:
        # JAX 0.4 cannot `linear_transpose` the polynomial of Dopri5's interpolation)
        nk = len(self.k.ks)
        dtype = jnp.result_type(t0, self.t0, float)
        scalar = jnp.zeros((), dtype)

        def interpolate(y0: Array, y1: Array, k: Array) -> Array:
            return self.base(t0=self.t0, t1=self.t1, y0=y0, y1=y1, k=k).evaluate(t0)

        _, transpose = jax.vjp(interpolate, scalar, scalar, jnp.zeros(nk, dtype))
        w_y0, w_y1, w_k = transpose(jnp.ones((), dtype))
        weights = [w_y0, w_y1, *w_k]
        used = [i for i in range(nk + 2) if i not in self.zero]

        def leaf(*x: ArrayLike) -> Array:
            total = weights[used[0]] * x[used[0]]
            for i in used[1:]:
                total = total + weights[i] * x[i]
            return total.astype(jnp.result_type(x[0]))

        return jtu.tree_map(leaf, self.y0, self.y1, *self.k.ks)


def _interpolation(
    base: Callable[..., dx.AbstractLocalInterpolation], zero: tuple[int, ...]
) -> Callable[..., dx.AbstractLocalInterpolation]:
    # `base` for stacked stages (Diffrax's steps), `_StagesInterpolation` for the
    # stages of an unrolled step
    def interpolation(
        *, t0: RealScalarLike, t1: RealScalarLike, y0: Y, y1: Y, k: PyTree
    ) -> dx.AbstractLocalInterpolation:
        if type(k) is _Stages:
            return _StagesInterpolation(t0, t1, y0, y1, k, base, zero)
        return base(t0=t0, t1=t1, y0=y0, y1=y1, k=k)

    return interpolation


class UnrolledTsit5(_UnrolledERK, dx.Tsit5):
    """Diffrax's `Tsit5`, with its stages unrolled."""

    # Diffrax types this as its own interpolation class; y1 has a zero weight
    interpolation_cls = staticmethod(  # ty: ignore[invalid-assignment]
        _interpolation(dx.Tsit5.interpolation_cls, zero=(1,))
    )


class UnrolledDopri5(_UnrolledERK, dx.Dopri5):
    """Diffrax's `Dopri5`, with its stages unrolled."""

    # Diffrax types this as its own interpolation class; k[1] has a zero weight
    interpolation_cls = staticmethod(  # ty: ignore[invalid-assignment]
        _interpolation(dx.Dopri5.interpolation_cls, zero=(3,))
    )


class UnrolledDopri8(_UnrolledERK, dx.Dopri8):
    """Diffrax's `Dopri8`, with its stages unrolled."""

    # Diffrax types this as its own interpolation class; y1 and k[1:5] have zero
    # weights
    interpolation_cls = staticmethod(  # ty: ignore[invalid-assignment]
        _interpolation(dx.Dopri8.interpolation_cls, zero=(1, 3, 4, 5, 6))
    )
