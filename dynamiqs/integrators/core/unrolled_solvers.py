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

import diffrax as dx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import ArrayLike, PyTree


def _unroll_stages() -> bool:
    # On CPU, XLA fuses each stage's output into the later stages that read it and
    # recomputes it there, which makes unrolled steps slower; there is also no
    # synchronisation to save. Unroll on GPU only.
    return jax.default_backend() == 'gpu'


def _combine(y0: PyTree, coefficients: ArrayLike, ks: list[PyTree]) -> PyTree:
    # y0 + sum_j coefficients[j] * ks[j], skipping the zero coefficients
    def leaf(y: ArrayLike, *k: ArrayLike) -> ArrayLike:
        total = y
        for a, kj in zip(coefficients, k, strict=True):
            if a != 0:
                total = total + float(a) * kj
        return total

    return jtu.tree_map(leaf, y0, *ks)


def _dot(coefficients: ArrayLike, ks: list[PyTree]) -> PyTree:
    # sum_j coefficients[j] * ks[j], skipping the zero coefficients
    def leaf(*k: ArrayLike) -> ArrayLike:
        terms = [float(a) * kj for a, kj in zip(coefficients, k, strict=True) if a != 0]
        total = terms[0]
        for term in terms[1:]:
            total = total + term
        return total

    return jtu.tree_map(leaf, *ks)


class _UnrolledERK(dx.AbstractERK):
    def _unrolled(
        self,
        terms: dx.AbstractTerm,
        t0: ArrayLike,
        t1: ArrayLike,
        y0: PyTree,
        args: PyTree,
    ) -> bool:
        vf_expensive, _ = self._common(terms, t0, t1, y0, args)
        tableau = self.tableau
        return (
            _unroll_stages()
            and not vf_expensive
            and isinstance(tableau, dx.ButcherTableau)
            and not tableau.implicit
        )

    def init(
        self,
        terms: dx.AbstractTerm,
        t0: ArrayLike,
        t1: ArrayLike,
        y0: PyTree,
        args: PyTree,
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
        t0: ArrayLike,
        t1: ArrayLike,
        y0: PyTree,
        args: PyTree,
        solver_state: PyTree,
        made_jump: ArrayLike,
    ) -> tuple:
        if not self._unrolled(terms, t0, t1, y0, args):
            return super().step(terms, t0, t1, y0, args, solver_state, made_jump)
        _, fsal = self._common(terms, t0, t1, y0, args)
        tableau = self.tableau
        control = terms.contr(t0, t1)
        dt = t1 - t0

        def f(t: ArrayLike, y: PyTree) -> PyTree:
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
        fs = [f_first]
        ks = [terms.prod(f_first, control)]

        # === other stages
        yi = y0
        for c, a_lower in zip(tableau.c, tableau.a_lower, strict=True):
            ti = t1 if c == 1 else t0 + float(c) * dt
            yi = _combine(y0, a_lower, ks)
            fi = f(ti, yi)
            fs.append(fi)
            ks.append(terms.prod(fi, control))

        # === solution, error estimate and dense output
        y1 = yi if tableau.ssal else _combine(y0, tableau.b_sol, ks)
        y_error = _dot(tableau.b_error, ks)
        k = jtu.tree_map(lambda *x: jnp.stack(x), *ks)
        dense_info = dict(y0=y0, y1=y1, k=k)
        new_solver_state = (jnp.array(False), fs[-1]) if fsal else None
        return y1, y_error, dense_info, new_solver_state, dx.RESULTS.successful


class UnrolledTsit5(_UnrolledERK, dx.Tsit5):
    """Diffrax's `Tsit5`, with its stages unrolled."""


class UnrolledDopri5(_UnrolledERK, dx.Dopri5):
    """Diffrax's `Dopri5`, with its stages unrolled."""


class UnrolledDopri8(_UnrolledERK, dx.Dopri8):
    """Diffrax's `Dopri8`, with its stages unrolled."""
