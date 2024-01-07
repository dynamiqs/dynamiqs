from __future__ import annotations

import logging
from abc import ABC
from typing import Any

import jax
import jax.numpy as jnp

from dynamiqs import plot_wigner_mosaic
from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver

from .mesolve.open_system import OpenSystem
from .sesolve.closed_system import ClosedSystem
from .system import System


class SolverTester(ABC):
    def test_batching(self):
        pass

    def _test_batching(
        self,
        system: ClosedSystem,
        solver: Solver,
        *,
        options: dict[str, Any] | None = None,
    ):
        n = system.n
        m = 1 if isinstance(system, ClosedSystem) else n
        bH = len(system.Hb)
        by = len(system.y0b)
        nE = len(system.E)
        nt = 11
        tsave = system.tsave(nt)

        def run(Hb, yb):
            H = system.H if not Hb else system.Hb
            y0 = system.y0 if not yb else system.y0b
            return system.run(tsave, solver, options=options, H=H, y0=y0)

        # === test regular batching (cartesian product)

        # no batching
        result = run(Hb=False, yb=False)
        assert result.ysave.shape == (nt, n, m)
        assert result.Esave.shape == (nE, nt)

        # batched H
        result = run(Hb=True, yb=False)
        assert result.ysave.shape == (bH, nt, n, m)
        assert result.Esave.shape == (bH, nE, nt)

        # batched y0
        result = run(Hb=False, yb=True)
        assert result.ysave.shape == (by, nt, n, m)
        assert result.Esave.shape == (by, nE, nt)

        # batched H and y0
        result = run(Hb=True, yb=True)
        assert result.ysave.shape == (bH, by, nt, n, m)
        assert result.ysave.shape == (bH, by, nt, n, m)

        if isinstance(system, OpenSystem):
            bL = len(system.Lb[0])

            def run(Hb, Lb, yb):
                H = system.H if not Hb else system.Hb
                L = system.L if not Lb else system.Lb
                y0 = system.y0 if not yb else system.y0b
                return system.run(tsave, solver, options=options, H=H, L=L, y0=y0)

            # batched L
            result = run(Hb=False, Lb=True, yb=False)
            assert result.ysave.shape == (bL, nt, n, n)
            assert result.Esave.shape == (bL, nE, nt)

            # batched H and L
            result = run(Hb=True, Lb=True, yb=False)
            assert result.ysave.shape == (bH, bL, nt, n, n)
            assert result.Esave.shape == (bH, bL, nE, nt)

            # batched L and y0
            result = run(Hb=False, Lb=True, yb=True)
            assert result.ysave.shape == (bL, by, nt, n, n)
            assert result.Esave.shape == (bL, by, nE, nt)

            # batched H and L and y0
            result = run(Hb=True, Lb=True, yb=True)
            assert result.ysave.shape == (bH, bL, by, nt, n, n)
            assert result.Esave.shape == (bH, bL, by, nE, nt)

        # === test non cartesian batching
        options = {} if options is None else options
        options['cartesian_batching'] = False
        b = 2

        Hb = system.Hb[:b]
        y0b = system.y0b[:b]

        if isinstance(system, ClosedSystem):
            result = system.run(tsave, solver, options=options, H=Hb, y0=y0b)
            assert result.ysave.shape == (b, nt, n, 1)
            assert result.Esave.shape == (b, nE, nt)
        elif isinstance(system, OpenSystem):
            Lb = [L[:b] for L in system.Lb]
            result = system.run(tsave, solver, options=options, H=Hb, L=Lb, y0=y0b)
            assert result.ysave.shape == (b, nt, n, n)
            assert result.Esave.shape == (b, nE, nt)

    def _test_correctness(
        self,
        system: System,
        solver: Solver,
        *,
        options: dict[str, Any] | None = None,
        ntsave: int = 11,
        ysave_atol: float = 1e-3,
        esave_rtol: float = 1e-3,
        esave_atol: float = 1e-5,
    ):
        tsave = system.tsave(ntsave)
        result = system.run(tsave, solver, options=options)

        # === test ysave
        errs = jnp.linalg.norm(result.ysave - system.states(tsave), axis=(-2, -1))
        logging.warning(f'errs = {errs}')
        assert jnp.all(errs <= ysave_atol)

        # === test Esave
        true_Esave = system.expects(tsave)
        logging.warning(f'Esave      = {result.Esave}')
        logging.warning(f'true_Esave = {true_Esave}')
        assert jnp.allclose(result.Esave, true_Esave, rtol=esave_rtol, atol=esave_atol)

    def test_correctness(self):
        pass

    def _test_gradient(
        self,
        system: System,
        solver: Solver,
        gradient: Gradient,
        *,
        options: dict[str, Any] | None = None,
        ntsave: int = 11,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        tsave = system.tsave(ntsave)
        # result = system.run(tsave, solver, gradient=gradient, options=options)

        # === test gradients depending on final ysave

        def loss_states(params):
            result = system.run(
                tsave,
                solver,
                options=options,
                y0=system.y0,
                params=params,
            )
            l = system.loss_state(result.ysave[-1])
            return l

        # loss_state = system.loss_state(result.ysave[-1])
        # grads_state = torch.autograd.grad(
        #     loss_state, system.params, retain_graph=True, allow_unused=True
        # )
        grads_state = jax.grad(loss_states, 0)(system.params)
        grads_state = jnp.stack(grads_state)
        true_grads_state = system.grads_state(tsave[-1])

        logging.warning(f'grads_state       = {grads_state}')
        logging.warning(f'true_grads_state  = {true_grads_state}')

        # === test gradients depending on final Esave
        # todo: this part does not pass (yet)
        # loss_expect = system.loss_expect(result.Esave[:, -1])
        def loss_expect(params):
            result = system.run(
                tsave,
                solver,
                options=options,
                y0=system.y0,
                params=params,
            )
            l = system.loss_expect(result.Esave[:, -1])
            return l

        # grads_expect = [
        #     jnp.stack(torch.autograd.grad(loss, system.params, retain_graph=True))
        #     for loss in loss_expect
        # ]
        grads_expect = jax.jacrev(loss_expect, 0)(system.params)
        grads_expect = jnp.stack(grads_expect)
        true_grads_expect = system.grads_expect(tsave[-1])

        print()
        print(f'grads_expect      =')
        print(grads_expect)
        print(f'true_grads_expect = ')
        print(true_grads_expect)
        logging.warning(f'grads_expect      = {grads_expect}')
        logging.warning(f'true_grads_expect = {true_grads_expect}')

        assert jnp.allclose(grads_state, true_grads_state, rtol=rtol, atol=atol)
        assert jnp.allclose(grads_expect, true_grads_expect, rtol=rtol, atol=atol)
