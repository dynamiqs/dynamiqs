from __future__ import annotations

import logging
from abc import ABC
from typing import Any

import torch

from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver

from .mesolve.open_system import OpenSystem
from .sesolve.closed_system import ClosedSystem
from .system import System


class SolverTester(ABC):
    def test_batching(self):
        pass

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
        errs = torch.linalg.norm(result.ysave - system.states(tsave), dim=(-2, -1))
        logging.warning(f'errs = {errs}')
        assert torch.all(errs <= ysave_atol)

        # === test exp_save
        true_exp_save = system.expects(tsave)
        logging.warning(f'exp_save      = {result.exp_save}')
        logging.warning(f'true_exp_save = {true_exp_save}')
        assert torch.allclose(
            result.exp_save, true_exp_save, rtol=esave_rtol, atol=esave_atol
        )

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
        result = system.run(tsave, solver, gradient=gradient, options=options)

        # === test gradients depending on final ysave
        loss_state = system.loss_state(result.ysave[-1])
        grads_state = torch.autograd.grad(
            loss_state, system.params, retain_graph=True, allow_unused=True
        )
        grads_state = torch.stack(grads_state)
        true_grads_state = system.grads_state(tsave[-1])

        logging.warning(f'grads_state       = {grads_state}')
        logging.warning(f'true_grads_state  = {true_grads_state}')

        # === test gradients depending on final exp_save
        loss_expect = system.loss_expect(result.exp_save[:, -1])
        grads_expect = [
            torch.stack(torch.autograd.grad(loss, system.params, retain_graph=True))
            for loss in loss_expect
        ]
        grads_expect = torch.stack(grads_expect)
        true_grads_expect = system.grads_expect(tsave[-1])

        logging.warning(f'grads_expect      = {grads_expect}')
        logging.warning(f'true_grads_expect = {true_grads_expect}')

        assert torch.allclose(grads_state, true_grads_state, rtol=rtol, atol=atol)
        assert torch.allclose(grads_expect, true_grads_expect, rtol=rtol, atol=atol)


class ClosedSolverTester(SolverTester):
    def _test_batching(
        self,
        system: ClosedSystem,
        solver: Solver,
        *,
        options: dict[str, Any] | None = None,
    ):
        """Test the batching of `H` and `y0`, and the returned object sizes."""
        n = system.n
        n_exp_ops = len(system.exp_ops)
        bH = len(system.H_batched)
        by = len(system.y0_batched)
        ntsave = 11
        tsave = system.tsave(ntsave)

        run = lambda H, y0: system._run(H, y0, tsave, solver, options=options)

        # no batching
        result = run(system.H, system.y0)
        assert result.ysave.shape == (ntsave, n, 1)
        assert result.exp_save.shape == (n_exp_ops, ntsave)

        # batched H
        result = run(system.H_batched, system.y0)
        assert result.ysave.shape == (bH, ntsave, n, 1)
        assert result.exp_save.shape == (bH, n_exp_ops, ntsave)

        # batched y0
        result = run(system.H, system.y0_batched)
        assert result.ysave.shape == (by, ntsave, n, 1)
        assert result.exp_save.shape == (by, n_exp_ops, ntsave)

        # batched H and y0
        result = run(system.H_batched, system.y0_batched)
        assert result.ysave.shape == (bH, by, ntsave, n, 1)
        assert result.exp_save.shape == (bH, by, n_exp_ops, ntsave)


class OpenSolverTester(SolverTester):
    def _test_batching(
        self,
        system: OpenSystem,
        solver: Solver,
        *,
        options: dict[str, Any] | None = None,
    ):
        """Test the batching of `H` and `y0`, and the returned object sizes."""
        n = system.n
        n_exp_ops = len(system.exp_ops)
        bH = len(system.H_batched)
        bL = system.jump_ops_batched[0].shape[0]
        by = len(system.y0_batched)
        ntsave = 11
        tsave = system.tsave(ntsave)

        run = lambda H, jump_ops, y0: system._run(
            H, jump_ops, y0, tsave, solver, options=options
        )

        # no batching
        result = run(system.H, system.jump_ops, system.y0)
        assert result.ysave.shape == (ntsave, n, n)
        assert result.exp_save.shape == (n_exp_ops, ntsave)

        # batched H
        result = run(system.H_batched, system.jump_ops, system.y0)
        assert result.ysave.shape == (bH, ntsave, n, n)
        assert result.exp_save.shape == (bH, n_exp_ops, ntsave)

        # batched jump_ops
        result = run(system.H, system.jump_ops_batched, system.y0)
        assert result.ysave.shape == (bL, ntsave, n, n)
        assert result.exp_save.shape == (bL, n_exp_ops, ntsave)

        # batched y0
        result = run(system.H, system.jump_ops, system.y0_batched)
        assert result.ysave.shape == (by, ntsave, n, n)
        assert result.exp_save.shape == (by, n_exp_ops, ntsave)

        # batched H and jump_ops
        result = run(system.H_batched, system.jump_ops_batched, system.y0)
        assert result.ysave.shape == (bH, bL, ntsave, n, n)
        assert result.exp_save.shape == (bH, bL, n_exp_ops, ntsave)

        # batched H and y0
        result = run(system.H_batched, system.jump_ops, system.y0_batched)
        assert result.ysave.shape == (bH, by, ntsave, n, n)
        assert result.exp_save.shape == (bH, by, n_exp_ops, ntsave)

        # batched jump_ops and y0
        result = run(system.H, system.jump_ops_batched, system.y0_batched)
        assert result.ysave.shape == (bL, by, ntsave, n, n)
        assert result.exp_save.shape == (bL, by, n_exp_ops, ntsave)

        # batched H and jump_ops and y0
        result = run(system.H_batched, system.jump_ops_batched, system.y0_batched)
        assert result.ysave.shape == (bH, bL, by, ntsave, n, n)

        # batched second jump op but not the first one
        result = run(
            system.H_batched,
            [system.jump_ops_batched[0]] + system.jump_ops[1:],
            system.y0_batched,
        )
        assert result.ysave.shape == (bH, bL, by, ntsave, n, n)
