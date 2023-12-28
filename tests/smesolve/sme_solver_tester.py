from __future__ import annotations

import logging
from abc import ABC
from typing import Any

import torch

from dynamiqs.gradient import Gradient
from dynamiqs.solver import Solver

# from .mesolve.open_system import OpenSystem
# from .sesolve.closed_system import ClosedSystem
# from .system import System
from .monitored_system import MonitoredSystem


class SMESolverTester(ABC):
    def test_batching(self):
        pass

    def _test_batching(
        self,
        system: MonitoredSystem,
        solver: Solver,
        *,
        options: dict[str, Any] | None = None,
    ):
        n = system.n
        bH = len(system.Hb)
        bL = len(system.Lb[0])
        by = len(system.y0b)
        nE = len(system.E)
        nLm = len(system.etas[system.etas != 0])
        ntrajs = 10
        nt = 11
        tsave = system.tsave(nt)

        def run(Hb, Lb, yb):
            H = system.H if not Hb else system.Hb
            L = system.L if not Lb else system.Lb
            y0 = system.y0 if not yb else system.y0b
            return system.run(tsave, solver, options=options, H=H, L=L, y0=y0)

        # === test regular batching (cartesian product)

        # no batching
        result = run(Hb=False, Lb=False, yb=False)
        assert result.ysave.shape == (ntrajs, nt, n, n)
        assert result.Esave.shape == (ntrajs, nE, nt)
        assert result.Lmsave.shape == (ntrajs, nLm, nt - 1)

        # batched H
        result = run(Hb=True, Lb=False, yb=False)
        assert result.ysave.shape == (bH, ntrajs, nt, n, n)
        assert result.Esave.shape == (bH, ntrajs, nE, nt)
        assert result.Lmsave.shape == (bH, ntrajs, nLm, nt - 1)

        # batched y0
        result = run(Hb=False, Lb=False, yb=True)
        assert result.ysave.shape == (by, ntrajs, nt, n, n)
        assert result.Esave.shape == (by, ntrajs, nE, nt)
        assert result.Lmsave.shape == (by, ntrajs, nLm, nt - 1)

        # batched H and y0
        result = run(Hb=True, Lb=False, yb=True)
        assert result.ysave.shape == (bH, by, ntrajs, nt, n, n)
        assert result.ysave.shape == (bH, by, ntrajs, nt, n, n)
        assert result.Lmsave.shape == (bH, by, ntrajs, nLm, nt - 1)

        # batched L
        result = run(Hb=False, Lb=True, yb=False)
        assert result.ysave.shape == (bL, ntrajs, nt, n, n)
        assert result.Esave.shape == (bL, ntrajs, nE, nt)
        assert result.Lmsave.shape == (bL, ntrajs, nLm, nt - 1)

        # batched H and L
        result = run(Hb=True, Lb=True, yb=False)
        assert result.ysave.shape == (bH, bL, ntrajs, nt, n, n)
        assert result.Esave.shape == (bH, bL, ntrajs, nE, nt)
        assert result.Lmsave.shape == (bH, bL, ntrajs, nLm, nt - 1)

        # batched L and y0
        result = run(Hb=False, Lb=True, yb=True)
        assert result.ysave.shape == (bL, by, ntrajs, nt, n, n)
        assert result.Esave.shape == (bL, by, ntrajs, nE, nt)
        assert result.Lmsave.shape == (bL, by, ntrajs, nLm, nt - 1)

        # batched H and L and y0
        result = run(Hb=True, Lb=True, yb=True)
        assert result.ysave.shape == (bH, bL, by, ntrajs, nt, n, n)
        assert result.Esave.shape == (bH, bL, by, ntrajs, nE, nt)
        assert result.Lmsave.shape == (bH, bL, by, ntrajs, nLm, nt - 1)

        # === test non cartesian batching
        options = {} if options is None else options
        options['cartesian_batching'] = False
        b = 2

        Hb = system.Hb[:b]
        y0b = system.y0b[:b]

        Lb = [L[:b] for L in system.Lb]
        result = system.run(tsave, solver, options=options, H=Hb, L=Lb, y0=y0b)
        assert result.ysave.shape == (b, ntrajs, nt, n, n)
        assert result.Esave.shape == (b, ntrajs, nE, nt)
        assert result.Lmsave.shape == (b, ntrajs, nLm, nt - 1)

    def _test_correctness(
        self,
        system: MonitoredSystem,
        solver: Solver,
        *,
        options: dict[str, Any] | None = None,
        ntsave: int = 11,
        ysave_atol: float = 1e-3,
        esave_rtol: float = 1e-3,
        esave_atol: float = 1e-5,
    ):
        tsave = system.tsave(ntsave)
        result = system.run(tsave, solver, options=options, ntrajs=1)

        # === test ysave
        errs = torch.linalg.norm(result.ysave[0] - system.states(tsave), dim=(-2, -1))
        logging.warning(f'errs = {errs}')
        assert torch.all(errs <= ysave_atol)

        # === test Esave
        true_Esave = system.expects(tsave)
        logging.warning(f'Esave      = {result.Esave[0]}')
        logging.warning(f'true_Esave = {true_Esave}')
        assert torch.allclose(
            result.Esave[0], true_Esave, rtol=esave_rtol, atol=esave_atol
        )

    def test_correctness(self):
        pass

    def _test_gradient(
        self,
        system: MonitoredSystem,
        solver: Solver,
        gradient: Gradient,
        *,
        options: dict[str, Any] | None = None,
        ntsave: int = 11,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        tsave = system.tsave(ntsave)
        result = system.run(tsave, solver, gradient=gradient, options=options, ntrajs=1)

        # === test gradients depending on final ysave
        loss_state = system.loss_state(result.ysave[0, -1])
        grads_state = torch.autograd.grad(
            loss_state, system.params, retain_graph=True, allow_unused=True
        )
        print(grads_state)
        print(system.etas.requires_grad)
        print(system.kappa.requires_grad)
        grads_state = torch.stack(grads_state)
        true_grads_state = system.grads_state(tsave[-1])

        logging.warning(f'grads_state       = {grads_state}')
        logging.warning(f'true_grads_state  = {true_grads_state}')

        # === test gradients depending on final Esave
        loss_expect = system.loss_expect(result.Esave[0, :, -1])
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
