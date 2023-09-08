import logging
from abc import ABC
from typing import Any

import torch

from .system import System


class SolverTester(ABC):
    def _test_batching(self, solver: str, options: dict[str, Any], system: System):
        """Test the batching of `H` and `y0`, and the returned object sizes."""
        m, n = system._state_shape
        n_exp_ops = len(system.exp_ops)
        b_H = len(system.H_batched)
        b_y0 = len(system.y0_batched)
        num_t_save = 11
        t_save = system.t_save(num_t_save)

        run = lambda H, y0: system._run(H, y0, t_save, solver, options)

        # no batching
        result = run(system.H, system.y0)
        assert result.y_save.shape == (num_t_save, m, n)
        assert result.exp_save.shape == (n_exp_ops, num_t_save)

        # batched H
        result = run(system.H_batched, system.y0)
        assert result.y_save.shape == (b_H, num_t_save, m, n)
        assert result.exp_save.shape == (b_H, n_exp_ops, num_t_save)

        # batched y0
        result = run(system.H, system.y0_batched)
        assert result.y_save.shape == (b_y0, num_t_save, m, n)
        assert result.exp_save.shape == (b_y0, n_exp_ops, num_t_save)

        # batched H and y0
        result = run(system.H_batched, system.y0_batched)
        assert result.y_save.shape == (b_H, b_y0, num_t_save, m, n)
        assert result.exp_save.shape == (b_H, b_y0, n_exp_ops, num_t_save)

    def test_batching(self):
        pass

    def _test_correctness(
        self,
        solver: str,
        options: dict[str, Any],
        system: System,
        *,
        num_t_save: int,
        y_save_norm_atol: float = 1e-3,
        exp_save_rtol: float = 1e-3,
        exp_save_atol: float = 1e-5,
    ):
        t_save = system.t_save(num_t_save)
        result = system.run(t_save, solver, options)

        # === test y_save
        errs = torch.linalg.norm(result.y_save - system.states(t_save), dim=(-2, -1))
        logging.warning(f'errs = {errs}')
        assert torch.all(errs <= y_save_norm_atol)

        # === test exp_save
        true_exp_save = system.expects(t_save)
        logging.warning(f'exp_save      = {result.exp_save}')
        logging.warning(f'true_exp_save = {true_exp_save}')
        assert torch.allclose(
            result.exp_save, true_exp_save, rtol=exp_save_rtol, atol=exp_save_atol
        )

    def test_correctness(self):
        pass

    def _test_gradient(
        self,
        solver: str,
        options: dict[str, Any],
        system: System,
        *,
        num_t_save: int,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        t_save = system.t_save(num_t_save)
        result = system.run(t_save, solver, options)

        # === test gradients depending on final y_save
        loss_state = system.loss_state(result.y_save[-1])
        grads_state = torch.autograd.grad(
            loss_state, system.parameters, retain_graph=True
        )
        grads_state = torch.stack(grads_state)
        true_grads_state = system.grads_state(t_save[-1])

        logging.warning(f'grads_state       = {grads_state}')
        logging.warning(f'true_grads_state  = {true_grads_state}')

        assert torch.allclose(grads_state, true_grads_state, rtol=rtol, atol=atol)

        # === test gradients depending on final exp_save
        loss_expect = system.loss_expect(result.exp_save[:, -1])
        grads_expect = [
            torch.stack(torch.autograd.grad(loss, system.parameters, retain_graph=True))
            for loss in loss_expect
        ]
        grads_expect = torch.stack(grads_expect)
        true_grads_expect = system.grads_expect(t_save[-1])

        logging.warning(f'grads_expect      = {grads_expect}')
        logging.warning(f'true_grads_expect = {true_grads_expect}')

        assert torch.allclose(grads_expect, true_grads_expect, rtol=rtol, atol=atol)
