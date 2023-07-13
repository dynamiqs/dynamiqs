import logging

import torch

import dynamiqs as dq
from dynamiqs.options import Options

from .closed_system import ClosedSystem


class SESolverTester:
    def _test_batching(self, options: Options, system: ClosedSystem):
        """Test the batching of `H` and `psi0` in `sesolve`, and the returned object
        shapes."""
        n = system.n
        n_exp_ops = len(system.exp_ops)
        b_H = len(system.H_batched)
        b_psi0 = len(system.psi0_batched)
        num_t_save = 11

        run_sesolve = lambda H, psi0: dq.sesolve(
            H,
            psi0,
            system.t_save(num_t_save),
            exp_ops=system.exp_ops,
            options=options,
        )

        # no batching
        result = run_sesolve(system.H, system.psi0)
        assert result.y_save.shape == (num_t_save, n, 1)
        assert result.exp_save.shape == (n_exp_ops, num_t_save)

        # batched H
        result = run_sesolve(system.H_batched, system.psi0)
        assert result.y_save.shape == (b_H, num_t_save, n, 1)
        assert result.exp_save.shape == (b_H, n_exp_ops, num_t_save)

        # batched psi0
        result = run_sesolve(system.H, system.psi0_batched)
        assert result.y_save.shape == (b_psi0, num_t_save, n, 1)
        assert result.exp_save.shape == (b_psi0, n_exp_ops, num_t_save)

        # batched H and psi0
        result = run_sesolve(system.H_batched, system.psi0_batched)
        assert result.y_save.shape == (b_H, b_psi0, num_t_save, n, 1)
        assert result.exp_save.shape == (b_H, b_psi0, n_exp_ops, num_t_save)

    def test_batching(self):
        pass

    def _test_correctness(
        self,
        options: Options,
        system: ClosedSystem,
        *,
        num_t_save: int,
        y_save_norm_atol: float = 1e-2,
        exp_save_rtol: float = 1e-2,
        exp_save_atol: float = 1e-2,
    ):
        t_save = system.t_save(num_t_save)
        result = system.sesolve(t_save, options)

        # test y_save
        errs = torch.linalg.norm(result.y_save - system.psis(t_save), dim=(-2, -1))
        assert torch.all(errs <= y_save_norm_atol)

        # test exp_save
        assert torch.allclose(
            result.exp_save,
            system.expects(t_save),
            rtol=exp_save_rtol,
            atol=exp_save_atol,
        )

    def test_correctness(self):
        pass


class SEAutogradSolverTester(SESolverTester):
    def _test_autograd(
        self,
        options: Options,
        system: ClosedSystem,
        *,
        num_t_save: int,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        t_save = system.t_save(num_t_save)
        result = system.sesolve(t_save, options)

        # === test gradients depending on y_save
        loss_psi = system.loss_psi(result.y_save[-1])
        grads_loss_psi = torch.autograd.grad(
            loss_psi, system.parameters, retain_graph=True
        )
        grads_loss_psi = torch.stack(grads_loss_psi)
        true_grads_loss_psi = system.grads_loss_psi(t_save[-1])

        logging.warning(f'grads_loss_psi           = {grads_loss_psi}')
        logging.warning(f'true_grads_loss_psi      = {true_grads_loss_psi}')

        assert torch.allclose(grads_loss_psi, true_grads_loss_psi, rtol=rtol, atol=atol)

        # === test gradient depending on exp_save
        losses_expect = system.losses_expect(result.exp_save[:, -1])
        grads_losses_expect = [
            torch.stack(torch.autograd.grad(loss, system.parameters, retain_graph=True))
            for loss in losses_expect
        ]
        grads_losses_expect = torch.stack(grads_losses_expect)
        true_grads_losses_expect = system.grads_losses_expect(t_save[-1])

        logging.warning(f'grads_losses_expect      = {grads_losses_expect}')
        logging.warning(f'true_grads_losses_expect = {true_grads_losses_expect}')

        assert torch.allclose(
            grads_losses_expect, true_grads_losses_expect, rtol=rtol, atol=atol
        )

    def test_autograd(self):
        pass
