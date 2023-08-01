import logging

import torch

import dynamiqs as dq
from dynamiqs.options import Options

from .open_system import OpenSystem


class MESolverTester:
    def _test_batching(self, options: Options, system: OpenSystem):
        """Test the batching of `H` and `rho0` in `mesolve`, and the returned object
        shapes."""
        n = system.n
        n_exp_ops = len(system.exp_ops)
        b_H = len(system.H_batched)
        b_rho0 = len(system.rho0_batched)
        num_t_save = 11

        run_mesolve = lambda H, rho0: dq.mesolve(
            H,
            system.jump_ops,
            rho0,
            system.t_save(num_t_save),
            exp_ops=system.exp_ops,
            options=options,
        )

        # no batching
        result = run_mesolve(system.H, system.rho0)
        assert result.y_save.shape == (num_t_save, n, n)
        assert result.exp_save.shape == (n_exp_ops, num_t_save)

        # batched H
        result = run_mesolve(system.H_batched, system.rho0)
        assert result.y_save.shape == (b_H, num_t_save, n, n)
        assert result.exp_save.shape == (b_H, n_exp_ops, num_t_save)

        # batched rho0
        result = run_mesolve(system.H, system.rho0_batched)
        assert result.y_save.shape == (b_rho0, num_t_save, n, n)
        assert result.exp_save.shape == (b_rho0, n_exp_ops, num_t_save)

        # batched H and rho0
        result = run_mesolve(system.H_batched, system.rho0_batched)
        assert result.y_save.shape == (b_H, b_rho0, num_t_save, n, n)
        assert result.exp_save.shape == (b_H, b_rho0, n_exp_ops, num_t_save)

    def test_batching(self):
        pass

    def _test_correctness(
        self,
        options: Options,
        system: OpenSystem,
        *,
        num_t_save: int,
        y_save_norm_atol: float = 1e-2,
        exp_save_rtol: float = 1e-2,
        exp_save_atol: float = 1e-2,
    ):
        t_save = system.t_save(num_t_save)
        result = system.mesolve(t_save, options)

        # test y_save
        errs = torch.linalg.norm(result.y_save - system.rhos(t_save), dim=(-2, -1))
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


class MEGradientSolverTester(MESolverTester):
    def _test_gradient(
        self,
        options: Options,
        system: OpenSystem,
        *,
        num_t_save: int,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        t_save = system.t_save(num_t_save)
        result = system.mesolve(t_save, options)

        # === test gradients depending on y_save
        loss_rho = system.loss_rho(result.y_save[-1])
        grads_rho = torch.autograd.grad(loss_rho, system.parameters, retain_graph=True)
        grads_rho = torch.stack(grads_rho)
        true_grads_rho = system.grads_rho(t_save[-1])

        logging.warning(f'grads_rho         = {grads_rho}')
        logging.warning(f'true_grads_rho    = {true_grads_rho}')

        assert torch.allclose(grads_rho, true_grads_rho, rtol=rtol, atol=atol)

        # === test gradient depending on exp_save
        losses_expect = system.losses_expect(result.exp_save[:, -1])
        grads_expect = [
            torch.stack(torch.autograd.grad(loss, system.parameters, retain_graph=True))
            for loss in losses_expect
        ]
        grads_expect = torch.stack(grads_expect)
        true_grads_expect = system.grads_expect(t_save[-1])

        logging.warning(f'grads_expect      = {grads_expect}')
        logging.warning(f'true_grads_expect = {true_grads_expect}')

        assert torch.allclose(grads_expect, true_grads_expect, rtol=rtol, atol=atol)
