import torch

import dynamiqs as dq
from dynamiqs.options import Options
from dynamiqs.utils.solver_utils import bexpect

from .open_system import OpenSystem


class MESolverTester:
    def _test_batching(self, options: Options, system: OpenSystem):
        """Test the batching of `H` and `rho0` in `mesolve`, and the returned object
        sizes."""
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
        rho_save, exp_save = run_mesolve(system.H, system.rho0)
        assert rho_save.shape == (num_t_save, n, n)
        assert exp_save.shape == (n_exp_ops, num_t_save)

        # batched H
        rho_save, exp_save = run_mesolve(system.H_batched, system.rho0)
        assert rho_save.shape == (b_H, num_t_save, n, n)
        assert exp_save.shape == (b_H, n_exp_ops, num_t_save)

        # batched rho0
        rho_save, exp_save = run_mesolve(system.H, system.rho0_batched)
        assert rho_save.shape == (b_rho0, num_t_save, n, n)
        assert exp_save.shape == (b_rho0, n_exp_ops, num_t_save)

        # batched H and rho0
        rho_save, exp_save = run_mesolve(system.H_batched, system.rho0_batched)
        assert rho_save.shape == (b_H, b_rho0, num_t_save, n, n)
        assert exp_save.shape == (b_H, b_rho0, n_exp_ops, num_t_save)

    def test_batching(self):
        pass

    def _test_rho_save(
        self,
        options: Options,
        system: OpenSystem,
        *,
        num_t_save: int,
        norm_atol: float = 1e-2,
    ):
        t_save = system.t_save(num_t_save)
        rho_save, _ = system.mesolve(t_save, options)

        errs = torch.norm(rho_save - system.rhos(t_save), dim=(-2, -1))
        assert torch.all(errs <= norm_atol)

    def test_rho_save(self):
        pass


class MEAdjointSolverTester(MESolverTester):
    def _test_adjoint(
        self,
        options: Options,
        system: OpenSystem,
        *,
        num_t_save: int,
        rtol: float = 1e-5,
        atol: float = 3e-3,
    ):
        t_save = system.t_save(num_t_save)

        # compute autograd gradients
        system.reset()  # required to not backward through the same graph twice
        options.gradient_alg = 'autograd'
        rho_save, _ = system.mesolve(t_save, options)
        loss = system.loss(rho_save[-1])
        grad_autograd = torch.autograd.grad(loss, system.parameters)

        # compute adjoint gradients
        system.reset()  # required to not backward through the same graph twice
        options.gradient_alg = 'adjoint'
        options.parameters = system.parameters
        rho_save, _ = system.mesolve(t_save, options)
        loss = system.loss(rho_save[-1])
        grad_adjoint = torch.autograd.grad(loss, system.parameters)

        # check gradients are equal
        for g1, g2 in zip(grad_autograd, grad_adjoint):
            assert torch.allclose(g1, g2, rtol=rtol, atol=atol)

    def test_adjoint(self):
        pass

    def _test_adjoint_expops(
        self,
        options: Options,
        system: OpenSystem,
        *,
        num_t_save: int,
        rtol: float = 1e-5,
        atol: float = 1e-4,
    ):
        t_save = system.t_save(num_t_save)

        # define options
        options.gradient_alg = 'adjoint'
        options.parameters = system.parameters

        # compute autograd gradients
        system.reset()  # required to not backward through the same graph twice
        rho_save, _ = system.mesolve(t_save, options)
        exp_save = bexpect(torch.stack(system.exp_ops), rho_save).mT
        grad_rho = torch.autograd.grad(exp_save.abs().sum(), system.parameters)

        # compute adjoint gradients
        system.reset()  # required to not backward through the same graph twice
        rho_save, exp_save = system.mesolve(t_save, options)
        grad_exp = torch.autograd.grad(exp_save.abs().sum(), system.parameters)

        # check gradients are equal
        for g1, g2 in zip(grad_rho, grad_exp):
            assert torch.allclose(g1, g2, rtol=rtol, atol=atol)

    def test_adjoint_expops(self):
        pass
