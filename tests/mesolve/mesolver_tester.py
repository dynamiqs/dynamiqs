import torch

import torchqdynamics as tq
from torchqdynamics.options import Options

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

        run_mesolve = lambda H, rho0: tq.mesolve(
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

        rho_save, _ = tq.mesolve(
            system.H,
            system.jump_ops,
            system.rho0,
            t_save,
            exp_ops=system.exp_ops,
            options=options,
        )

        errs = torch.norm(rho_save - system.rhos(t_save), dim=(-2, -1))
        assert torch.all(errs <= norm_atol)

    def test_rho_save(self):
        pass


class AdjointMESolverTester(MESolverTester):
    def _test_adjoint(
        self,
        options: Options,
        system: OpenSystem,
        *,
        num_t_save: int,
        atol: float = 1e-3,
        rtol: float = 1e-5,
    ):
        # function to run mesolve with a specific gradient algorithm
        def run_mesolve(gradient_alg, parameters):
            return tq.mesolve(
                system.H,
                system.jump_ops,
                system.rho0,
                system.t_save(num_t_save),
                exp_ops=system.exp_ops,
                options=options,
                gradient_alg=gradient_alg,
                parameters=parameters,
            )

        # compute autograd gradients
        system.init_operators()  # required to not backward through the same graph twice
        rho_save, _ = run_mesolve('autograd', None)
        loss = system.loss(rho_save[-1])
        grad_autograd = torch.autograd.grad(loss, system.parameters)

        # compute adjoint gradients
        system.init_operators()  # required to not backward through the same graph twice
        rho_save, _ = run_mesolve('adjoint', system.parameters)
        loss = system.loss(rho_save[-1])
        grad_adjoint = torch.autograd.grad(loss, system.parameters)

        # check gradients are equal
        for g1, g2 in zip(grad_autograd, grad_adjoint):
            assert torch.allclose(g1, g2, atol=atol, rtol=rtol)

    def test_adjoint(self):
        pass
