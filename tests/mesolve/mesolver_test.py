import torch

import torchqdynamics as tq
from torchqdynamics.options import Options

from .open_system import OpenSystem


class MESolverTest:
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
