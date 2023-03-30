import torch

import torchqdynamics as tq
from torchqdynamics.solver_options import SolverOption

from .open_system import OpenSystem


class MESolverTest:
    def _test_batching(self, solver: SolverOption, system: OpenSystem):
        """Test the batching of `H` and `rho0` in `mesolve`, and the returned object
        sizes."""
        n = system.n
        n_exp_ops = len(system.exp_ops)
        b_H = len(system.H_batched)
        b_rho0 = len(system.rho0_batched)
        nt = 11

        run_mesolve = lambda H, rho0: tq.mesolve(
            H,
            system.jump_ops,
            rho0,
            system.t_save(nt),
            exp_ops=system.exp_ops,
            solver=solver,
        )

        # no batching
        rho_save, exp_save = run_mesolve(system.H, system.rho0)
        assert rho_save.shape == (nt, n, n)
        assert exp_save.shape == (n_exp_ops, nt)

        # batched H
        rho_save, exp_save = run_mesolve(system.H_batched, system.rho0)
        assert rho_save.shape == (b_H, nt, n, n)
        assert exp_save.shape == (b_H, n_exp_ops, nt)

        # batched rho0
        rho_save, exp_save = run_mesolve(system.H, system.rho0_batched)
        assert rho_save.shape == (b_rho0, nt, n, n)
        assert exp_save.shape == (b_rho0, n_exp_ops, nt)

        # batched H and rho0
        rho_save, exp_save = run_mesolve(system.H_batched, system.rho0_batched)
        assert rho_save.shape == (b_H, b_rho0, nt, n, n)
        assert exp_save.shape == (b_H, b_rho0, n_exp_ops, nt)

    def test_batching(self):
        pass

    def _test_rho_save(
        self,
        solver: SolverOption,
        system: OpenSystem,
        *,
        nt: int,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ):
        t_save = system.t_save(nt)

        rho_save, _ = tq.mesolve(
            system.H,
            system.jump_ops,
            system.rho0,
            t_save,
            exp_ops=system.exp_ops,
            solver=solver,
        )

        assert torch.allclose(rho_save, system.rhos(t_save), rtol=rtol, atol=atol)

    def test_rho_save(self):
        pass
