import torch

import torchqdynamics as tq
from torchqdynamics.solver_options import SolverOption

from .closed_system import ClosedSystem


class SESolverTest:
    def _test_batching(self, solver: SolverOption, system: ClosedSystem):
        """Test the batching of `H` and `psi0` in `sesolve`, and the returned object
        sizes."""
        n = system.n
        n_exp_ops = len(system.exp_ops)
        b_H = len(system.H_batched)
        b_psi0 = len(system.psi0_batched)
        nt = 11

        run_sesolve = lambda H, psi0: tq.sesolve(
            H,
            psi0,
            system.t_save(nt),
            exp_ops=system.exp_ops,
            solver=solver,
        )

        # no batching
        psi_save, exp_save = run_sesolve(system.H, system.psi0)
        assert psi_save.shape == (nt, n, 1)
        assert exp_save.shape == (n_exp_ops, nt)

        # batched H
        psi_save, exp_save = run_sesolve(system.H_batched, system.psi0)
        assert psi_save.shape == (b_H, nt, n, 1)
        assert exp_save.shape == (b_H, n_exp_ops, nt)

        # batched psi0
        psi_save, exp_save = run_sesolve(system.H, system.psi0_batched)
        assert psi_save.shape == (b_psi0, nt, n, 1)
        assert exp_save.shape == (b_psi0, n_exp_ops, nt)

        # batched H and psi0
        psi_save, exp_save = run_sesolve(system.H_batched, system.psi0_batched)
        assert psi_save.shape == (b_H, b_psi0, nt, n, 1)
        assert exp_save.shape == (b_H, b_psi0, n_exp_ops, nt)

    def test_batching(self):
        pass

    def _test_psi_save(
        self,
        solver: SolverOption,
        system: ClosedSystem,
        *,
        nt: int,
        rtol: float = 1e-05,
        atol: float = 1e-08,
    ):
        t_save = system.t_save(nt)

        psi_save, _ = tq.sesolve(
            system.H,
            system.psi0,
            t_save,
            exp_ops=system.exp_ops,
            solver=solver,
        )

        assert torch.allclose(psi_save, system.psis(t_save), rtol=rtol, atol=atol)

    def test_psi_save(self):
        pass
