import torch

import dynamiqs as dq
from dynamiqs.options import Options

from .closed_system import ClosedSystem


class SESolverTester:
    def _test_batching(self, options: Options, system: ClosedSystem):
        """Test the batching of `H` and `psi0` in `sesolve`, and the returned object
        sizes."""
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
        psi_save, exp_save = run_sesolve(system.H, system.psi0)
        assert psi_save.shape == (num_t_save, n, 1)
        assert exp_save.shape == (n_exp_ops, num_t_save)

        # batched H
        psi_save, exp_save = run_sesolve(system.H_batched, system.psi0)
        assert psi_save.shape == (b_H, num_t_save, n, 1)
        assert exp_save.shape == (b_H, n_exp_ops, num_t_save)

        # batched psi0
        psi_save, exp_save = run_sesolve(system.H, system.psi0_batched)
        assert psi_save.shape == (b_psi0, num_t_save, n, 1)
        assert exp_save.shape == (b_psi0, n_exp_ops, num_t_save)

        # batched H and psi0
        psi_save, exp_save = run_sesolve(system.H_batched, system.psi0_batched)
        assert psi_save.shape == (b_H, b_psi0, num_t_save, n, 1)
        assert exp_save.shape == (b_H, b_psi0, n_exp_ops, num_t_save)

    def test_batching(self):
        pass

    def _test_psi_save(
        self,
        options: Options,
        system: ClosedSystem,
        *,
        num_t_save: int,
        norm_atol: float = 1e-2,
    ):
        t_save = system.t_save(num_t_save)

        psi_save, _ = dq.sesolve(
            system.H,
            system.psi0,
            t_save,
            exp_ops=system.exp_ops,
            options=options,
        )

        errs = torch.norm(psi_save - system.psis(t_save), dim=(-2, -1))
        assert torch.all(errs <= norm_atol)

    def test_psi_save(self):
        pass
