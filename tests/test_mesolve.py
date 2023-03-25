import numpy as np
import qutip as qt

import torchqdynamics as tq


def test_mesolve_batching():
    """Test the batching of H and rho0 in mesolve, and the returned object sizes."""
    # parameters
    n = 8
    kappa = 1.0
    delta = 2 * np.pi
    alpha0 = 1.0

    # operators
    a = qt.destroy(n)
    adag = a.dag()
    H = delta * adag * a
    H_batched = [0.5 * H, H, 2 * H]
    b_H = len(H_batched)
    jump_ops = [np.sqrt(kappa) * a, np.sqrt(kappa) * a]
    exp_ops = [(a + adag) / np.sqrt(2), (a - adag) / (np.sqrt(2) * 1j)]
    num_exp_ops = len(exp_ops)

    # other arguments
    rho0 = qt.coherent(n, alpha0)
    rho0_batched = [
        qt.coherent(n, alpha0),
        qt.coherent(n, 1j * alpha0),
        qt.coherent(n, -alpha0),
        qt.coherent(n, -1j * alpha0)
    ]
    b_rho0 = len(rho0_batched)
    num_t_save = 51
    t_save = np.linspace(0.0, delta / (2 * np.pi), num_t_save)  # a full rotation
    solver = tq.solver.Rouchon(dt=1e-4, order=1)

    run_mesolve = lambda H, rho0: tq.mesolve(
        H, jump_ops, rho0, t_save, exp_ops=exp_ops, solver=solver
    )

    # no batching
    states, exp = run_mesolve(H, rho0)
    assert states.shape == (num_t_save, n, n)
    assert exp.shape == (num_exp_ops, num_t_save)

    # batched H
    states, exp = run_mesolve(H_batched, rho0)
    assert states.shape == (b_H, num_t_save, n, n)
    assert exp.shape == (b_H, num_exp_ops, num_t_save)

    # batched rho0
    states, exp = run_mesolve(H, rho0_batched)
    assert states.shape == (b_rho0, num_t_save, n, n)
    assert exp.shape == (b_rho0, num_exp_ops, num_t_save)

    # batched H and rho0
    states, exp = run_mesolve(H_batched, rho0_batched)
    assert states.shape == (b_H, b_rho0, num_t_save, n, n)
    assert exp.shape == (b_H, b_rho0, num_exp_ops, num_t_save)
