import numpy as np
import qutip as qt

import torchqdynamics as tq


def test_sesolve_euler():
    """Cheap test of the Euler method of sesolve."""
    # parameters
    n = 8
    delta = 2 * np.pi
    alpha0 = 1.0

    # operators
    a = qt.destroy(n)
    adag = a.dag()
    H = delta * adag * a
    exp_ops = [(a + adag) / np.sqrt(2), (a - adag) / (np.sqrt(2) * 1j)]
    num_exp_ops = len(exp_ops)

    # other arguments
    rho0 = qt.coherent(n, alpha0)
    num_t_save = 51
    t_save = np.linspace(0.0, delta / (2 * np.pi), num_t_save)  # a full tour
    solver = tq.solver.Euler(dt=1e-3)

    # run solver
    states, exp = tq.sesolve(H, rho0, t_save, exp_ops=exp_ops, solver=solver)
    assert states.shape == (num_t_save, n, 1)
    assert exp.shape == (num_exp_ops, num_t_save)
