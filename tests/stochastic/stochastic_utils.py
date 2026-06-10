import jax.numpy as jnp
import jax.tree_util as jtu
import optimistix as optx

import dynamiqs as dq

# all four stochastic solver APIs
SOLVERS = ['jssesolve', 'dssesolve', 'jsmesolve', 'dsmesolve']


def event_method() -> dq.method.Event:
    # Event method with a Newton root finder to refine the click times, same setup
    # as in `tests/jssesolve/test_jssesolve.py`
    root_finder = optx.Newton(1e-4, 1e-4, jtu.Partial(optx.rms_norm))
    return dq.method.Event(root_finder=root_finder)


def run_stochastic_solver(
    solver,
    method,
    H,
    jump_ops,
    psi0,
    tsave,
    keys,
    *,
    exp_ops=None,
    eta=1.0,
    theta=0.0,
    save_states=True,
):
    """Call one of the four stochastic solver APIs with a uniform signature.

    The initial state `psi0` is a ket, it is converted to a density matrix for the
    SME solvers. All measured channels share the same measurement efficiency `eta`,
    and the same dark count rate `theta` for the jump SME.
    """
    if solver == 'jssesolve':
        return dq.jssesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            keys,
            exp_ops=exp_ops,
            method=method,
            save_states=save_states,
        )
    elif solver == 'dssesolve':
        return dq.dssesolve(
            H,
            jump_ops,
            psi0,
            tsave,
            keys,
            exp_ops=exp_ops,
            method=method,
            save_states=save_states,
        )
    elif solver == 'jsmesolve':
        thetas = theta * jnp.ones(len(jump_ops))
        etas = eta * jnp.ones(len(jump_ops))
        return dq.jsmesolve(
            H,
            jump_ops,
            thetas,
            etas,
            psi0.todm(),
            tsave,
            keys,
            exp_ops=exp_ops,
            method=method,
            save_states=save_states,
        )
    elif solver == 'dsmesolve':
        etas = eta * jnp.ones(len(jump_ops))
        return dq.dsmesolve(
            H,
            jump_ops,
            etas,
            psi0.todm(),
            tsave,
            keys,
            exp_ops=exp_ops,
            method=method,
            save_states=save_states,
        )
    else:
        raise ValueError(f'Unknown stochastic solver `{solver}`.')
