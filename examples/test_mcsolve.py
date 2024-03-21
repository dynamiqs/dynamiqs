import os
#os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import timecallable, Options
import matplotlib.pyplot as plt
from dynamiqs.solver import Dopri5, Dopri8, Euler, Propagator, Solver, Tsit5

omega = 2.0 * jnp.pi * 1.0
amp = 2.0 * jnp.pi * 0.5


def H_func(t, omega, omega_d, amp):
    return -0.5 * omega * dq.sigmaz() + jnp.cos(omega_d * t) * amp * dq.sigmax()

tsave = jnp.linspace(0, 1.0, 101)

# exp_ops = [dq.basis(2, 0) @ dq.tobra(dq.basis(2, 0)),
#            dq.basis(2, 1) @ dq.tobra(dq.basis(2, 1))
#            ]
exp_ops = []

jump_ops = [1.0 * dq.basis(2, 0) @ dq.tobra(dq.basis(2, 1)), ]

options = Options(t0=tsave[0], t1=tsave[-1])

# run simulation
result = dq.mcsolve(
    timecallable(H_func, args=(omega, omega, amp)),
    jump_ops,
    dq.basis(2, 0),
    tsave,
    exp_ops=exp_ops,
    solver=Euler(dt=0.001),
    options=options,
)

# fig, ax = plt.subplots()
# plt.plot(tsave, result.expects[0], label="0")
# plt.plot(tsave, result.expects[1], label="1")
# plt.show()
#
print(result)
