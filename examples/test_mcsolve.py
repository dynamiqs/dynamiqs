import os
os.environ["JAX_TRACEBACK_FILTERING"] = "off"

import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import timecallable
import matplotlib.pyplot as plt

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

# run simulation
result = dq.mcsolve(
    timecallable(H_func, args=(omega, omega, amp)),
    jump_ops,
    dq.basis(2, 0),
    tsave,
    exp_ops=exp_ops
)

fig, ax = plt.subplots()
plt.plot(tsave, result.expects[0], label="0")
plt.plot(tsave, result.expects[1], label="1")
plt.show()

print(result)
