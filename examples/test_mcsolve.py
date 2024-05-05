import dynamiqs as dq
import jax.numpy as jnp
from dynamiqs import timecallable, unit
from dynamiqs.solver import Tsit5
from jax.random import PRNGKey
import matplotlib.pyplot as plt


omega = 2.0 * jnp.pi * 1.0
amp = 2.0 * jnp.pi * 0.0


def H_func(t):
    return -0.5 * omega * dq.sigmaz() + jnp.cos(omega * t) * amp * dq.sigmax()

tsave = jnp.linspace(0, 1.0, 41)
jump_ops = [0.4 * dq.basis(2, 0) @ dq.tobra(dq.basis(2, 1)),]
exp_ops = [dq.basis(2, 0) @ dq.tobra(dq.basis(2, 0)), dq.basis(2, 1) @ dq.tobra(dq.basis(2, 1))]

initial_states = [dq.basis(2, 1),]

num_traj = 31
options = dq.Options(ntraj=num_traj)
result = dq.mcsolve(
    timecallable(H_func),
    jump_ops,
    initial_states,
    tsave,
    key=PRNGKey(4242434),
    exp_ops=exp_ops,
    solver=Tsit5(),
    options=options
)
result_me = dq.mesolve(
    timecallable(H_func),
    jump_ops,
    initial_states,
    tsave,
    exp_ops=exp_ops,
    solver=Tsit5(),
)

fig, ax = plt.subplots()
plt.plot(tsave, jnp.real(result.expects[1, 0]), label="0")
plt.plot(tsave, jnp.real(result.expects[1, 1]), label="1")
plt.plot(tsave, jnp.real(result_me.expects[1, 0]), ls="--", label="me 0")
plt.plot(tsave, jnp.real(result_me.expects[1, 1]), ls="--", label="me 1")
ax.set_ylabel("population")
ax.set_xlabel("time [ns]")
ax.legend()
plt.show()

print(0)
