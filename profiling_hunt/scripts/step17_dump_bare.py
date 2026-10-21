"""Step 17 - HLO dump for BARE diffrax, throw=False, at the SAME size as Step 11's
dynamiqs dump (N=64, ntsave=40), for a direct buffer-assignment comparison. Step 11
already showed the dynamiqs-patched module's temp buffer is owned by
`loop_dynamic_update_slice_fusion.1` -- the loop's own save-buffer. This checks
whether the SAME fusion instruction shows up in bare diffrax's temp allocation (and
therefore whether bare diffrax reaches 1 R by NOT needing that fusion at all, or by
having it but still aliasing it into the output).
"""
import os
import sys

os.environ["XLA_FLAGS"] = f"--xla_dump_to={sys.argv[1]}"

import jax  # noqa: E402
jax.config.update("jax_enable_x64", True)

import diffrax as dx  # noqa: E402
import jax.numpy as jnp  # noqa: E402

N = 64
R = 16 * N * N
nt = 40

M = jax.random.normal(jax.random.key(0), (N, N), dtype=jnp.complex128)
y0 = jax.random.normal(jax.random.key(1), (N, N), dtype=jnp.complex128)
term = dx.ODETerm(lambda t, y, args: M @ y)
ts = jnp.linspace(0.0, 0.1, nt)
saveat = dx.SaveAt(ts=ts)


def f(y0):
    return dx.diffeqsolve(
        term, dx.Tsit5(), t0=0.0, t1=0.1, dt0=None, y0=y0, saveat=saveat,
        stepsize_controller=dx.PIDController(rtol=1e-6, atol=1e-6),
        max_steps=100_000, throw=False,
    )


c = jax.jit(f).lower(y0).compile()
print(f"N={N} R={R:,d} B  ntsave={nt}  traj={nt * R:,d} B")
print(c.memory_analysis())
