"""Step 7b - causal test: is the 2 R of temp really diffrax's throw=True error check?

The HLO buffer assignment for a save_states=True solve shows exactly three
trajectory-sized buffers, and names two of them:

  jit(diffeqsolve)/jit(branched_error_if_impl)/cond/branch_1_fun/pure_callback

That is equinox's "raise" error handler, which is two CHAINED jax.pure_callback calls
(equinox/_errors.py lines 111 and 115). diffrax pipes the entire Solution, trajectory
included, through it because diffeqsolve defaults to throw=True. A pure_callback is a
custom-call: its operand and its result cannot alias, so each one forces a fresh
full-size buffer.

This script tests that causally, by compiling the identical solve under each value of
EQX_ON_ERROR (read once at equinox import, hence one process per setting) and reading
the compiler's own accounting. If the diagnosis is right, the temp slope should fall
from 2 R per saved state to ~0.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import equinox  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

N = 64
R = 16 * N * N
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1

a = dq.destroy(N)
H = DELTA * dq.dag(a) @ a
Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
rho0 = dq.fock_dm(N, 0).asdense()

setting = os.environ.get("EQX_ON_ERROR", "<unset, defaults to raise>")
print(f"EQX_ON_ERROR = {setting}   (equinox sees: {equinox._config.EQX_ON_ERROR})",
      flush=True)
print(f"N={N}  R={R:,d} B", flush=True)


def compile_for(nt):
    ts = jnp.linspace(0.0, T_FINAL, nt)

    def f(rho0):
        return dq.mesolve(
            H, Ls, rho0, ts,
            method=dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000),
            save_states=True, progress_meter=False,
        )

    return jax.jit(f).lower(rho0).compile()


print(f"{'ntsave':>7} {'output B':>13} {'temp B':>13} {'out+temp':>13} "
      f"{'total/traj':>11}")
xs, outs, temps = [], [], []
for nt in (10, 20, 40, 80, 160):
    m = compile_for(nt).memory_analysis()
    tot = m.output_size_in_bytes + m.temp_size_in_bytes
    xs.append(nt)
    outs.append(m.output_size_in_bytes)
    temps.append(m.temp_size_in_bytes)
    print(f"{nt:>7} {m.output_size_in_bytes:>13,} {m.temp_size_in_bytes:>13,} "
          f"{tot:>13,} {tot / (nt * R):>11.4f}")

x = np.array(xs, float)
A = np.column_stack([np.ones_like(x), x])
for name, y in (("output", np.array(outs, float)), ("temp", np.array(temps, float)),
                ("output+temp", np.array(outs, float) + np.array(temps, float))):
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    print(f"  {name:<12s} = {b[0]:>12,.0f} B + {b[1] / R:>7.4f} R per saved state")
