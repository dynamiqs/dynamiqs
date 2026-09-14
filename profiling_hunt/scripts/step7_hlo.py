"""Step 7 - where do the 3 trajectory copies live? Ask the compiler, not the allocator.

Part 4.B established that save_states=True costs 3 R per saved state at peak. Reading
the diffrax source (see the writeup) produced a candidate chain: the preallocated
buffer, the while loop carry, and a pair of nested lax.cond over the whole save_state
that diffrax always emits to handle the degenerate t0 == t1 case.

This script does not measure the allocator. It asks XLA directly, two ways:

  A) compiled.memory_analysis() across a sweep of ntsave at fixed N. This is the
     compiler's own static accounting, split into output vs temp, so the 3 can be
     decomposed rather than just observed.
  B) the buffer assignment dump for one configuration, so that the individual
     allocations of exactly ntsave * R bytes can be attributed to the HLO
     instructions that own them, by name.
"""

import os
import sys

DUMP = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hlo"
os.environ["XLA_FLAGS"] = f"--xla_dump_to={DUMP}"

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

MIB = 1024**2
N = 64
R = 16 * N * N
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1


def build(N):
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    L2 = np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N))
    L1 = np.sqrt(K1) * a
    return H, [L2, L1], dq.fock_dm(N, 0).asdense()


H, Ls, rho0 = build(N)
print(f"N={N}  R = 16 N^2 = {R:,d} B = {R / MIB:.4f} MiB", flush=True)
print(f"XLA_FLAGS = {os.environ['XLA_FLAGS']}", flush=True)
print(f"backend = {jax.default_backend()}  device = {jax.local_devices()[0]}\n",
      flush=True)


def compile_for(ntsave, save_states=True):
    ts = jnp.linspace(0.0, T_FINAL, ntsave)

    def f(rho0):
        return dq.mesolve(
            H, Ls, rho0, ts,
            method=dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000),
            save_states=save_states, progress_meter=False,
        )

    return jax.jit(f).lower(rho0).compile()


print("=" * 96)
print("A) compiled.memory_analysis(), sweep over ntsave at fixed N")
print("=" * 96)
print(f"{'ntsave':>7} {'traj=nt*R':>14} {'output B':>14} {'temp B':>14} "
      f"{'out+temp':>14} {'(out+temp)/traj':>16}")
rows = []
for nt in (10, 20, 40, 80, 160):
    c = compile_for(nt)
    m = c.memory_analysis()
    traj = nt * R
    tot = m.output_size_in_bytes + m.temp_size_in_bytes
    rows.append((nt, traj, m.output_size_in_bytes, m.temp_size_in_bytes, tot))
    print(f"{nt:>7} {traj:>14,} {m.output_size_in_bytes:>14,} "
          f"{m.temp_size_in_bytes:>14,} {tot:>14,} {tot / traj:>16.4f}")

# slope of each component against ntsave, in units of R
x = np.array([r[0] for r in rows], float)
for j, name in ((2, "output"), (3, "temp"), (4, "output+temp")):
    y = np.array([r[j] for r in rows], float)
    A = np.column_stack([np.ones_like(x), x])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ b
    print(f"  {name:<12s} = {b[0]:>14,.0f} B + {b[1] / R:>7.4f} R per saved state"
          f"   max|resid| = {np.abs(resid).max():,.0f} B")

print("\n  same sweep with save_states=False, as a control:")
for nt in (10, 160):
    c = compile_for(nt, save_states=False)
    m = c.memory_analysis()
    print(f"    ntsave={nt:>4}  output={m.output_size_in_bytes:>12,} B  "
          f"temp={m.temp_size_in_bytes:>12,} B")

print("\n" + "=" * 96)
print(f"B) buffer assignment dump written to {DUMP}")
print("=" * 96)
NT = 40
c = compile_for(NT)
print(f"   compiled one module at ntsave={NT}, traj = {NT * R:,d} B")
print(f"   {c.memory_analysis()}")
