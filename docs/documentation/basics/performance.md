# Performance

Dynamiqs is built on [JAX](https://jax.readthedocs.io/), so a simulation is first compiled and then executed on the CPU, GPU or TPU. This page collects the settings that have the largest impact on run time, and explains how to measure their effect on your own simulations.

```python
import dynamiqs as dq
import jax.numpy as jnp
```

## Turn off the progress meter

By default, solvers display a [tqdm](https://github.com/tqdm/tqdm) progress meter. Updating it requires a callback from the device to the Python host at every solver step, which forces a synchronization and prevents JAX from dispatching work asynchronously. This is invisible for long simulations, but it is a noticeable fraction of the run time for short ones, and it is more costly on GPU than on CPU.

For production runs, disable it with the `progress_meter` argument:

```python
H = dq.sigmax()
psi0 = dq.fock(2, 0)
tsave = jnp.linspace(0.0, 1.0, 11)
result = dq.sesolve(H, psi0, tsave, progress_meter=False)
```

or globally with [`dq.set_progress_meter()`][dynamiqs.set_progress_meter]:

```python
dq.set_progress_meter(False)
```

<!-- restore the default progress meter
```python
dq.set_progress_meter(True)
```
-->

## Store operators in the `dia` layout

Many operators of interest in quantum physics are banded: $a$, $a^\dagger$, $a^\dagger a$, the Pauli matrices, and any tensor product thereof only have a few non-zero diagonals. Dynamiqs stores such operators in a sparse diagonal layout (`dia`), which is the default, and only falls back to a dense layout when needed. See [`dq.set_layout()`][dynamiqs.set_layout] to change the default layout.

Two common patterns silently convert operators to dense:

- converting to a plain array with `.to_jax()` or `jnp.asarray()`, then building the Hamiltonian from arrays,
- adding a dense operator to a sparse one, which makes the whole sum dense.

If performance matters, check the layout of your Hamiltonian and jump operators before passing them to a solver:

```pycon
>>> dq.destroy(8).dag() @ dq.destroy(8)
QArray: shape=(8, 8), dims=(8,), dtype=complex64, layout=dia, ndiags=1
[[  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
 [  ⋅    1.+0.j   ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
 [  ⋅      ⋅    2.+0.j   ⋅      ⋅      ⋅      ⋅      ⋅   ]
 [  ⋅      ⋅      ⋅    3.+0.j   ⋅      ⋅      ⋅      ⋅   ]
 [  ⋅      ⋅      ⋅      ⋅    4.+0.j   ⋅      ⋅      ⋅   ]
 [  ⋅      ⋅      ⋅      ⋅      ⋅    5.+0.j   ⋅      ⋅   ]
 [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅    6.+0.j   ⋅   ]
 [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅      ⋅    7.+0.j]]
```

## Declare the discontinuities of time-dependent operators

Adaptive step-size methods assume that the vector field is smooth. Whenever a Hamiltonian or a jump operator jumps discontinuously, the solver has to shrink the step size until the jump falls outside of it, rejecting many steps in the process.

To avoid this, Dynamiqs collects the discontinuity times of all time-dependent operators of a simulation, and instructs the solver to stop just before each of them and to restart just after. This is automatic for [`dq.pwc()`][dynamiqs.pwc] and for the `tstart`/`tend` bounds set by [`TimeQArray.clip()`][dynamiqs.TimeQArray], but a function passed to [`dq.modulated()`][dynamiqs.modulated] or [`dq.timecallable()`][dynamiqs.timecallable] is opaque, so its discontinuities must be declared explicitly:

```python
f = lambda t: jnp.where(t < 0.5, 1.0, -1.0)
H = dq.modulated(f, dq.sigmax(), discontinuity_ts=[0.5])
```

## Trade accuracy for speed

Simulations run in single precision (`complex64`) by default, see [the sharp bits](../getting_started/sharp-bits.md#floating-point-precision).

On GPUs and TPUs, matrix multiplications can be made faster still by lowering their internal precision below `float32`, using [`dq.set_matmul_precision()`][dynamiqs.set_matmul_precision]. Dynamiqs sets it to `'highest'` upon import, which keeps the full `float32` precision. Setting it to `'high'` uses `tensorfloat32` (or `bfloat16_3x`) when the hardware supports it:

```python
dq.set_matmul_precision('high')  # 'highest' by default
```

<!-- restore the default matmul precision
```python
dq.set_matmul_precision('highest')
```
-->

This reduces the number of significand bits carried through each matrix product. Adaptive methods compensate for the lost accuracy by taking smaller steps, so the speedup is not guaranteed. Always check the result against a `'highest'` run before relying on it.

Finally, `save_states=False` avoids storing and returning the state at every time in `tsave` when only expectation values are needed, which matters for large systems and many save times.

## Advanced: environment variables

These variables are read when JAX and Equinox are imported, so they must be set before importing Dynamiqs either from the shell, or with `os.environ` at the very top of the script.

### Turn off runtime error checking

Dynamiqs (and Diffrax) guard simulations with runtime checks (non-Hermitian Hamiltonians, unsorted save times, the maximum number of solver steps) implemented as [`equinox.error_if()`](https://docs.kidger.site/equinox/api/errors/#equinox.error_if). Each check adds a device-side conditional to the compiled program, which the XLA compiler cannot always optimize away. Setting `EQX_ON_ERROR=off` removes all of them:

```shell
EQX_ON_ERROR=off python simulation.py
```

Because of compiler differences, the gain depends on the backend, but it can amount to several tens of percent speedup. This disables the checks entirely, so an invalid input silently produces a wrong result rather than an error. Validate your inputs with a short run first, then turn the checks off for production.

### Simulate larger systems than the GPU memory

On NVIDIA GPUs, XLA can be allowed to spill its allocations out of the device into system memory (unified memory), which makes it possible to simulate systems that do not fit in the GPU memory:

```shell
TF_FORCE_UNIFIED_MEMORY=1 XLA_PYTHON_CLIENT_MEM_FRACTION=4 python simulation.py
```

`XLA_PYTHON_CLIENT_MEM_FRACTION` is the fraction of the GPU memory that XLA preallocates; any integer greater than 1 lets the allocation overflow into host memory. Accesses to the spilled pages go over the CPU-GPU interconnect, so this trades speed for capacity. It is especially useful on systems with a fast interconnect such as Grace Hopper, but works in principle on any NVIDIA GPU.

## Measure

Dynamiqs ships a benchmark suite that times representative simulations and compares two runs, for example before and after a change:

```shell
uv run task bench --tier physics --out before.json
uv run task bench --tier physics --out after.json
python -m benchmarks compare before.json after.json
```

See [benchmarks/README.md](https://github.com/dynamiqs/dynamiqs/blob/main/benchmarks/README.md) for the list of cases and for instructions on running them on a GPU. To time your own simulation, remember that the first call includes compilation, and that JAX dispatches work asynchronously (use `jax.block_until_ready` on the result before stopping the clock):

```python
import jax

simulate = jax.jit(lambda: dq.sesolve(H, psi0, tsave, progress_meter=False))
_ = jax.block_until_ready(simulate())  # compile
result = jax.block_until_ready(simulate())  # time this one
```
