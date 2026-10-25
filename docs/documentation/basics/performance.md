# Performance

Dynamiqs is built on [JAX](https://docs.jax.dev/), so a simulation is first compiled and then executed on the CPU, GPU or TPU. This page collects the settings that have the largest impact on run time, and explains how to measure their effect on your own simulations.

```python
import dynamiqs as dq
import jax.numpy as jnp
```

## Turn off the progress meter

By default, solvers display a [tqdm](https://github.com/tqdm/tqdm) progress meter. Updating it requires a callback from the device to the Python host every few solver steps, which forces a synchronization and prevents JAX from dispatching work asynchronously. This is invisible for long simulations, but it is a noticeable fraction of the run time for short ones, and it is more costly on GPU than on CPU.

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

## Choose a method

The integration method, passed with the `method` argument of each solver, often has the largest impact on run time. The best choice depends on the problem:

- [`Tsit5`][dynamiqs.method.Tsit5] (the default of [`dq.sesolve()`][dynamiqs.sesolve], [`dq.mesolve()`][dynamiqs.mesolve] and [`dq.floquet()`][dynamiqs.floquet]) is an explicit adaptive step size method of order 5, and a good default for most problems.
- [`Dopri5`][dynamiqs.method.Dopri5] and [`Dopri8`][dynamiqs.method.Dopri8] are the explicit adaptive Dormand-Prince methods of order 5 and 8. The higher order of `Dopri8` pays off when tight tolerances are required.
- [`Kvaerno3`][dynamiqs.method.Kvaerno3] and [`Kvaerno5`][dynamiqs.method.Kvaerno5] are implicit adaptive methods, suited to stiff problems, typically those with Hamiltonians or Liouvillians whose eigenvalues span several orders of magnitude (e.g. high-order polynomials of the bosonic operators in large dimensions). Each step is more expensive, but explicit methods would need many more steps on such problems.
- [`Expm`][dynamiqs.method.Expm] explicitly computes the propagator by matrix exponentiation. It only supports constant or piecewise constant operators, and is efficient for small systems ([`dq.sepropagator()`][dynamiqs.sepropagator] and [`dq.mepropagator()`][dynamiqs.mepropagator] use it by default for such operators, and `Tsit5` otherwise). It converts operators to dense, and is not recommended for large open systems due to the $\mathcal{O}(n^6)$ cost of exponentiating the Liouvillian.
- [`Rouchon1`][dynamiqs.method.Rouchon1], [`Rouchon2`][dynamiqs.method.Rouchon2] and [`Rouchon3`][dynamiqs.method.Rouchon3] are methods tailored to the Lindblad master equation, which write each step as a Kraus map. The density matrix stays positive (and with `normalize=True`, the default, trace-preserving) even at large step sizes.
- [`LowRank`][dynamiqs.method.LowRank] evolves a low-rank factor $m(t)$ of the density matrix, $\rho(t) = m(t)m(t)^\dagger$, instead of the full $n\times n$ matrix. It is suited to large open systems whose state remains close to low rank.
- [`JumpMonteCarlo`][dynamiqs.method.JumpMonteCarlo] and [`DiffusiveMonteCarlo`][dynamiqs.method.DiffusiveMonteCarlo] solve the master equation by averaging stochastic trajectories of state vectors, which can be cheaper than evolving the density matrix in large dimensions. For individual trajectories, the jump SSE solver [`dq.jssesolve()`][dynamiqs.jssesolve] uses [`Event`][dynamiqs.method.Event] or [`EulerJump`][dynamiqs.method.EulerJump], see the [Stochastic simulations](stochastic-simulations.md) tutorial.

Each solver lists the methods it supports in its API documentation. For adaptive methods, loosening the tolerances `rtol` and `atol` reduces the number of steps, at the cost of accuracy.

## Compute steady states with `LyapunovGMRES`

For large systems, compute steady states with [`LyapunovGMRES`][dynamiqs.method.LyapunovGMRES] rather than the default [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState]. The default integrates with the implicit method [`Kvaerno5`][dynamiqs.method.Kvaerno5], whose steps solve $n^2\times n^2$ linear systems with a dense solver, as does the default gradient [`Implicit`][dynamiqs.gradient.Implicit]. This costs $O(n^4)$ in memory and $O(n^6)$ in time: on a 40 GB GPU, it runs out of memory from $n=200$. `LyapunovGMRES` uses an iterative solver whose memory grows as $O(n^2)$, for the steady state and its gradient:

```python
a = dq.destroy(16)
H = a.dag() @ a + 0.5 * (a + a.dag())
method = dq.method.LyapunovGMRES()
result = dq.mesteadystate(H, [a], method=method)
```

In our benchmarks on an NVIDIA A100 GPU, in double precision and excluding compilation, `LyapunovGMRES` was 24 times faster than `DiffraxSteadyState` on a cat qubit of dimension 40, and 550 times faster at dimension 160. At dimension 160, it used 0.06 GB of memory instead of 31.5 GB.

`LyapunovGMRES` requires a unique steady state and fails on some systems (see the [Steady states](steady-states.md#choosing-a-method) tutorial). For these, `DiffraxSteadyState(me_method=dq.method.Tsit5())` avoids the dense linear systems, but takes many steps when the Liouvillian gap is small. Below a gap of about $10^{-5}$, both methods can be inaccurate in single precision: use double precision (see [Devices and precision](devices-and-precision.md)).

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

## Store the operators of several subsystems as composite qarrays

A [composite qarray](composite-qarrays.md) stores an operator as tensor products of small factors, one per subsystem. Whether it beats the full matrix depends on the factors. In our benchmarks of [`dq.mesolve()`][dynamiqs.mesolve] on an NVIDIA A100 GPU:

- **Dense factors**: with dense local Hamiltonians, couplings and jump operators on two subsystems, composite qarrays were 2.8 times faster than the full dense matrix at dimension 1024, and 10 times faster at dimension 4096. They compiled as fast or faster.
- **Sparse factors**: on coupled cavities and qubit chains, the full matrix in the `dia` layout was 3 to 4 times faster than composite qarrays, and compiled up to 10 times faster on qubit chains. Composite qarrays were still 6 to 7 times faster than the full dense matrix.

Qubit chains are slow in composite form because identity factors are stored and applied like any other: each term of a chain of $N$ qubits carries about $N$ of them. Peak memory was the same within 25% in all cases, since the state and the ODE stages dominate it. Some solvers and methods build the full matrix anyway (see the [Composite qarrays](composite-qarrays.md#simulating-with-composite-qarrays) tutorial).

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

See the [Devices and precision](devices-and-precision.md) tutorial for how to switch between single and double precision, and how to run simulations on a GPU.

Finally, when only expectation values are needed, pass `save_states=False` to the solver to avoid storing and returning the state at every time in `tsave`. This matters for large systems and many save times.

## Reuse compiled programs across runs

The first call of a simulation also traces and compiles it, which on a GPU can take longer than the simulation itself. Later calls with the same shapes reuse the compiled program, but only within the same Python process. To also skip the compilation in a new process, for example when running a script again or restarting a notebook kernel, turn on JAX's [persistent compilation cache](https://docs.jax.dev/en/latest/persistent_compilation_cache.html) before the first simulation:

<!-- skip: next -->
```python
import jax

jax.config.update('jax_compilation_cache_dir', '/path/to/cache')
```

or set the `JAX_COMPILATION_CACHE_DIR` environment variable before starting Python. A new process still traces the simulation, but loads its compiled program from this directory instead of compiling it, if an earlier process compiled the same program with the same jaxlib version, `XLA_FLAGS`, device and driver. Anything that changes the program, such as another Dynamiqs or JAX version, another precision or other solver options, compiles it again. Only programs that took more than one second to compile are stored (`jax_persistent_cache_min_compile_time_secs`), and the directory grows until you clear it. Do not use a directory that other users can write to: JAX runs the programs it loads from it.

For example, in our benchmarks, this shortened the first call of a `mesolve` of Hilbert space dimension 900 from 116 s to 27 s (17 s of which run the simulation), and that of a reverse-mode gradient through a `mesolve` of dimension 56 from 16 s to 7 s. On CPU, compilation is a smaller part of the first call, and the gain is smaller.

## Advanced: environment variables

These variables are read when JAX and Equinox are imported, so they must be set before importing Dynamiqs, either from the shell or with `os.environ` at the very top of the script.

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

Dynamiqs ships a benchmark suite that times representative simulations and compares two runs, for example before and after a change. From a clone of the repository, run:

```shell
uv run task bench --tier physics --out before.json
uv run task bench --tier physics --out after.json
uv run python -m benchmarks compare before.json after.json
```

See [benchmarks/README.md](https://github.com/dynamiqs/dynamiqs/blob/main/benchmarks/README.md) for the list of cases and for instructions on running them on a GPU. To time your own simulation, remember that the first call includes compilation, and that JAX dispatches work asynchronously (use `jax.block_until_ready` on the result before stopping the clock):

```python
import jax

simulate = jax.jit(lambda: dq.sesolve(H, psi0, tsave, progress_meter=False))
_ = jax.block_until_ready(simulate())  # compile
result = jax.block_until_ready(simulate())  # time this one
```
