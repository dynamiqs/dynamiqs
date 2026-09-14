# Dynamiqs memory leak profiling 

## Aim of this document

This report answers one question: **where does the memory go when a dynamiqs simulation runs, and is any of it a leak?** In December 2025, CNOT gate tomography simulations run with Baccachat hit OOM failures, and the memory actually consumed was far above what back-of-the-envelope arithmetic predicted. the document has 2 aims: 

- Establish a **robust way to profile memory use in the JAX ecosystem on GPU**.
- Use that method to build a **correct predictive model of dynamiqs' memory use**.

<aside>

## TL;DR - conclusion

- XLA applies **`XLA_PYTHON_CLIENT_MEM_FRACTION=.XX`** the memory fraction to *free* device memory, not to total physical memory
- Default on Higgs is **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** this seems to prevent correct preallocation of memory and fixes it at the standard limit of 75% of free GPU memory. ether use one or the other.

4 formulas for peak measured memory with respect to simuation parameters :

K = 48 MiB, R = size of density matrix,  S = stages of the method, n_ts = saved states

$$
\footnotesize \text{peak} = \max\Big(K,\ (7.07 + S)\,R\Big) \qquad \text{save\_states : FALSE, gradient : FALSE} \\

\text{peak} = \max\Big(K,\ 3R(n_{ts}+1)\Big) \qquad \text{save\_states : TRUE, gradient FALSE} \\

\text{peak} = \max\Big(K,\ (9.07 + S)\,R\Big) \qquad \text{save\_states FALSE, gradient DIRECT} \\

\text{peak} = \max\Big(K,\ 3R\,n_{ts} + (7.07 + S)\,R\Big) \qquad \text{save\_states TRUE, gradient DIRECT}
$$

All four land at R² = 1.0 on the data that has at least 56 datapoints. 

**The Biggest red flag in those models is the factor 3 in front of the number of saved states. present in `diffrax` and `dynamiqs`**

</aside>

## Method

The investigation was run incrementally on one whole L40S per job, in `float64`/`complex128`. Sizes of problems are chosen so that a single state copy is roughly 10⁵ times the allocator's 256-byte granularity, which makes one redundant copy unmistakable rather than a rounding artefact.

Five ground rules apply to every run and are what make the numbers comparable. 

- No `jit` is added by any script, so per-operation allocation stays visible; where a JAX library function carries its own internal `@jit` (`jnp.kron` does) the stats are read after each individual call, which preserves the same visibility
- Every allocation is wrapped in `try`/`except`, because OOM is the expected outcome of a sweep and not a failure.
- Every allocation is followed by `.block_until_ready()` before stats are read, so asynchronous dispatch cannot report memory that has not been allocated yet
- Between configurations the state is cleaned with `del`, `gc.collect()` and `jax.clear_caches()`, then re-read to confirm a return to baseline
- Finally, **each configuration runs in its own process**, because `peak_bytes_in_use` is a high-water mark that never resets and a shared process would let the first configuration mask every later one.

The investigation is then split into four parts, each isolating one variable and answering one question.

| Part | Isolates | Question it answers |
| --- | --- | --- |
| 1 | the GPU and the environment | What is the hard memory ceiling, and can the reported one be trusted? |
| 2 | the instrument | Does `memory_stats()` reproduce hand arithmetic, and does it predict OOM? |
| 3 | diffrax alone, no physics | Does solver memory scale the way the Runge–Kutta algorithm says, and does anything accumulate across calls? |
| 4 | Lindblad and dynamiqs | What does the physics add over a bare ODE, what does dynamiqs add over the physics, and what does saving a trajectory cost? |

## Part 1 — What does JAX give us by default, and can we get more?

**The question in context.** Every later prediction is a fraction of a ceiling, so the ceiling has to be measured rather than assumed, and the natural first question is what a script gets with no configuration at all. The card itself is not in question, `nvidia-smi` reports 46068 MiB total and 0 3MiB resident, matching the brief exactly, so any gap between "physical memory" and "memory JAX will use" has to come from XLA's defaults, and is worth pinning down two independent ways: reading the value JAX itself reports, and bisecting on the size of a single buffer to see what the allocator actually hands over.

python

```python
def try_alloc(nelem):    try:        buf = jnp.zeros(nelem, dtype=jnp.float64); buf.block_until_ready()        return dev.memory_stats()["bytes_in_use"] >= nelem * 8    except Exception:                    # OOM is the expected outcome        return False    finally:        del buf; gc.collect(); jax.clear_caches()# JAX's own reported ceiling, read the second way:dev.memory_stats()["bytes_limit"]
```

**Results.**

| Quantity | GiB (2³⁰) | bytes | Source |
| --- | --- | --- | --- |
| Physical total | 44.9883 | 48 305 799 168 | `nvidia-smi` |
| **Default `bytes_limit` (no config set)** | **33.2969** | **35 752 247 296** | `memory_stats()`, job 69371 |
| Largest single allocation that succeeds, default config | 33.2969 | 35 752 247 296 | bisection, job 69371 |
| `bytes_limit` after raising `XLA_PYTHON_CLIENT_MEM_FRACTION=.95` | 42.1758 | 45 285 900 288 | job 69371 |
| Largest single allocation that succeeds, `.95` + preallocation on | 42.1757 | 45 285 762 064 | job 69371 |
| Same `.95` fraction, preallocation left at its login-shell default (`false`) | 31.9995 | 34 359 198 056 | job 69371 |

> **Conclusion 1.** **By default JAX allows 75 % of the GPU**. `memory_stats()["bytes_limit"]` reports 35 752 247 296 B with no environment variables set, and bisecting the largest allocatable `jnp.zeros` buffer lands on the same number. Both methods agree exactly. This is 0.740123 of physical memory, which looks like 74 %; Conclusion 3 shows the true fraction is 0.75, applied to a smaller base.
> 

> **Conclusion 2.** **The ceiling can be raised, but only with preallocation forced on.** `XLA_PYTHON_CLIENT_MEM_FRACTION` controls it (default 0.75); setting it to `.95` raises `bytes_limit` to 42.1758 GiB, but only if `XLA_PYTHON_CLIENT_PREALLOCATE=true` is also set explicitly. This cluster's login shell exports `PREALLOCATE=false`, and SLURM's `export=ALL` passes it into every job. Under that default, `bytes_limit` still reports 42.1758 GiB, but the grow on demand allocator only delivers 31.9995 GiB, exactly 2³⁵ bytes, a 10.18 GiB shortfall.
> 

> **Conclusion 3.** **The fraction applies to free memory, not total memory**. Solving `free = bytes_limit / fraction` for both configurations gives 47 669 663 061 B and 47 669 368 724 B, agreement to 0.28 MiB, which is free memory at backend init, not the 48 305 799 168 B `nvidia-smi` reports as total. The difference, 636 430 443 B ≈ 607 MiB, is a CUDA context reserved before JAX sees the card. Dividing the default `bytes_limit` by physical memory gives 0.7401; dividing by free memory gives exactly 0.75. The 607 MiB figure is specific to this driver and card, and should be remeasured elsewhere rather than assumed.
> 

## Part 2 — Is `memory_stats()` a faithful and precise instrument?

**The question in context.** `device.memory_stats()` is the tool the whole investigation depends on, so it must be validated before it is trusted. Two distinct properties matter and they turn out to be carried by two different metrics: *fidelity* (does a reported number match hand arithmetic?) and *precision as an OOM proxy* (does the reported number predict where the run actually dies?). The test case is a geometric Kronecker ladder with a fixed 2×2 right operand, for which `predicted_bytes(n,k) = 8·n²·4^k` is exact, **swept over n = 2…40** with stats read after every single call.

```python
A_step = jax.random.uniform((2,2))
for k in range(1, K_MAX + 1):
    b = jnp.kron(b, A_step)      # side ×2, elements ×4
    b.block_until_ready()
    stats = dev.memory_stats()   # read after EVERY individual kron
```

Three hypotheses were pre-registered for the peak: 

- H1 = 1.00× resident (only the result is ever live)
- H2 = 1.25× (live memory is output + input which is 1/4 the shape fo the output).
- H3 = 2.25× (as H2 plus a materialising copy in `kron`'s trailing reshape).

**Results** — 520 rows, 481 successful, 39 OOM, 0 unexpected errors.

!image.png

!image.png

| Measurement | Result |
| --- | --- |
| `bytes_in_use` vs prediction | **`predicted + 256 B` exactly.** this accounts for the A_step additional cost memory footprint**. so this is bite exact** |
| `peak_bytes_in_use` vs prediction | `1.250000 × predicted + 256 B`, same exactness → **H2 confirmed, H1 and H3 rejected** |
| OOM point predicted by resident size | 27/39 correct |
| OOM point predicted by peak (H2) | **34/39 correct** |
| Largest H2 requirement that survived | 37.539 GiB = 89.0 % of `bytes_limit` |
| Smallest H2 requirement that failed | 40.000 GiB = 94.8 % of `bytes_limit` |

> **Conclusion 4.** **`peak_bytes_in_use` is byte exact** once the negligible 2×2 `A_step` matrix is factored in. The +256 B offset is that fixed operand plus the PRNG key, rounded to allocator granularity, not a hidden copy.
> 

> **Conclusion 5.** **`peak_bytes_in_use` is a good but not perfect proxy for OOM**, because of **fragmentation tied to allocation history**. Replacing `jnp.kron` with plain `jnp.zeros` reproduces the same OOM point exactly, 5/5, so `kron` itself adds nothing. The cost is the grow while holding pattern: the old buffer stays in place while the new one is requested, so the new allocation has to fit around it rather than into a clean pool, and this prevents the pool from ever reaching 100 % full. **A two-buffer control confirms this: allocated fresh, two buffers reach 99.9994 % of `bytes_limit.`** This is the pattern every ODE solver stage uses. Budget rule adopted for the rest of the work: assume less then 42.18 GiB are available.
> 

## Part 3 — Does diffrax scale the way the algorithm says?

**The question in context.** Bare diffrax, no physics: `dY/dt = M @ Y` with `Y, M` of shape (4000, 4000), `float64`, so one state copy is **122.07 MiB** and a single redundant copy per stage would be impossible to miss. Three things are tested separately :

- Growth with step count,
- Growth with stage count,
- Retention across repeated calls.

```python
# M is the (N, N) operator defining the ODE dY/dt = M @ Y
# per-stage cost: hold n_steps fixed at 10, vary only the solver's stage count
for solver in [Euler, Ralston, Bosh3, Tsit5, Dopri8]:
    diffeqsolve(term, solver, ..., saveat=SaveAt(t1=True))

# retention: same solve six times, FRESH M and Y0 each time
# two vector fields tested, identical except for how M reaches them

def vf_closure(t, y, args):     # M captured from the enclosing scope
    return M @ y

def vf_args(t, y, args):        # M passed in explicitly as a traced argument
    return args @ y

for i in range(6):
    M, Y0 = fresh_arrays()
    diffeqsolve(term_closure, solver, ..., y0=Y0)              # vf_closure, args=None
    diffeqsolve(term_args,    solver, ..., y0=Y0, args=M)      # vf_args, args=M
```

**Results.**

`S = 8 · N²     (float64 = 8 bytes/element)` is the size of one state.

!image.png

| Hypothesis | Prediction | Measured | Verdict |
| --- | --- | --- | --- |
| `SaveAt(t1=True)` flat in step count | flat | byte-identical at 10 vs 100 steps | no growth |
| `SaveAt(steps=True)` linear in steps | `n_steps × S` | `max_steps × S`, ratio 1.0000 | sizes on **`max_steps`**, not steps taken |
| peak for a stored trajectory | ≈ 1× trajectory | `2.000·S + 3.000·S·max_steps`, residuals ±16 B on 39 GB | **two extra full copies** of the max steps data  |
| per-stage cost via `diffeqsolve` | `b = 1 × S` | `b = 1.000000 × S`, residuals exactly 0.0 B | **no redundant copies** |
| retention across calls, operator captured | — | **+1 full operator copy per solve, unbounded** | **leak** |
| retention across calls, operator via `args` | — | ~2.5 kB total, flat | clean |
| peak is affine in stages per step | `P = a + b x S` | `P = 5.393 × S + 1.0 × S · n_stages` |  |

**The overall diffrax memory law.** Combining the per-stage fit and the trajectory-storage fit gives the peak as a max of two regimes, not a sum, since the two allocations don't both peak at once:

```
peak(SaveAt(steps=True))= max( (5.393 + 1.0·n_stages) · S ,  (2.0 + 3.0·max_steps) · S )

peak(SaveAt(steps=False))= S + (5.393 + 1.0·n_stages) · S
```

*Do we believe this?* Both halves are affine fits with essentially zero residual, 0.0 B on the stage term, ±16 B on 39 GB for the trajectory term, so each half is trustworthy on its own. The `max` rather than `sum` is inferred rather than directly measured: if the two costs added, the trajectory term's intercept (2.000·S) would have to include the solver's own working set (5.393·S for a fixed solver, more for higher-stage ones), and it doesn't, the intercept is far too small.

> **Conclusion 6.** The trajectory term dominates fast, and that is the real problem. `max_steps` runs into the hundreds while `n_stages` tops out at 14, so **`3.000·S·max_steps`** overtakes the stage term almost immediately. **The factor of 3 is itself the red flag:** a stored trajectory costs three times its resident size, and that overshoot grows with `max_steps`, not fixed. This sets the practical ceiling on trajectory length long before solver cost matters at all.
> 

> **Conclusion 7.** `SaveAt(steps=True)` sizes on the static bound, not on what runs. Storage scales with `max_steps`, the pre-allocated slot count, not with steps actually taken. At the default `max_steps=4096`, N=4000 requests 488 GiB, an instant OOM regardless of integration length. Set `max_steps` tightly.
> 

> **Conclusion 8.** One real leak exists, and it is narrow and does not apply to dynamiqs. A vector field that captures its operator via a Python closure leaks one full copy per `diffeqsolve` call, unbounded, until `jax.clear_caches()` runs. `diffeqsolve`'s cache is keyed on the term, so each new closure adds a permanent entry; a plain `jax.jit`'s cache dies with the function and stays flat. Passing the operator through `args` removes the leak. It only fires when the vector field is rebuilt per call, a hand written sweep or gradient loop. Part 4 confirms dynamiqs passes operators as `args` style pytree leaves and never triggers it.
> 

## Part 4 — What do Lindblad and dynamiqs each add?

### **Part 4.A — Diffrax linear ODE → diffrax Lindblad → dynamiqs → dynamiqs sparse**

**The question in context.** One question, one variable at a time: does the Lindblad dissipator cost what the equation says it should, and does dynamiqs cost more or less than writing that equation by hand, dense or sparse? Four points sit on the same axis, all measured in peak memory, all at the same size, same solver, same step budget, so the only thing that changes between them is the code path.

**Experiment.** N = 2000, so one density matrix is R = 16 N² = 64,000,000 B, about **61 MiB**, and `complex128` throughout. Two jump operators in every Lindblad row. The solver is T`sit5, adaptive, rtol = atol = 1e6`. Trajectory storage is off in every row, so the number of steps the adaptive solver actually takes does not matter, Part 3 already established that peak memory is flat in step count once the trajectory is not stored, only the final state is kept.

- Row 1, a bare linear ODE, dY/dt = M @ Y, no physics at all.
- Row 2, a hand written dense Lindblad generator, caching Lk, Lk†, and Lk†Lk for each jump operator.
- Row 3, dynamiqs on the exact same H and Lk as row 2, dense operators..
- Row 4, the same cat qubit forced dense, added as a control so row 4 differs from row 5 in layout and nothing else.
- Row 5, dynamiqs on a cat qubit with native sparse DIA operators

```python
# row 2, hand written, caches Lk, Lk dagger, Lk dagger Lk -> predicted 7 R of operators
drho = -1j*(H@rho - rho@H) + sum(L@rho@Ld - 0.5*(LdL@rho + rho@LdL) for ...)

# row 3, dynamiqs, same H and Lk, no caching
dq.mesolve(H, Ls, rho0, ts, options=Options(save_states=False))
```

**Results.**

R here is for Rho and stands for the size of the density matrix computed by hand. 

!image.png

| Row | Operator storage | Measured peak |
| --- | --- | --- |
| 1, bare ODE | 1.000 R | 12.628 R |
| 2, hand written Lindblad | 7.000 R | 24.000 R |
| 3, dynamiqs dense, same physics as row 2 | 3.000 R | 19.066 R |
| 4, dynamiqs dense, layout control for row 4 | 3.000 R | 19.066 R |
| 5, dynamiqs sparse DIA | 0.00200 R (128 kB, predicted 128 kB) | 14.065 R |

> **Conclusion 9.** **The physics costs what the equation says**, and **Dynamiqs costs less than writing that equation by hand**. Row 1 and row 2 land within 2.6% of an expression derived purely from the algebra, with operator storage matching exactly, so nothing is unaccounted for.
> 

### **Part 4.B — Does the memory law hold everywhere, and what is it exactly?**

**The question in context.** Every coefficient measured so far rests on one N and one solver. This part runs a **full grid search over problem size, solver, save length, and gradient choice at once**, to see whether the closed form from Part 4.A still holds, and to fit its final coefficients with confidence rather than from a single point.

**Grid.** 960 configurations, one process each, `peak_bytes_in_use` read from every one.

| Axis | Search values |
| --- | --- |
| save_states | False, True |
| N | 64, 128, 256, 512, 1024, 2048 |
| solver, stage count S | rk2 (2), rk3 (4), rk5 (7), rk8 (14), all adaptive, through dynamiqs mesolve |
| number of saved points, ntsave | 10, 120, 230, 340, 450, 560, 670, 780, 890, 1000 |
| gradient | none, Direct |

888 of the 960 configurations completed, the other 72 hit the GPU ceiling at the largest N and ntsave combinations, which is itself part of the answer: the model predicts exactly where that happens.

Two things needed correcting before the coefficients could be trusted. 

- First, a fixed floor, K = 50,404,608 B, about 48 MiB, sits underneath every configuration and is independent of everything else, the single N tested in Part 4.A never crossed it.
- Second, the gradient = Direct penalty is a constant tied to solver stage count, not a cost per saved state as first guessed from a single measurement.

**Results, the four formulas.** S is the number of solver stages, ntsave the number of saved time points, K the fixed floor above. Each was fit on its own filtered slice of the grid, filtering removes points still sitting on the K floor.

$$
\footnotesize \text{peak} = \max\Big(K,\ (7.07 + S)\,R\Big) \qquad \text{save\_states false, gradient none} \qquad (n=60) \\

\text{peak} = \max\Big(K,\ 3R(n_{ts}+1)\Big) \qquad \text{save\_states true, gradient none} \qquad (n=120) \\

\text{peak} = \max\Big(K,\ (9.07 + S)\,R\Big) \qquad \text{save\_states false, gradient Direct} \qquad (n=56) \\

\text{peak} = \max\Big(K,\ 3R\,n_{ts} + (7.07 + S)\,R\Big) \qquad \text{save\_states true, gradient Direct} \qquad (n=123) \\
$$

All four land at R² = 1.0000000 on their filtered slice of the grid.

> **Conclusion 10.** The Part 4.A story survives a grid spanning 32 times the range in N and 7 times the range in stage count, with two corrections. There is a fixed floor, about 48 MiB, that a small problem never clears no matter what else is asked of it, so memory is always the larger of that floor and the formula above, never their sum. And the gradient = Direct penalty is a fixed number of extra solver stage buffers, not a cost that grows with how many states are saved, the earlier single point measurement mistook a coincidence of matching numbers for a real dependency. Once both are accounted for, the four formulas above predict peak memory across the whole grid with nothing left to explain.
> 

## Part 5 — What causes the factor 3 in front of `ntsave`?

**The question in context.** The factor 3 is the single largest term in the model and the one that sets how long a trajectory can be. It appears identically in raw diffrax and in dynamiqs, it is exact to four significant figures, and it is independent of N, of solver, of stage count and of `max_steps`. Parts 3 and 4 measured it from the outside. This part says what the two extra copies actually are.

```python
# what XLA says it will allocate, before anything runs
compiled = jax.jit(f).lower(rho0).compile()
compiled.memory_analysis()        # output_size_in_bytes, temp_size_in_bytes

# and which HLO instructions own those bytes
XLA_FLAGS="--xla_dump_to=DIR"     # then read *-buffer-assignment.txt
```

**The 3 splits into 1 returned and 2 scratch.** Sweeping `ntsave` at N = 64:

| Quantity | Fit against `ntsave` |
| --- | --- |
| `output_size_in_bytes` | 205 B + **1.0001 R** per saved state |
| `temp_size_in_bytes` | 137 013 B + **2.0001 R** per saved state |
| total | 137 218 B + **3.0002 R** per saved state |

One copy is the trajectory you asked for. The other two are scratch, and the buffer assignment for the `ntsave = 40` module names the instructions that own every trajectory sized buffer in the program. There are exactly three:

```
allocation 0  (2 621 440 B, maybe-live-out)       <- the returned trajectory
    copy.77 , pure_callback.61{4} , cond.197{0}
allocation 70 (5 380 224 B, preallocated-temp)    <- 2 × trajectory + 137 kB
    offset 0         : pure_callback.50{4}
    offset 2 621 440 : while.288{10} , while.295{3} , broadcast and update fusions
```

with this metadata attached to the two `pure_callback` instructions:

```
op_name=".../jit(diffeqsolve)/jit(branched_error_if_impl)/cond/branch_1_fun/pure_callback"
```

**The mechanism.** `diffeqsolve` defaults to `throw=True`, and its last act is `result.error_if(sol, ...)`, which hands the **entire** `Solution`, trajectory included, to equinox's error handler. The default handler, `EQX_ON_ERROR=raise`, is two chained `jax.pure_callback` calls. A `pure_callback` lowers to an XLA custom call, and a custom call's operand and its result cannot share a buffer. So each callback materialises a fresh copy of everything passed through it. Two callbacks, two copies of the trajectory.

The trajectory is passed at all because the check has to be data dependent. `error_if` returns its argument unchanged, and it is that returned value the rest of the program consumes; if the trajectory did not flow through the callback, XLA would delete the check as dead code and no error would ever be raised.

**The save path itself is innocent**, which is worth stating because it is the obvious suspect. In the allocation above, the while loop carry (`while.288{10}`, `while.295{3}`) shares a single temp slot with the broadcast and update fusions that write into it, so equinox's buffer machinery is doing exactly its job and the trajectory is filled in place. The `t0 == t1` conditional (`cond.197{0}`) lives inside the output allocation and costs nothing extra. Neither the preallocation, nor the loop, nor the saving is responsible.

**Causal test.** Compiling the identical solve under each value of `EQX_ON_ERROR`, one process per setting since equinox reads it at import:

| `EQX_ON_ERROR` | output | temp | total |
| --- | --- | --- | --- |
| `raise` (default) | 1.0001 R | **2.0001 R** | **3.0002 R** |
| `nan` | 1.0001 R | **1.0000 R** | **2.0001 R** |
| `off` | 1.0001 R | **0.0017 R** | **1.0019 R** |

Exactly linear in the number of callbacks emitted. Two, then one, then none.

> **Conclusion 11.** **The factor 3 is not a save path cost, it is the error check.** Storing the trajectory costs 1 R per saved state, exactly as the arithmetic says. The other 2 R are two copies made by equinox's error handler, which diffrax feeds the whole solution so that `throw=True` cannot be optimised away. It scales with `ntsave` only because the trajectory happens to be the largest thing in the object being checked.
> 

**What it is worth on the real allocator**, N = 2000, `ntsave` = 101, measured with `peak_bytes_in_use` like everything else in this report:

| Setting | peak | slope | trace(ρ_final) | checksum |
| --- | --- | --- | --- | --- |
| default | 18.24 GiB | 3.0000 R | 1.000000000000 | 1.3492017380e+02 |
| `nan` | 12.16 GiB | 1.9901 R | 1.000000000000 | 1.3492017380e+02 |
| `off` | 12.49 GiB | 2.0452 R | 1.000000000000 | 1.3492017380e+02 |

Identical answers to every digit. The capacity gain is demonstrated rather than extrapolated: at N = 2000, `ntsave = 303` runs out of memory on the default and completes in 36.57 GiB with the callbacks disabled. Note also the gap between the two instruments. The compiler says 1 R is reachable, the allocator delivers 2 R, because the compile only measurement puts the whole call under a single `jit` while a direct `dq.mesolve` leaves one further copy live in its own output path. A third copy is still on the table.

**The cost of turning it off.** With `nan` or `off` a genuine failure stops raising. On a solve forced to fail with `max_steps=1`, the default raises, `nan` returns NaN states silently and `off` returns inf states silently. This is worse in dynamiqs than in bare diffrax, because dynamiqs keeps only `solution.stats` and discards `solution.result`, so the structured diffrax error code never reaches the caller. Non finite states are then the only remaining signal.

> **Conclusion 12.** `EQX_ON_ERROR=nan` takes the slope from 3 R to 2 R today, at the price of converting a raised exception into a silent NaN. It is usable, but only with an explicit `np.isfinite(res.states).all()` after every solve.
> 

**Bug or feature.** It is a feature whose cost was never meant to scale this way. Routing the payload through the callback is a deliberate correctness device: it is the only thing that keeps the check alive through dead code elimination, and without it `throw=True` would silently stop working. What is accidental is that the payload is the whole `Solution`. The check needs one scalar, the result code, and pays for the largest array in the program instead.

That makes it fixable in two independent places, neither of which requires giving up the error:

- In diffrax or equinox, pass a sentinel rather than the payload. A scalar, or a single slice of the trajectory, keeps the check data dependent and drops 2 R.
- In dynamiqs, keep `solution.result` in the `Result` object and call `diffeqsolve` with `throw=False`. The failure can then be raised in Python after the solve, which is both cheaper and more informative than the current behaviour, since dynamiqs currently discards that field.

> **Conclusion 13.** The largest single term in the dynamiqs memory model is not physics, not the solver and not the trajectory. It is two defensive copies made by an error check, and both are avoidable without weakening the check. Removing them multiplies the longest storable trajectory by up to three.
> 

<aside>

## TL;DR - conclusion

- XLA applies **`XLA_PYTHON_CLIENT_MEM_FRACTION=.XX`** the memory fraction to *free* device memory, not to total physical memory
- Default on Higgs is **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** this seems to prevent correct preallocation of memory and fixes it at the standard limit of 75% of free GPU memory. ether use one or the other.

4 formulas for peak measured memory with respect to simuation parameters :

K = 48 MiB, R = size of density matrix,  S = stages of the method, n_ts = saved states

$$
\footnotesize \text{peak} = \max\Big(K,\ (7.07 + S)\,R\Big) \qquad \text{save\_states : FALSE, gradient : FALSE} \\

\text{peak} = \max\Big(K,\ 3R(n_{ts}+1)\Big) \qquad \text{save\_states : TRUE, gradient FALSE} \\

\text{peak} = \max\Big(K,\ (9.07 + S)\,R\Big) \qquad \text{save\_states FALSE, gradient DIRECT} \\

\text{peak} = \max\Big(K,\ 3R\,n_{ts} + (7.07 + S)\,R\Big) \qquad \text{save\_states TRUE, gradient DIRECT}
$$

All four land at R² = 1.0 on the data that has at least 56 datapoints. 

The Biggest red flag in those models is the factor 3 in front of the number of saved states. present in `diffrax` and `dynamiqs`

</aside>