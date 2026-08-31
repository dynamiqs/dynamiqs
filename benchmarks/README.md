# Benchmarks

Lightweight timing benchmarks for dynamiqs. They measure **performance only** — compile time, wall-clock run time and solver step counts. Correctness is the job of the test suite in `tests/`.

Cases come in two tiers:

- **`physics`** — representative user workloads, one per solver API. Run these to catch regressions with `compare` (see below).
- **`features`** — families of cases that differ in exactly one knob (layout, method, option, gradient), registered consecutively so that two adjacent rows of a single run answer a question on their own: *is SparseDIA faster than dense here? does `vectorized=True` pay off at this `n`?*

## Quickstart

```shell
uv run task bench                     # both tiers (~2 minutes on a laptop CPU)
uv run task bench --tier physics      # workload regressions only (~1 minute)
uv run task bench --tier features     # feature comparisons only (~1 minute)
uv run task bench --quick             # tiny sizes, a few seconds (sanity check)
uv run task bench --filter cavity     # only cases whose key contains "cavity"
uv run task bench --out results.json  # also write results to a JSON file
```

Options: `--tier {physics,features}` (default: both), `--filter S` (substring match on the case key, e.g. `--filter n=128` or `--filter feat_layout`), `--quick` (tiny problem sizes), `--repeats N` (timed runs per case, default 5), `--out FILE`.

Output columns:

- `compile (s)` — ahead-of-time compilation time (tracing + lowering + compilation, no execution),
- `median (s)` — median wall-clock time over the timed runs, after compilation,
- `nsteps` — number of solver steps, maximum over batched simulations (the slowest batch element governs wall-clock time),
- `nrej` — number of rejected steps (adaptive step-size methods only).

Two caveats when reading a table:

- The four stochastic cases cannot be wrapped in an outer `jax.jit` (their `tsave` must stay concrete), so they compile themselves on their first call and their `compile (s)` includes one execution.
- Rows under a millisecond are dominated by timer noise at the default `--repeats 5`. Raise `--repeats` before drawing a conclusion from one of them.

## The `physics` tier

| case | what it stands for |
| --- | --- |
| `sesolve_transmon` | driven transmon, smooth analytic (DRAG-like) pulse — the small-`n` regime where per-step overhead, not matrix products, dominates |
| `sesolve_spin_chain` | transverse-field Ising chain, Hamiltonian assembled from tensor products of Paulis — many-body ket dynamics |
| `sesolve_pwc` | adaptive stepper crossing piecewise-constant discontinuities; `nrej` reports what they cost |
| `mesolve_cavity` | canonical large open bosonic system, constant operators — the `n²` density-matrix scaling |
| `mesolve_cat` | cat qubit stabilized by two-photon dissipation, batched over the drive amplitude — a parameter scan |
| `mesolve_cross_resonance` | two coupled transmons with decay and dephasing, batched over the drive — a gate-calibration sweep |
| `mesolve_grad` | reverse-mode gradient of a scalar loss through `mesolve` — the pulse-optimization workload |
| `sepropagator_expm` | propagator of a constant generator by explicit matrix exponentiation |
| `mepropagator_expm` | same for the Liouvillian, kept small because of the `O(n⁶)` scaling |
| `floquet_driven_kerr` | one-period propagator, eigendecomposition, forward propagation |
| `jssesolve_cavity`, `jsmesolve_cavity` | jump unravelings (SSE and SME), fixed-step and key-batched over trajectories |
| `dssesolve_cavity`, `dsmesolve_cavity` | diffusive unravelings (SSE and SME), likewise |

## The `features` tier

| family | variants | what the ratio between the rows means |
| --- | --- | --- |
| `feat_layout_mesolve`, `feat_layout_sesolve` | `dia` vs `dense` | the SparseDIA speedup, on banded bosonic ladder operators and on tensor products of Paulis |
| `feat_method_rouchon` | `Tsit5` vs `Rouchon2` vs `Rouchon3` | per-step cost and step count of the Lindblad-tailored schemes against a generic explicit RK |
| `feat_method_expm_sesolve`, `feat_method_expm_mesolve` | `Tsit5` vs `Expm` | what one matrix exponential per save interval costs when all you want is the state — price it against `sepropagator_expm`, where the same cost buys a full propagator |
| `feat_vectorized` | `vectorized=False` vs `True`, at three `n` | how fast the vectorized Liouvillian's advantage shrinks with `n` (the explicit `n²×n²` superoperator sets a memory ceiling not far above the largest point) |
| `feat_assume_hermitian` | `True` vs `False` | how much of the run time the halved vector-field matmuls actually are |
| `feat_save_states` | `True` vs `False` | cost of materializing and saving states versus expectation values only. Memory- and IO-bound, so it barely registers on CPU: this case earns its place on GPU runs |
| `feat_gradient` | `BackwardCheckpointed` vs `Forward`, at 1 and 20 parameters | where forward-mode stops winning and reverse-mode takes over |
| `feat_lowrank` | full `Tsit5` vs `LowRank` at two ranks | what rank truncation buys against evolving the full density matrix |
| `feat_batch` | `batch` = 1, 16, 256 | the marginal cost of a batch element — near-linear on CPU, near-flat on GPU until saturation |

**These families compare methods at equal *tolerance*, not at equal accuracy.** This suite times only; it has no reference solutions. A method that takes fewer steps is not thereby more accurate.

## Comparing two runs (A/B)

To measure the performance impact of a change:

```shell
git checkout main
uv run task bench --tier physics --out before.json
git checkout my-branch
uv run task bench --tier physics --out after.json
python -m benchmarks compare before.json after.json
```

The comparison table aligns cases by name and parameters and reports the change in median run time (positive = slower). The JSON files record the device, precision, package versions and git SHA of each run; `compare` warns if device or precision differ.

## Benchmarking on GPU

Install a CUDA-enabled JAX build (e.g. `uv pip install -U "jax[cuda12]"`), then run as usual — JAX picks up the GPU automatically and the device is recorded in the results. For stable numbers, run on an idle machine; timings are synchronized with `jax.block_until_ready()`, so asynchronous dispatch does not skew them. Only compare runs made on the same device and precision.

## Adding a case

Physics systems live in `systems.py` as builders that return the ingredients of a simulation (Hamiltonian, jump operators, initial state, save times) without calling a solver, so one system can serve several cases. They take an explicit `layout` argument rather than switching the global default, because `feat_layout` needs both layouts in one process.

In `cases.py`, write a builder returning a plain zero-argument closure (the runner jits it, times its ahead-of-time compilation, then times its execution), add its sizes to `_FULL_GRID` and `_QUICK_GRID`, and register it in one of the tier functions with a comment stating the question it answers. The CI smoke test (`tests/test_benchmarks.py`) picks up new cases automatically; run `uv run task durations` afterwards, since the new case keys change the CI test sharding.
