# Benchmarks

Lightweight timing benchmarks for dynamiqs, on a small set of cases representative of typical user workloads. They measure **performance only** — compile time, wall-clock run time and solver step counts. Correctness is the job of the test suite in `tests/`.

## Quickstart

```shell
uv run task bench                     # full suite (~2 minutes on a laptop CPU)
uv run task bench --quick             # tiny sizes, a few seconds (sanity check)
uv run task bench --filter cavity     # only cases whose name contains "cavity"
uv run task bench --out results.json  # also write results to a JSON file
```

Options: `--filter S` (substring match on the case key, e.g. `--filter n=128`), `--quick` (tiny problem sizes), `--repeats N` (timed runs per case, default 5), `--out FILE`.

Output columns:

- `compile (s)` — duration of the first call: compilation plus one run (an upper bound on compilation time),
- `median (s)` — median wall-clock time over the timed runs, after compilation,
- `nsteps` — number of solver steps, maximum over batched simulations (the slowest batch element governs wall-clock time),
- `nrej` — number of rejected steps (adaptive step-size methods only).

## Comparing two runs (A/B)

To measure the performance impact of a change:

```shell
git checkout main
uv run task bench --out before.json
git checkout my-branch
uv run task bench --out after.json
python -m benchmarks compare before.json after.json
```

The comparison table aligns cases by name and parameters and reports the change in median run time (positive = slower). The JSON files record the device, precision, package versions and git SHA of each run; `compare` warns if device or precision differ.

## Benchmarking on GPU

Install a CUDA-enabled JAX build (e.g. `uv pip install -U "jax[cuda12]"`), then run as usual — JAX picks up the GPU automatically and the device is recorded in the results. For stable numbers, run on an idle machine; timings are synchronized with `jax.block_until_ready()`, so asynchronous dispatch does not skew them. Only compare runs made on the same device and precision.

## Adding a case

In `cases.py`, write a builder returning a zero-argument jitted closure (use `eqx.filter_jit` if it returns a solver `Result`) and register it in `benchmark_cases()` with both full and `quick` sizes. The CI smoke test (`tests/test_benchmarks.py`) picks up new cases automatically.
