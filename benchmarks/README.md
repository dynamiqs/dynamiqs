# Dynamiqs solver benchmark suite

This opt-in benchmark suite compares deterministic Dynamiqs solvers on representative quantum-dynamics problems. It is designed for local solver studies and long-term regression tracking without making normal CI slow.

## Cases

The suite includes six complementary MVP-style cases:

1. `cross_resonance_modulated_sesolve` — closed two-qubit Schrödinger dynamics with modulated drives.
2. `driven_damped_oscillator_mesolve` — one-mode Lindblad dynamics with an analytical amplitude reference.
3. `batched_kerr_oscillator_mesolve` — batched nonlinear Kerr oscillator with damping.
4. `ising_chain_<N>q_sesolve` — closed many-body transverse-field Ising chain (`N` depends on profile).
5. `two_mode_pwc_batched_mesolve` — two-mode open system with batched piecewise-constant drives.
6. `reduced_zeno_cnot_mesolve` — reduced three-mode dissipative gate-like model.

## Methods

The runner benchmarks the primary deterministic methods requested in the issue:

- `Tsit5`, `Dopri5`, `Dopri8`, `Kvaerno3`, `Kvaerno5`, `Euler`, and `Expm` for compatible `sesolve`/`mesolve` cases.
- `Rouchon1`, `Rouchon2`, and `Rouchon3` for `mesolve` cases.

Stochastic solvers are intentionally not part of the default suite; the data model can be extended with stochastic-specific metrics later.

## Run locally

Smoke run for development:

```bash
python -m benchmarks.dynamiqs_benchmarks --profile smoke --output-dir benchmark_results/smoke
```

Standard opt-in run:

```bash
python -m benchmarks.dynamiqs_benchmarks --profile standard --output-dir benchmark_results/standard
```

Full scaling run:

```bash
python -m benchmarks.dynamiqs_benchmarks --profile full --output-dir benchmark_results/full
```

Filter cases or methods with comma-separated names. Explicit `--method Kvaerno3,Kvaerno5` selections also opt into slow implicit mesolve/many-body combinations that the default run skips to keep the suite practical:

```bash
python -m benchmarks.dynamiqs_benchmarks --profile smoke --case driven_damped_oscillator_mesolve --method Tsit5,Dopri8,Expm
```

## Outputs

Each run writes:

- `results.csv` — one row per case/method with runtime, step counts, relative error, status, reference strategy, and platform metadata.
- `leaderboard.csv` — passing rows sorted per benchmark by accuracy then runtime.
- `metadata.json` — Dynamiqs/JAX/Python/backend metadata.

Create visual summaries from a result CSV:

```bash
python -m benchmarks.dynamiqs_benchmarks.plot benchmark_results/standard/results.csv
```

This creates `timing_vs_accuracy.png` and `runtime_by_solver.png`.

## Reproducibility notes

- Benchmarks default to double precision and disable progress bars.
- Timings exclude first-call JAX compilation by default via one warmup solve per case/method. Pass `--no-warmup` to include compilation.
- The `smoke` profile is suitable for tests and quick sanity checks; use `standard` or `full` for meaningful solver comparison.
- Solver failures are recorded as CSV rows with `status=fail` instead of aborting the run, making stability regressions visible.
