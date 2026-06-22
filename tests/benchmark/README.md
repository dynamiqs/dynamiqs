# Dynamiqs Solver Benchmarks

This directory contains an opt-in benchmark suite for comparing Dynamiqs solvers
on representative quantum-dynamics problems.

The suite records:

- benchmark name,
- solver method and fixed-step `dt` when applicable,
- runtime,
- mean/min/max solver steps over the batch when available,
- mean/min/max trajectory infidelity over the batch,
- benchmark and reference precision,
- status and error messages,
- Dynamiqs/JAX/backend metadata.

Run a lightweight smoke benchmark:

```bash
python -m tests.benchmark.runner --suite smoke --out benchmark-smoke.csv
```

Run the full deterministic benchmark suite:

```bash
python -m tests.benchmark.runner --suite full --out benchmark-full.csv
```

Limit to selected methods:

```bash
python -m tests.benchmark.runner --suite full --methods Tsit5 Dopri8 Rouchon3
```

Select all three fixed-step configurations in a family:

```bash
python -m tests.benchmark.runner --suite full --methods Euler Rouchon1
```

Run the benchmark itself in double precision:

```bash
python -m tests.benchmark.runner --suite full --precision double
```

The reference defaults to double precision and can be changed independently with
`--reference-precision`. Each problem saves 100 equally spaced states. Accuracy is
reported as the mean `1 - fidelity` over those states. Batched problems first average
over time and then report mean, minimum, and maximum across the batch.

The method matrix is:

- `sesolve`: Euler at `dt=1e-2`, `1e-3`, and `1e-4`; Dopri5; Dopri8; Tsit5;
  Kvaerno3; Kvaerno5.
- `mesolve`: the same methods plus Rouchon1 at the three `dt` values, Rouchon2,
  and Rouchon3.

Problems with compatible constant Hamiltonians may additionally opt into Expm. The
Ising-chain benchmark currently does so.

The normal test suite only runs a small smoke test. The full benchmark is intended
for local solver comparison and regression tracking across commits.

## Benchmark Problems

- `cross_resonance_modulated_sesolve`: time-dependent closed two-qubit dynamics,
  referenced against high-accuracy `Dopri8`.
- `driven_damped_harmonic_oscillator`: one-mode Lindblad dynamics, referenced
  against its analytical coherent-state trajectory.
- `batched_kerr_oscillator_mesolve`: batched nonlinear open-system dynamics,
  referenced against high-accuracy `Dopri8`.
- `ising_chain_8q_sesolve`: closed many-body state-vector evolution, referenced
  against high-accuracy `Dopri8`. The problem is kept at 8 qubits by default so
  that it remains reasonable for local CPU runs; it can be scaled to 12 qubits by
  constructing `IsingChainSESolve(num_qubits=12)`.
- `two_mode_pwc_vmap_mesolve`: two-mode dissipative PWC drive with batched pulse
  amplitudes, referenced against high-accuracy `Dopri8`.
- `zeno_cnot_reduced_mesolve`: reduced three-mode dissipative gate-like model,
  referenced against high-accuracy `Dopri8`.
