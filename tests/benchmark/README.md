# Dynamiqs Solver Benchmarks

This directory contains an opt-in benchmark suite for comparing Dynamiqs solvers
on representative quantum-dynamics problems.

The suite records:

- benchmark name,
- solver method,
- runtime,
- number of solver steps when available,
- accuracy against an analytical or high-accuracy reference,
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

The normal test suite only runs a small smoke test. The full benchmark is intended
for local solver comparison and regression tracking across commits.

## Benchmark Problems

- `cross_resonance_modulated_sesolve`: time-dependent closed two-qubit dynamics,
  referenced against high-accuracy `Dopri8`.
- `driven_damped_harmonic_oscillator`: one-mode Lindblad dynamics, referenced
  against the analytical oscillator amplitude.
- `batched_kerr_oscillator_mesolve`: batched nonlinear open-system dynamics,
  referenced against high-accuracy `Dopri8`.
- `ising_chain_8q_sesolve`: closed many-body state-vector evolution, referenced
  against high-accuracy `Dopri8`. The problem is kept at 8 qubits by default so
  that it remains reasonable for local CPU runs; it can be scaled to 12 qubits by
  calling `ising_chain_sesolve(num_qubits=12)`.
- `two_mode_pwc_vmap_mesolve`: two-mode dissipative PWC drive with batched pulse
  amplitudes, referenced against high-accuracy `Dopri8`.
- `zeno_cnot_reduced_mesolve`: reduced three-mode dissipative gate-like model,
  referenced against high-accuracy `Dopri8`.
