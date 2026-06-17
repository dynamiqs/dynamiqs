# Dynamiqs Solver Benchmark Suite

Reproducible benchmark pipeline for comparing numerical solvers across
representative quantum dynamics problems.

## Usage

```bash
# From the project root
python tests/benchmark/
```

The benchmark is opt-in only and will NOT run during normal test collection.

## Problems

| # | Problem | Type | Solvers |
|---|---------|------|---------|
| 1 | **Closed two-qubit** | Schrödinger, constant H | Tsit5, Dopri5/8, Kvaerno3/5, Euler, Rouchon1/2/3, Expm |
| 2 | **Driven-damped oscillator** | Lindblad, cavity | Tsit5, Dopri5/8, Kvaerno3/5, Euler, Rouchon1/2, Expm |
| 3 | **Batched Kerr oscillator** | Lindblad, batched | Tsit5, Dopri5/8, Kvaerno3, Euler, Rouchon2, Expm |
| 4 | **Large-scale closed (12-qubit)** | Schrödinger, 4096-dim | Tsit5, Dopri5, Dopri8 |
| 5 | **Time-dependent qubit** | Schrödinger, time-dep H | Tsit5, Dopri5/8 |
| 6 | **Open time-dependent qubit** | Lindblad, time-dep H | Tsit5, Dopri5/8 |

Each problem defines which solvers apply and their parameters.

## Metrics

- **Runtime** — wall-clock time after JIT compilation, via `block_until_ready()`
- **Fidelity error** — `1 - mean(|<sim|ref>|^2)` over save times
- **L2 state error** — normalised Frobenius norm of state difference
- **nsteps** — solver step count from `result.infos.nsteps`

## Reference Solutions

References follow precedence:
1. Analytical solution (preferred gold standard)
2. `Expm` (matrix exponentiation)
3. High-tolerance `Dopri8(rtol=1e-10)` fallback

Results are cached per problem so all methods share the same reference.

## Output

Results saved to `tests/benchmark/results/benchmark_results.csv`.
A formatted leaderboard is printed to stdout.
