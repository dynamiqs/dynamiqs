# Dynamiqs Solver Benchmark Suite

Run with:

```bash
cd tests/benchmark && python -m benchmark
```

Or from the project root:

```bash
python tests/benchmark/
```

This benchmark suite is opt-in only and will NOT run during normal test collection.

## Problems

The benchmark covers:

1. **Closed two-qubit Schrödinger** - Cross-resonance inspired two-qubit system
2. **Driven-damped harmonic oscillator** - Lindblad master equation with OCavity
3. **Batched Kerr oscillator mesolve** - Batch of drive amplitudes
4. **12-qubit Ising chain sesolve** - Large-scale Schrödinger evolution
5. **Time-dependent qubit** - TDQubit with time-varying Hamiltonian
6. **Open time-dependent qubit** - OTDQubit with relaxation

## Output

Results are saved to `tests/benchmark/results/` as CSV files and a leaderboard summary is printed to stdout.