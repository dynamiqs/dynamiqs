# Benchmark example notebook

Open `dynamiqs_solver_benchmark_suite.ipynb` from the repository root (or from this folder) to run the Dynamiqs solver benchmark suite interactively.

The notebook provides:

- a researched overview of how the benchmark suite maps to Dynamiqs solver APIs;
- configurable `smoke`, `standard`, and `full` benchmark runs;
- CSV loading for previously generated runs;
- leaderboard, failure/skip summaries, and per-case rankings;
- timing-vs-accuracy scatter plots;
- runtime and accuracy heatmaps across benchmarks and solvers;
- Pareto-frontier views for speed/accuracy tradeoffs.

For a quick run, keep `PROFILE = 'smoke'`. For publishable comparisons, use `PROFILE = 'standard'` or `PROFILE = 'full'` and keep the generated CSV/metadata files under `benchmark_results/`.
