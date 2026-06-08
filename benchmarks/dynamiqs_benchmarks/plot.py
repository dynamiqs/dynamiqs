from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_results(results_csv: Path, output_dir: Path | None = None) -> list[Path]:
    output_dir = output_dir or results_csv.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = _read_passed(results_csv)
    if not rows:
        raise ValueError(f'No passing benchmark rows found in {results_csv}.')
    paths = [
        _plot_accuracy_timing(rows, output_dir / 'timing_vs_accuracy.png'),
        _plot_runtime_by_solver(rows, output_dir / 'runtime_by_solver.png'),
    ]
    return paths


def _read_passed(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [row for row in rows if row['status'] == 'pass']


def _plot_accuracy_timing(rows: list[dict[str, str]], path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    solvers = sorted({row['solver'] for row in rows})
    cmap = plt.get_cmap('tab10')
    colors = {solver: cmap(i % 10) for i, solver in enumerate(solvers)}
    for solver in solvers:
        group = [row for row in rows if row['solver'] == solver]
        ax.scatter(
            [float(row['runtime_s']) for row in group],
            [max(float(row['error']), 1e-16) for row in group],
            label=solver,
            color=colors[solver],
            alpha=0.85,
        )
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Runtime after warmup (s)')
    ax.set_ylabel('Relative error vs reference')
    ax.set_title('Dynamiqs solver timing vs accuracy')
    ax.grid(True, which='both', alpha=0.25)
    ax.legend(ncols=2, fontsize='small')
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_runtime_by_solver(rows: list[dict[str, str]], path: Path) -> Path:
    runtimes: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        runtimes[row['solver']].append(float(row['runtime_s']))
    solvers = sorted(runtimes, key=lambda solver: np.median(runtimes[solver]))
    values = [runtimes[solver] for solver in solvers]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(values, tick_labels=solvers, showfliers=True)
    ax.set_yscale('log')
    ax.set_ylabel('Runtime after warmup (s)')
    ax.set_title('Runtime distribution by solver')
    ax.grid(True, axis='y', which='both', alpha=0.25)
    fig.autofmt_xdate(rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Plot Dynamiqs benchmark CSV results.')
    parser.add_argument('results_csv', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args(argv)
    paths = plot_results(args.results_csv, args.output_dir)
    for path in paths:
        print(path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
