from __future__ import annotations

import csv
import json
from pathlib import Path

from benchmarks.dynamiqs_benchmarks.cases import benchmark_cases
from benchmarks.dynamiqs_benchmarks.runner import run_suite


def test_benchmark_case_catalog_smoke_profile():
    cases = benchmark_cases('smoke')
    assert {case.name for case in cases} >= {
        'cross_resonance_modulated_sesolve',
        'driven_damped_oscillator_mesolve',
        'batched_kerr_oscillator_mesolve',
        'ising_chain_3q_sesolve',
        'two_mode_pwc_batched_mesolve',
        'reduced_zeno_cnot_mesolve',
    }
    assert all(case.reference_strategy for case in cases)
    assert all(case.tsave.shape[0] >= 2 for case in cases)


def test_benchmark_runner_writes_csv_outputs(tmp_path: Path):
    rows = run_suite(
        output_dir=tmp_path,
        profile='smoke',
        selected_cases={'driven_damped_oscillator_mesolve'},
        selected_methods={'Tsit5', 'Expm'},
        warmup=False,
    )
    assert len(rows) == 2
    assert any(row['status'] == 'pass' for row in rows)
    assert (tmp_path / 'results.csv').exists()
    assert (tmp_path / 'leaderboard.csv').exists()
    assert (tmp_path / 'metadata.json').exists()
    with (tmp_path / 'results.csv').open() as f:
        csv_rows = list(csv.DictReader(f))
    assert {row['solver'] for row in csv_rows} == {'Tsit5', 'Expm'}
    assert all(
        row['benchmark'] == 'driven_damped_oscillator_mesolve' for row in csv_rows
    )


def test_benchmark_example_notebook_documents_visual_dashboard():
    notebook_path = Path('benchmark_example/dynamiqs_solver_benchmark_suite.ipynb')
    assert notebook_path.exists()
    notebook = json.loads(notebook_path.read_text())
    source = '\n'.join(''.join(cell.get('source', [])) for cell in notebook['cells'])
    assert 'Accuracy vs speed' in source
    assert 'solver heatmaps' in source
    assert 'Pareto frontiers' in source
    assert 'run_suite(' in source
