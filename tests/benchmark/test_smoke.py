from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import pytest

import dynamiqs as dq

from tests.order import TEST_LONG

from .metrics import extract_nsteps_stats, state_infidelity_stats
from .problems import (
    CrossResonanceModulatedSESolve,
    DrivenDampedHarmonicOscillator,
    IsingChainSESolve,
    ZenoCNOTReducedMESolve,
    all_problems,
    smoke_problems,
)
from .runner import run_suite


@pytest.mark.run(order=TEST_LONG)
def test_benchmark_smoke_suite_runs():
    rows = run_suite(smoke_problems(), method_filter={'Tsit5'})

    assert len(rows) == 1
    row = rows[0]
    assert row.status == 'pass'
    assert row.runtime_s is not None
    assert row.nsteps_mean is not None
    assert row.nsteps_min is not None
    assert row.nsteps_max is not None
    assert row.error_mean is not None
    assert row.error_min is not None
    assert row.error_max is not None
    assert row.error_mean < 1e-3


def test_method_matrix():
    sesolve_names = [x.name for x in CrossResonanceModulatedSESolve.methods]
    mesolve_names = [x.name for x in ZenoCNOTReducedMESolve.methods]

    assert sum(name.startswith('Euler(') for name in sesolve_names) == 3
    assert not any(name.startswith('Rouchon') for name in sesolve_names)
    assert sum(name.startswith('Euler(') for name in mesolve_names) == 3
    assert sum(name.startswith('Rouchon1(') for name in mesolve_names) == 3
    assert 'Rouchon2' in mesolve_names
    assert 'Rouchon3' in mesolve_names
    assert 'Expm' in [x.name for x in IsingChainSESolve.methods]
    assert 'Expm' not in sesolve_names


def test_state_infidelity_stats_preserves_batch():
    states = dq.stack(
        [
            dq.stack([dq.fock(2, 0), dq.fock(2, 0)]),
            dq.stack([dq.fock(2, 0), dq.fock(2, 1)]),
        ]
    )
    reference = dq.stack(
        [
            dq.stack([dq.fock(2, 0), dq.fock(2, 0)]),
            dq.stack([dq.fock(2, 0), dq.fock(2, 0)]),
        ]
    )

    stats = state_infidelity_stats(states, reference)

    assert stats.mean == pytest.approx(0.25)
    assert stats.minimum == pytest.approx(0.0)
    assert stats.maximum == pytest.approx(0.5)


def test_extract_nsteps_stats_preserves_batch():
    result = SimpleNamespace(infos=SimpleNamespace(nsteps=jnp.asarray([4, 7, 10])))

    stats = extract_nsteps_stats(result)

    assert stats is not None
    assert stats.mean == pytest.approx(7.0)
    assert stats.minimum == pytest.approx(4.0)
    assert stats.maximum == pytest.approx(10.0)


def test_driven_damped_reference_shape():
    problem = DrivenDampedHarmonicOscillator()

    reference = dq.to_jax(problem.reference())

    assert reference.shape == (problem.nsave, problem.trunc, problem.trunc)


def test_all_problems_save_100_states():
    assert all(problem.nsave == 100 for problem in all_problems())
