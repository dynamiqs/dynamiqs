from __future__ import annotations

import pytest

from tests.order import TEST_LONG

from .problems import smoke_problems
from .runner import run_suite


@pytest.mark.run(order=TEST_LONG)
def test_benchmark_smoke_suite_runs():
    rows = run_suite(smoke_problems(), method_filter={'Tsit5'})

    assert len(rows) == 1
    assert rows[0].status == 'pass'
    assert rows[0].runtime_s is not None
    assert rows[0].nsteps is not None
    assert rows[0].error is not None
    assert rows[0].error < 1e-3
