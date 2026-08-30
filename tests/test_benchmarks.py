import pytest

from benchmarks import Case, benchmark_cases, run_case

from .order import TEST_SHORT


@pytest.mark.run(order=TEST_SHORT)
@pytest.mark.parametrize('case', benchmark_cases(quick=True), ids=lambda c: c.key)
def test_benchmark_case_runs(case: Case):
    record = run_case(case, repeats=1)
    assert record['compile_s'] > 0
    assert record['median_s'] > 0
    if case.name != 'mesolve_grad':
        assert record['nsteps'] > 0
    else:
        assert record['nsteps'] is None
