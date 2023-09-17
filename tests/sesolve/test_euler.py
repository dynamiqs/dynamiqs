from ..solver_tester import SolverTester
from .closed_system import cavity_8, grad_cavity_8


class TestSEEuler(SolverTester):
    def test_batching(self):
        options = dict(dt=1e-2)
        self._test_batching(cavity_8, 'euler', options=options)

    def test_correctness(self):
        options = dict(dt=1e-4)
        self._test_correctness(
            cavity_8,
            'euler',
            options=options,
            num_tsave=11,
            ysave_norm_atol=1e-2,
            exp_save_rtol=1e-1,
            exp_save_atol=1e-3,
        )

    def test_autograd(self):
        options = dict(dt=1e-4)
        self._test_gradient(
            grad_cavity_8,
            'euler',
            'autograd',
            options=options,
            num_tsave=11,
            rtol=1e-1,
            atol=1e-2,
        )
