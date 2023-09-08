from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMEEuler(SolverTester):
    def test_batching(self):
        options = dict(dt=1e-2)
        self._test_batching('euler', options, leaky_cavity_8)

    def test_correctness(self):
        options = dict(dt=1e-4)
        self._test_correctness(
            'euler',
            options,
            leaky_cavity_8,
            num_t_save=11,
            y_save_norm_atol=1e-2,
            exp_save_rtol=1e-2,
            exp_save_atol=1e-3,
        )

    def test_autograd(self):
        options = dict(dt=1e-3, gradient_alg='autograd')
        self._test_gradient(
            'euler', options, grad_leaky_cavity_8, num_t_save=11, rtol=1e-2, atol=1e-2
        )
