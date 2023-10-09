from ..solver_tester import SolverTester
from .open_system import grad_leaky_cavity_8, leaky_cavity_8


class TestMEAdaptive(SolverTester):
    def test_batching(self):
        self._test_batching(leaky_cavity_8, 'dopri5')

    def test_correctness(self):
        self._test_correctness(leaky_cavity_8, 'dopri5', num_tsave=11)

    def test_autograd(self):
        self._test_gradient(grad_leaky_cavity_8, 'dopri5', 'autograd', num_tsave=11)

    def test_adjoint(self):
        options = dict(parameters=grad_leaky_cavity_8.parameters)
        self._test_gradient(
            grad_leaky_cavity_8,
            'dopri5',
            'adjoint',
            options=options,
            num_tsave=11,
            atol=1e-2,
        )
