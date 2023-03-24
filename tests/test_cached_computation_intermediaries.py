import unittest

from torchqdynamics.decorators import cached_main, cached_depends_on
from torchqdynamics.odeint import ForwardQSolver


class FakeSolver(ForwardQSolver):
    def __init__(self):
        self.H_calls = 0
        self.Q_calls = 0
        self.last_t = None
        self.H_is_constant = False

    def forward(self, _t, _dt, _rho):
        pass

    @cached_main
    def H(self, t):
        self.H_calls += 1

        if self.H_is_constant:
            return 0, False

        if t < 1:
            return 1, True
        elif t < 2:
            return 2, True
        else:
            return 3, True

    @cached_depends_on({"_H": "H"})
    def Q(self, _H, dt):
        self.Q_calls += 1
        if dt < 1:
            return 1
        else:
            return dt


class TestCachedComputationIntermediaries(unittest.TestCase):
    def test_values_are_cached(self):
        solver = FakeSolver()
        solver.H(0.1)
        solver.Q(None, 0.1)
        self.assertEqual(solver.H_calls, 1)
        self.assertEqual(solver.Q_calls, 1)

        solver.H(0.1)
        solver.Q(None, 0.1)
        self.assertEqual(solver.H_calls, 1)
        self.assertEqual(solver.Q_calls, 1)

    def test_values_are_freezed(self):
        solver = FakeSolver()
        solver.H(0.1)
        solver.Q(None, 0.0)
        self.assertEqual(solver.Q_calls, 1)

        solver.H(0.2)
        solver.Q(None, 0.0)
        self.assertEqual(solver.Q_calls, 2)

        solver.H_is_constant = True
        solver.H(0.3)
        solver.Q(None, 0.0)
        self.assertEqual(solver.Q_calls, 2)

    def test_cache_is_flushed_on_arguments_change(self):
        solver = FakeSolver()
        solver.H(0.1)
        solver.Q(None, 0.1)
        self.assertEqual(solver.H_calls, 1)
        self.assertEqual(solver.Q_calls, 1)

        solver.H(0.1)
        solver.Q(None, 0.2)
        self.assertEqual(solver.H_calls, 1)
        self.assertEqual(solver.Q_calls, 2)

        solver.H(0.1)
        solver.Q(None, 0.2)
        self.assertEqual(solver.H_calls, 1)
        self.assertEqual(solver.Q_calls, 2)


if __name__ == '__main__':
    unittest.main()
