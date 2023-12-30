import jax
from jax import Array

from ..solvers.abstract import PropagatorSolver, SolverState
from ..solvers.base import MEIterativeSolver
from ..types import Scalar
from ..utils.vectorization import operator_to_vector, slindbladian, vector_to_operator


class MEPropagator(MEIterativeSolver, PropagatorSolver):
    def __init__(self, *args):
        super().__init__(*args)
        self.H = self.H(0.0)
        self.lindbladian = slindbladian(self.H, self.L)  # (n^2, n^2)

    def forward(self, t: Scalar, delta_t: Scalar, y: Array) -> Array:
        propagator = jax.scipy.linalg.expm(self.lindbladian * delta_t)
        return propagator @ y

    def init_solver_state(self) -> SolverState:
        solver_state = super().init_solver_state()
        y = operator_to_vector(solver_state.y)  # (n^2, 1)
        return solver_state._replace(y=y)

    def save(self, t: Scalar, solver_state: SolverState) -> SolverState:
        # override `save` method to convert `y` from vector to operator
        y = solver_state.y
        solver_state = solver_state._replace(y=vector_to_operator(y))
        solver_state = super().save(t, solver_state)
        solver_state = solver_state._replace(y=y)
        return solver_state


# todo: save in (n^2, 1) format, convert at the end
