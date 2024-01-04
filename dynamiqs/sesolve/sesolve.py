import diffrax
from jaxtyping import ArrayLike
from dynamiqs.result import Result
from dynamiqs.solvers import Dopri5


def sesolve(
    H: ArrayLike,
    psi0: ArrayLike,
    tsave: ArrayLike,
) -> Result:
    # === default solver
    solver = Dopri5()

    # === solver class
    solvers = {Dopri5: diffrax.Dopri5}
    if not isinstance(solver, tuple(solvers.keys())):
        supported_str = ', '.join(f'`{x.__name__}`' for x in solvers.keys())
        raise ValueError(
            f'Solver of type `{type(solver).__name__}` is not supported (supported'
            f' solver types: {supported_str}).'
        )
    solver_class = solvers[type(solver)]
