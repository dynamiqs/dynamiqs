from abc import ABC, abstractmethod
from enum import Enum
from typing import List

import torch
from tqdm import tqdm

from .solver import AdaptativeStep, FixedStep
from .utils import expect


class ForwardQSolver(ABC):
    @abstractmethod
    def forward(self, t, dt, rho):
        pass


class AdjointQSolver(ForwardQSolver):
    @abstractmethod
    def forward_adjoint(self, t, dt, phi):
        pass


class AutoDiffAlgorithm(Enum):
    AUTOGRAD = 'autograd'  # gradient computed by torch
    NONE = 'none'  # don't compute the gradients
    ADJOINT = 'adjoint'  # compute the gradient using the adjoint method


def odeint(
    qsolver,
    y0,
    t_save: torch.Tensor,
    exp_ops: List[torch.Tensor],
    save_states: bool,
    autodiff_algorithm=AutoDiffAlgorithm.AUTOGRAD,
):
    # check arguments
    check_t_save(t_save)

    # dispatch to appropriate odeint subroutine
    args = (qsolver, y0, t_save, exp_ops, save_states)
    if autodiff_algorithm == AutoDiffAlgorithm.NONE:
        return _odeint_inplace(*args)
    elif autodiff_algorithm == AutoDiffAlgorithm.AUTOGRAD:
        return _odeint_main(*args)
    elif autodiff_algorithm == AutoDiffAlgorithm.ADJOINT:
        return _odeint_adjoint(*args)
    else:
        raise ValueError(
            f'Auto differentiation algorithm {autodiff_algorithm} not defined'
        )


def _odeint_main(qsolver, y0, t_save, exp_ops, save_states):
    if isinstance(qsolver.options, FixedStep):
        return _fixed_odeint(
            qsolver, y0, t_save, qsolver.options.dt, exp_ops, save_states
        )
    elif isinstance(qsolver.options, AdaptativeStep):
        return _adaptive_odeint(qsolver, y0, t_save, exp_ops, save_states)


# for now we use *args and **kwargs for helper methods that are not implemented to ease the potential api
# changes that could occur later. When a method is implemented the methods should take the same arguments
# as all others
def _odeint_inplace(*args, **kwargs):
    # TODO: Simple solution for now so torch does not store gradients. This
    #       is probably slower than a genuine in-place solver.
    with torch.no_grad():
        return _odeint_main(*args, **kwargs)


def _odeint_adjoint(*_args, **_kwargs):
    raise NotImplementedError


def _adaptive_odeint(*_args, **_kwargs):
    raise NotImplementedError


def _fixed_odeint(qsolver, y0, t_save, dt, exp_ops, save_states):
    if not torch.allclose(torch.round(t_save / dt), t_save / dt):
        raise ValueError(
            'Every value of argument `t_save` must be a multiple of the time step `dt`.'
        )

    # initialize save tensor
    y_save, exp_save = None, None
    if save_states:
        y_save = torch.zeros(len(t_save), *y0.shape).to(y0)

    if len(exp_ops) > 0:
        exp_save = torch.zeros(len(exp_ops), len(t_save)).to(y0)

    # run the ode routine
    y = y0
    num_times = round(t_save[-1].item() / dt) + 1
    times = torch.linspace(0.0, t_save[-1], num_times)
    save_counter = 0
    for t in tqdm(times[:-1]):
        # save solution
        if t >= t_save[save_counter]:
            if save_states:
                y_save[save_counter] = y
            for j, op in enumerate(exp_ops):
                exp_save[j, save_counter] = expect(op, y)
            save_counter += 1

        # iterate solution
        y = qsolver.forward(t, dt, y)

    # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
    if save_states:
        y_save[save_counter] = y
    for j, op in enumerate(exp_ops):
        exp_save[j, save_counter] = expect(op, y)

    return y_save, exp_save


def check_t_save(t_save):
    """Check that `t_save` is valid (it must be a non-empty 1D tensor sorted in
    strictly ascending order and containing only positive values)."""
    if t_save.dim() != 1 or len(t_save) == 0:
        raise ValueError('Argument `t_save` must be a non-empty 1D torch.Tensor.')
    if not torch.all(torch.diff(t_save) > 0):
        raise ValueError(
            'Argument `t_save` must be sorted in strictly ascending order.'
        )
    if not torch.all(t_save >= 0):
        raise ValueError('Argument `t_save` must contain positive values only.')
