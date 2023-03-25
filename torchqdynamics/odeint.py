import warnings
from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm import tqdm

from .solver import AdaptativeStep, FixedStep
from .solver_utils import bexpect


class ForwardQSolver(ABC):
    def iterate(self, t: float, dt: float, y: Tensor, **kwargs):
        # Args:
        #     y: (..., m, n)
        #
        # Returns:
        #     (..., m, n)
        return self._forward(t, dt, y)

    @abstractmethod
    def _forward(self, t: float, dt: float, y: Tensor):
        pass


class AdjointQSolver(ForwardQSolver):
    def iterate(
        self, t: float, dt: float, y: Tensor, parameters: Tuple[nn.Parameter, ...]
    ):
        if self.forward_mode:
            return self._forward(t, dt, y)
        else:
            return self._backward_augmented(t, dt, y, parameters)

    @abstractmethod
    def _backward_augmented(
        self, t: float, dt: float, aug_y: Tensor, parameters: Tuple[nn.Parameter, ...]
    ):
        pass


def odeint(
    qsolver: ForwardQSolver, y0: Tensor, t_save: Tensor, exp_ops: Tensor,
    save_states: bool, gradient_alg: Optional[Literal['autograd', 'adjoint']],
    parameters: Optional[Tuple[nn.Parameter, ...]]
):
    # Args:
    #     y0: (..., m, n)

    # check arguments
    check_t_save(t_save)

    # raise warning if parameters are defined but not used
    if not (parameters is None or gradient_alg == 'adjoint'):
        warnings.warn('Parameters were supplied in `odeint` but not used.')

    # dispatch to appropriate odeint subroutine
    args = (qsolver, y0, t_save, exp_ops, save_states, parameters)
    if gradient_alg is None:
        return _odeint_inplace(*args)
    elif gradient_alg == 'autograd':
        return _odeint_main(*args)
    elif gradient_alg == 'adjoint':
        return _odeint_adjoint(*args)
    else:
        raise ValueError(
            f'Automatic differentiation algorithm {gradient_alg} is not defined.'
        )


def _odeint_main(
    qsolver: ForwardQSolver, y0: Tensor, t_save: Tensor, exp_ops: Tensor,
    save_states: bool, parameters: Optional[Tuple[nn.Parameter, ...]]
):
    if isinstance(qsolver.options, FixedStep):
        dt = qsolver.options.dt
        return _fixed_odeint(qsolver, y0, t_save, dt, exp_ops, save_states, parameters)
    elif isinstance(qsolver.options, AdaptativeStep):
        return _adaptive_odeint(qsolver, y0, t_save, exp_ops, save_states, parameters)


# For now we use *args and **kwargs for helper methods that are not implemented to ease
# the potential api changes that could occur later. When a method is implemented the
# methods should take the same arguments as all others.
def _odeint_inplace(*args, **kwargs):
    # TODO: Simple solution for now so torch does not store gradients. This
    #       is probably slower than a genuine in-place solver.
    with torch.no_grad():
        return _odeint_main(*args, **kwargs)


def _odeint_adjoint(
    qsolver: AdjointQSolver, y0: Tensor, t_save: Tensor, exp_ops: Tensor,
    save_states: bool, parameters: Tuple[nn.Parameter, ...]
):
    return ODEIntAdjoint.apply(qsolver, y0, t_save, exp_ops, save_states, *parameters)


def _adaptive_odeint(*_args, **_kwargs):
    raise NotImplementedError


def _fixed_odeint(
    qsolver: ForwardQSolver, y0: Tensor, t_save: Tensor, dt: float, exp_ops: Tensor,
    save_states: bool, parameters: Optional[Tuple[nn.Parameter, ...]]
):
    # Args:
    #     y0: (..., m, n)
    #
    # Returns:
    #     (y_save, exp_save) with
    #     - y_save: (..., len(t_save), m, n)
    #     - exp_save: (..., len(exp_ops), len(t_save))

    # assert that `t_save` values are multiples of `dt` (with the default
    # `rtol=1e-5` they differ by at most 0.001% from a multiple of `dt`)
    if not torch.allclose(torch.round(t_save / dt), t_save / dt):
        raise ValueError(
            'Every value of argument `t_save` must be a multiple of the time step `dt`.'
        )

    # initialize save tensor
    y_save, exp_save = None, None
    batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]
    if save_states:
        y_save = torch.zeros(*batch_sizes, len(t_save), m, n).to(y0)

    if len(exp_ops) > 0:
        exp_save = torch.zeros(*batch_sizes, len(exp_ops), len(t_save)).to(y0)

    # define time values
    # Note that defining the solver times as `torch.arange(0.0, t_save[-1], dt)`
    # could result in an error of order `dt` in the save time (e.g. for
    # `t_save[-1]=1.0` and `dt=0.999999` -> `times=[0.000000, 0.999999]`, so
    # the final state would be saved one time step too far at
    # `0.999999 + dt ~ 2.0`). The previous definition ensures that we never
    # perform an extra step, so the error on the correct save time is of
    # `0.001% * dt` instead of `dt`.
    num_times = round(t_save[-1].item() / dt) + 1
    times = torch.linspace(0.0, t_save[-1], num_times)

    # run the ode routine
    y = y0
    save_counter = 0
    for t in tqdm(times[:-1]):
        # save solution
        if t >= t_save[save_counter]:
            if save_states:
                y_save[..., save_counter, :, :] = y
            if len(exp_ops) > 0:
                exp_save[..., save_counter] = bexpect(exp_ops, y)
            save_counter += 1

        # iterate solution
        y = qsolver.iterate(t, dt, y, parameters)

    # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
    if save_states:
        y_save[..., save_counter, :, :] = y
    if len(exp_ops) > 0:
        exp_save[..., save_counter] = bexpect(exp_ops, y)

    return y_save, exp_save


class ODEIntAdjoint(torch.autograd.Function):
    """ODE integrator with a custom adjoint method backward pass."""
    @staticmethod
    def forward(ctx, qsolver, y0, t_save, exp_ops, save_states, *parameters):
        """Forward pass of the ODE integrator."""
        # save into context for backward pass
        ctx.qsolver = qsolver
        ctx.t_save = t_save if save_states else t_save[-1]

        # solve the ODE forward without storing the graph of operations
        y_save, exp_save = _odeint_inplace(qsolver, y0, t_save, exp_ops, save_states)

        # save results and model parameters
        ctx.save_for_backward(y_save, *parameters)

        return y_save, exp_save

    @staticmethod
    def backward(ctx, *grad_y):
        """Backward pass of the ODE integrator.

        An augmented ODE is integrated backwards starting from the final state computed
        during the forward pass. Integration is done in multiple sequential runs
        between every checkpoint of the forward pass, as defined by `t_save`. This
        helps with the stability of the backward integration.

        Note:
            The augmented state is defined as `(torch.stack((y, dL/dy)), dL/dp)` where
            `L` is the loss function, and `p` the parameters. Stacking `y` and `dL/dy`
            inside a single tensor is essential for solver performance. Doing otherwise
            results in a significant slowdown.
        """
        # disable gradient computation
        torch.set_grad_enabled(False)

        # unpack context
        qsolver = ctx.qsolver
        t_save = ctx.t_save
        y_save, *parameters = ctx.saved_tensors

        # prepare checkpoints time list
        t_checkpoint = t_save
        if t_checkpoint[0] != 0.0:
            t_checkpoint = torch.cat((torch.tensor([0]), t_checkpoint))

        # flag qsolver for backward propagation
        qsolver.forward_mode = False

        # initialize augmented state
        grad_p = tuple(torch.zeros_like(p).to(y_save) for p in parameters)
        aug_y = (torch.stack((y_save[-1], grad_y[-1][-1])), grad_p)

        # solve the augmented equation backward between every checkpoint
        for i in range(len(t_checkpoint) - 1, 0, -1):
            # initialize time between both checkpoints
            t1, t0 = t_checkpoint[i - 1], t_checkpoint[i]
            t_segment = torch.tensor([t0 - t1])
            qsolver.t0 = t0

            # run odeint on augmented solution
            aug_y, _ = _odeint_main(
                qsolver, aug_y, t_segment, exp_ops=torch.tensor([]), save_states=False,
                parameters=parameters
            )

            # replace y with its checkpointed version
            aug_y[0][0] = y_save[i - 1]

            # update adjoint wrt this time point by adding dL / dy(t)
            aug_y[0][1] += grad_y[-1][i - 1]

        # flag qsolver for forward propagation
        qsolver.forward_mode = True

        # convert gradients of real-valued parameters to real-valued gradients
        grad_p = tuple(
            g.real if p.is_floating_point() else g
            for (g, p) in zip(aug_y[-1], parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, aug_y[0][1], None, None, None, *grad_p


def check_t_save(t_save: Tensor):
    """Check that `t_save` is valid (it must be a non-empty 1D tensor sorted in
    strictly ascending order and containing only positive values)."""
    if t_save.dim() != 1 or len(t_save) == 0:
        raise ValueError('Argument `t_save` must be a non-empty 1D tensor.')
    if not torch.all(torch.diff(t_save) > 0):
        raise ValueError(
            'Argument `t_save` must be sorted in strictly ascending order.'
        )
    if not torch.all(t_save >= 0):
        raise ValueError('Argument `t_save` must contain positive values only.')
