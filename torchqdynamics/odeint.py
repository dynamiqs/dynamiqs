import warnings
from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from tqdm.auto import tqdm

from .solver import AdaptativeStep, FixedStep
from .solver_utils import add_tuples, bexpect, none_to_zeros_like


class ForwardQSolver(ABC):
    @abstractmethod
    def forward(self, t: float, dt: float, y: Tensor):
        # Args:
        #     y: (..., m, n)
        #
        # Returns:
        #     (..., m, n)
        pass


class AdjointQSolver(ForwardQSolver):
    @abstractmethod
    def backward_augmented(
        self, t: float, dt: float, y: Tensor, a: Tensor, parameters: Tuple[nn.Parameter,
                                                                           ...]
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
    args = (qsolver, y0, t_save, exp_ops, save_states)
    if gradient_alg is None:
        return _odeint_inplace(*args)
    elif gradient_alg == 'autograd':
        return _odeint_main(*args)
    elif gradient_alg == 'adjoint':
        return _odeint_adjoint(*args, parameters)
    else:
        raise ValueError(
            f'Automatic differentiation algorithm {gradient_alg} is not defined.'
        )


def _odeint_main(
    qsolver: ForwardQSolver, y0: Tensor, t_save: Tensor, exp_ops: Tensor,
    save_states: bool
):
    if isinstance(qsolver.options, FixedStep):
        dt = qsolver.options.dt
        return _fixed_odeint(qsolver, y0, t_save, dt, exp_ops, save_states)
    elif isinstance(qsolver.options, AdaptativeStep):
        return _adaptive_odeint(qsolver, y0, t_save, exp_ops, save_states)


# For now we use *args and **kwargs for helper methods that are not implemented to ease
# the potential api changes that could occur later. When a method is implemented the
# methods should take the same arguments as all others.
def _odeint_inplace(*args, **kwargs):
    # TODO: Simple solution for now so torch does not store gradients. This
    #       is probably slower than a genuine in-place solver.
    with torch.no_grad():
        return _odeint_main(*args, **kwargs)


def _adaptive_odeint(*_args, **_kwargs):
    raise NotImplementedError


def _fixed_odeint(
    qsolver: ForwardQSolver, y0: Tensor, t_save: Tensor, dt: float, exp_ops: Tensor,
    save_states: bool
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
    times = torch.linspace(0.0, t_save[-1], torch.round(t_save[-1] / dt).int() + 1)

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
        y = qsolver.forward(t, dt, y)

    # save final time step (`t` goes `0.0` to `t_save[-1]` excluded)
    if save_states:
        y_save[..., save_counter, :, :] = y
    if len(exp_ops) > 0:
        exp_save[..., save_counter] = bexpect(exp_ops, y)

    return y_save, exp_save


def _odeint_adjoint(
    qsolver: AdjointQSolver, y0: Tensor, t_save: Tensor, exp_ops: Tensor,
    save_states: bool, parameters: Tuple[nn.Parameter, ...]
):
    """Integrate an ODE using the adjoint method in the backward pass."""
    return ODEIntAdjoint.apply(qsolver, y0, t_save, exp_ops, save_states, *parameters)


def _odeint_augmented_main(
    qsolver: AdjointQSolver, y0: Tensor, a0: Tensor, g0: Tuple[Tensor, ...],
    t_span: Tensor, parameters: Tuple[nn.Parameter, ...]
):
    """Integrate the augmented ODE backward."""
    if isinstance(qsolver.options, FixedStep):
        dt = qsolver.options.dt
        return _fixed_odeint_augmented(qsolver, y0, a0, g0, t_span, dt, parameters)
    elif isinstance(qsolver.options, AdaptativeStep):
        return _adaptive_odeint_augmented(qsolver, y0, a0, g0, t_span, parameters)


def _adaptive_odeint_augmented(*_args, **_kwargs):
    """Integrate the augmented ODE backward using an adaptive time step solver."""
    raise NotImplementedError


def _fixed_odeint_augmented(
    qsolver: AdjointQSolver, y0: Tensor, a0: Tensor, g0: Tuple[Tensor, ...],
    t_span: Tensor, dt: float, parameters: Tuple[nn.Parameter, ...]
):
    """Integrate the augmented ODE backward using a fixed time step solver."""
    # check t_span
    if not (t_span.ndim == 1 and len(t_span) == 2):
        raise ValueError(
            f'`t_span` should be a tensor of shape (2,), but has shape {t_span.shape}.'
        )
    if t_span[1] <= t_span[0]:
        raise ValueError(f'`t_span` should be sorted in ascending order.')

    T = t_span[0] - t_span[1]
    if not torch.allclose(torch.round(T / dt), T / dt):
        raise ValueError(f'The total time of evolution should be a multiple of dt.')

    # define time values
    times = torch.linspace(t_span[0], t_span[-1], torch.round(T / dt).int() + 1)

    # run the ode routine
    y, a, g = y0, a0, g0
    y, a = y.requires_grad_(True), a.requires_grad_(True)
    for t in tqdm(times[:-1], leave=False):
        with torch.enable_grad():
            # compute y(t-dt) and a(t-dt) with the qsolver
            y, a = qsolver.backward_augmented(t, dt, y, a, parameters)

            # compute g(t-dt)
            dg = torch.autograd.grad(
                a, parameters, y, allow_unused=True, retain_graph=True
            )
            dg = none_to_zeros_like(dgrad, parameters)
            g = add_tuples(g, dg)

    return y, a, g


class ODEIntAdjoint(torch.autograd.Function):
    """Class for ODE integration with a custom adjoint method backward pass."""
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

        Throughout this function, `y` is the state, `a = dL/dy` is the adjoint state,
        and `g = dL/dp` is the gradient w.r.t. the parameters, where `L` is the loss
        function and `p` the parameters.

        TODO: Check that stacking `y` and `a` does not improve performance of the
        solver. There was funky stuff happening in this regard in some tests I ran.
        """
        # disable gradient computation
        torch.set_grad_enabled(False)

        # unpack context
        qsolver = ctx.qsolver
        t_save = ctx.t_save
        y_save, *parameters = ctx.saved_tensors

        # initialize time list
        t_span = t_save
        if t_span[0] != 0.0:
            t_span = torch.cat((torch.zeros(1), t_span))

        # initialize state, adjoint and gradients
        y = y_save[-1]
        a = grad_y[-1][-1]
        g = tuple(torch.zeros_like(_p).to(y) for _p in parameters)

        # solve the augmented equation backward between every checkpoint
        for i in tqdm(range(len(t_span) - 1, 0, -1)):
            # initialize time between both checkpoints
            t_span_segment = t_span[i - 1:i + 1]

            # run odeint on augmented solution
            y, a, g = _odeint_augmented(
                qsolver, y, a, g, t_span_segment, parameters=parameters
            )

            # replace y with its checkpointed version
            y = y_save[i - 1]

            # update adjoint wrt this time point by adding dL / dy(t)
            a += grad_y[-1][i - 1]

        # convert gradients of real-valued parameters to real-valued gradients
        g = tuple(
            _g.real if _p.is_floating_point() else _g
            for (_g, _p) in zip(g, parameters)
        )

        # return the computed gradients w.r.t. each argument in `forward`
        return None, a, None, None, None, *g


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
