import warnings

import torch

from .solver import AdaptativeStep, FixedStep


def odeint(qsolver, y0, tsave, sensitivity='autograd', variables=None):
    # check arguments
    tsave = check_tsave(tsave)
    if (variables is not None) and (sensitivity in [None, 'autograd']):
        warnings.warn('Argument `variables` was supplied in `odeint` but not used.')

    # dispatch to appropriate odeint subroutine
    if sensitivity is None:
        return odeint_inplace(qsolver, y0, tsave)
    elif sensitivity == 'autograd':
        return odeint_main(qsolver, y0, tsave)
    elif sensitivity == 'adjoint':
        return odeint_adjoint(qsolver, y0, tsave, variables)


def odeint_main(qsolver, y0, tsave):
    if isinstance(qsolver.options, FixedStep):
        return _fixed_odeint(qsolver, y0, tsave)
    elif isinstance(qsolver.options, AdaptativeStep):
        return _adaptive_odeint(qsolver, y0, tsave)


def odeint_inplace(qsolver, y0, tsave):
    # TODO: Simple solution for now so torch does not store gradients. This
    #       is probably slower than a genuine in-place solver.
    with torch.no_grad():
        return odeint_main(qsolver, y0, tsave)


def odeint_adjoint(qsolver, y0, tsave, variables):
    raise NotImplementedError


def _adaptive_odeint(qsolver, y0, tsave):
    raise NotImplementedError


def _fixed_odeint(qsolver, y0, tsave):
    # initialize save tensor
    ysave = torch.zeros(len(tsave), *y0.shape).to(y0)
    save_counter = 0
    if tsave[0] == 0.0:
        ysave[0] = y0
        save_counter += 1

    # get qsolver fixed time step
    dt = qsolver.options.dt

    # run the ODE routine
    t, y = 0.0, y0
    while t < tsave[-1]:
        # iterate solution
        y = qsolver.forward(t, dt, y)
        t = t + dt

        # save solution
        if t >= tsave[save_counter]:
            ysave[save_counter] = y
            save_counter += 1

    return ysave


def check_tsave(tsave):
    """Check that `tsave` is valid (it must be a non-empty 1D tensor sorted in
    strictly ascending order and containing only positive values)."""
    if tsave.dim() != 1 or len(tsave) == 0:
        raise ValueError('Argument `tsave` must be a non-empty 1D torch.Tensor.')
    if not torch.all(torch.diff(tsave) > 0):
        raise ValueError('Argument `tsave` must be sorted in strictly ascending order.')
    if not torch.all(tsave >= 0):
        raise ValueError('Argument `tsave` must contain positive values only.')
    return tsave
