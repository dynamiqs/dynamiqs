from __future__ import annotations

import torch.nn as nn

__all__ = [
    'Propagator',
    'Dopri5',
    'Euler',
    'Rouchon',
    'Rouchon1',
    'Rouchon1_5',
    'Rouchon2',
]


class Options:
    GRADIENT_ALG = ['autograd']

    def __init__(
        self,
        *,
        gradient_alg: str | None = None,
        save_states: bool = True,
        verbose: bool = True,
    ):
        """...

        Args:
            gradient_alg (str, optional): Algorithm used for computing gradients.
                Defaults to `None`.
            save_states (bool, optional): If `True`, the state is saved at every
                time value. If `False`, only the final state is stored and returned.
                Defaults to `True`.
        """
        self.gradient_alg = gradient_alg
        self.save_states = save_states
        self.verbose = verbose

        # check that the gradient algorithm is supported
        if self.gradient_alg is not None and self.gradient_alg not in self.GRADIENT_ALG:
            raise ValueError(
                f'Gradient algorithm {self.gradient_alg} is not defined or not yet'
                f' supported by solver {type(self)}.'
            )


class AdjointOptions(Options):
    GRADIENT_ALG = ['autograd', 'adjoint']

    def __init__(
        self,
        *,
        gradient_alg: str | None = None,
        save_states: bool = True,
        verbose: bool = True,
        parameters: tuple[nn.Parameter, ...] | None = None,
    ):
        """

        Args:
            parameters (tuple of nn.Parameter): Parameters with respect to which
                gradients are computed during the adjoint state backward pass.
        """
        super().__init__(
            gradient_alg=gradient_alg, save_states=save_states, verbose=verbose
        )
        self.parameters = parameters

        # check parameters were passed if gradient by the adjoint
        if self.gradient_alg == 'adjoint' and self.parameters is None:
            raise ValueError(
                'For adjoint state gradient computation, parameters must be passed to'
                ' the solver.'
            )


class ODEFixedStep(Options):
    def __init__(self, *, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt


class ODEAdaptiveStep(Options):
    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.factor = factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Propagator(Options):
    pass


class Dopri5(ODEAdaptiveStep):
    pass


class Euler(ODEFixedStep):
    pass


class Rouchon1(ODEFixedStep, AdjointOptions):
    pass


# make alias for Rouchon1
Rouchon = Rouchon1


class Rouchon1_5(ODEFixedStep, AdjointOptions):
    pass


class Rouchon2(ODEFixedStep, AdjointOptions):
    pass
