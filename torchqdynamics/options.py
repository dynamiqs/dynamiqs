from __future__ import annotations

from typing import Literal

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
    def __init__(
        self,
        *,
        gradient_alg: Literal | None = None,
        save_states: bool = True,
        verbose: bool = True,
    ):
        """...

        Args:
            save_states (bool, optional): If `True`, the state is saved at every
                time value. If `False`, only the final state is stored and returned.
                Defaults to `True`.
        """
        self.gradient_alg = gradient_alg
        self.save_states = save_states
        self.verbose = verbose

        self._gradient_algs = []

    def _check_gradient_alg(self):
        """Check that the gradient algorithm is supported."""
        if (
            self.gradient_alg is not None
            and self.gradient_alg not in self._gradient_algs
        ):
            raise ValueError(
                f'Gradient algorithm {self.gradient_alg} is not defined or not yet'
                f' supported by solver {type(self)}.'
            )


class Autograd(Options):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._gradient_algs.append('autograd')


class Adjoint(Options):
    def __init__(self, *, parameters: tuple[nn.Parameter, ...] | None = None, **kwargs):
        super().__init__(**kwargs)
        self.parameters = parameters
        self._gradient_algs.append('adjoint')


class ODEFixedStep(Autograd):
    def __init__(self, *, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt


class ODEAdaptiveStep(Autograd):
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


class Propagator(Autograd):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_gradient_alg()


class Dopri5(ODEAdaptiveStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_gradient_alg()


class Euler(ODEFixedStep):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_gradient_alg()


class Rouchon1(ODEFixedStep, Adjoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_gradient_alg()


# make alias for Rouchon1
Rouchon = Rouchon1


class Rouchon1_5(ODEFixedStep, Adjoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_gradient_alg()


class Rouchon2(ODEFixedStep, Adjoint):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._check_gradient_alg()
