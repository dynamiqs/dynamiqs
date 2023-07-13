from __future__ import annotations

from functools import cached_property

import torch
import torch.nn as nn

from .utils.tensor_types import dtype_complex_to_real

__all__ = [
    'Propagator',
    'Dopri5',
    'Euler',
    'Rouchon',
    'Rouchon1',
    'Rouchon2',
]


class Options:
    GRADIENT_ALG = []

    def __init__(
        self,
        *,
        gradient_alg: str | None = None,
        save_states: bool = True,
        verbose: bool = True,
        dtype: torch.complex64 | torch.complex128 | None = None,
        device: torch.device | None = None,
    ):
        """...

        Args:
            gradient_alg (str, optional): Algorithm used for computing gradients.
                Defaults to `None`.
            save_states (bool, optional): If `True`, the state is saved at every
                time value. If `False`, only the final state is stored and returned.
                Defaults to `True`.
            dtype (torch.dtype, optional): Complex data type to which all complex-valued
                tensors are converted. `t_save` is also converted to a real data type of
                the corresponding precision.
            device (torch.device, optional): Device on which the tensors are stored.
        """
        self.gradient_alg = gradient_alg
        self.save_states = save_states
        self.verbose = verbose
        self.dtype = dtype
        self.device = device

        # check that the gradient algorithm is supported
        if self.gradient_alg is not None and self.gradient_alg not in self.GRADIENT_ALG:
            available_gradient_alg_str = ', '.join(f'"{x}"' for x in self.GRADIENT_ALG)
            raise ValueError(
                f'Gradient algorithm "{self.gradient_alg}" is not defined or not yet'
                f' supported by solver {type(self).__name__} (supported:'
                f' {available_gradient_alg_str}).'
            )

    @cached_property
    def rdtype(self) -> torch.float32 | torch.float64:
        return dtype_complex_to_real(self.dtype)


class AutogradOptions(Options):
    GRADIENT_ALG = ['autograd']


class AdjointOptions(AutogradOptions):
    GRADIENT_ALG = ['autograd', 'adjoint']

    def __init__(self, *, parameters: tuple[nn.Parameter, ...] | None = None, **kwargs):
        """

        Args:
            parameters (tuple of nn.Parameter): Parameters with respect to which
                gradients are computed during the adjoint state backward pass.
        """
        super().__init__(**kwargs)
        self.parameters = parameters

        # check parameters were passed if gradient by the adjoint
        if self.gradient_alg == 'adjoint' and self.parameters is None:
            raise ValueError(
                'For adjoint state gradient computation, parameters must be passed to'
                ' the solver.'
            )


class ODEFixedStep(AutogradOptions):
    def __init__(self, *, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt


class ODEAdaptiveStep(AutogradOptions):
    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        safety_factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.safety_factor = safety_factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Propagator(AutogradOptions):
    pass


class Dopri5(ODEAdaptiveStep):
    pass


class Euler(ODEFixedStep):
    pass


class Rouchon1(ODEFixedStep, AdjointOptions):
    def __init__(self, *, trace_normalization: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.trace_normalization = trace_normalization


# make alias for Rouchon1
Rouchon = Rouchon1


class Rouchon2(ODEFixedStep, AdjointOptions):
    pass
