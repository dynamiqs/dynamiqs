from __future__ import annotations

from typing import Any

import torch

from .gradient import Adjoint, Gradient
from .solver import Solver
from .utils.array_types import dtype_complex_to_real, get_cdtype


class Options:
    def __init__(
        self, solver: Solver, gradient: Gradient | None, options: dict[str, Any] | None
    ):
        if gradient is not None and not solver.supports_gradient(gradient):
            supported_str = ', '.join(
                f'`{x.__name__}`' for x in solver.SUPPORTED_GRADIENT
            )
            raise ValueError(
                f'Gradient of type `{type(gradient).__name__}` is not supported by'
                f' solver of type `{type(solver).__name__}` (supported gradient types:'
                f' {supported_str}).'
            )

        if options is None:
            options = {}

        self.solver = solver
        self.gradient = gradient
        self.options = SharedOptions(**options)

        if isinstance(self.gradient, Adjoint):
            # move gradient parameters to the appropriate device
            for p in self.gradient.params:
                p.to(self.options.device)

    def __getattr__(self, name: str) -> Any:
        if name in dir(self.solver):
            return getattr(self.solver, name)
        elif name in dir(self.gradient):
            return getattr(self.gradient, name)
        elif name in dir(self.options):
            return getattr(self.options, name)
        else:
            raise AttributeError(
                f'Attribute `{name}` not found in `{type(self).__name__}`.'
            )


class SharedOptions:
    def __init__(
        self,
        *,
        save_states: bool = True,
        save_expects: bool = False,
        verbose: bool = True,
        dtype: torch.complex64 | torch.complex128 | None = None,
        cartesian_batching: bool = True,
    ):
        # save_states (bool, optional): If `True`, the state is saved at every
        #     time value. If `False`, only the final state is stored and returned.
        #     Defaults to `True`.
        # dtype (torch.dtype, optional): Complex data type to which all complex-valued
        #     arrays are converted. `tsave` is also converted to a real data type of
        #     the corresponding precision.
        self.save_states = save_states
        self.save_expects = save_expects
        self.verbose = verbose
        self.cdtype = get_cdtype(dtype)
        self.rdtype = dtype_complex_to_real(self.cdtype)
        self.cartesian_batching = cartesian_batching
