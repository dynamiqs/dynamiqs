from __future__ import annotations

from typing import Any

import torch

from .._utils import to_device
from ..solver import Solver
from ..utils.tensor_types import dtype_complex_to_real, get_cdtype


class Options:
    def __init__(self, solver: Solver, options: dict[str, Any] | None):
        if options is None:
            options = {}

        self.solver = solver
        self.gradient = solver.gradient
        self.options = SharedOptions(**options)

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

    # def as_dict(self) -> dict[str, Any]:
    #     return {
    #         'gradient_alg': self.gradient_alg,
    #         'save_states': self.save_states,
    #         'verbose': self.verbose,
    #         'dtype': self.cdtype,
    #         'device': self.device,
    #     }

    # def __repr__(self) -> str:
    #     return self.__str__()

    # def __str__(self) -> str:
    #     attributes_str = ', '.join(f'{k}={v}' for k, v in self.as_dict().items())
    #     return f'{type(self).__name__}({attributes_str})'


class SharedOptions:
    def __init__(
        self,
        *,
        save_states: bool = True,
        verbose: bool = True,
        dtype: torch.complex64 | torch.complex128 | None = None,
        device: str | torch.device | None = None,
    ):
        """...

        Args:
            save_states (bool, optional): If `True`, the state is saved at every
                time value. If `False`, only the final state is stored and returned.
                Defaults to `True`.
            dtype (torch.dtype, optional): Complex data type to which all complex-valued
                tensors are converted. `tsave` is also converted to a real data type of
                the corresponding precision.
            device (string or torch.device, optional): Device on which the tensors are
                stored.
        """
        self.save_states = save_states
        self.verbose = verbose
        self.cdtype = get_cdtype(dtype)
        self.rdtype = dtype_complex_to_real(self.cdtype)
        self.device = to_device(device)
