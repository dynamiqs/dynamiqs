from __future__ import annotations

from typing import get_args

import torch
from torch import Tensor

from .utils.tensor_types import OperatorLike, TDOperatorLike, to_tensor
from .utils.utils import is_ket, ket_to_dm


class TensorFormatter:
    def __init__(
        self,
        dtype: torch.complex64 | torch.complex128 | None,
        device: torch.device | None,
    ):
        self.dtype = dtype
        self.device = device

        self.H_is_batched = False
        self.state_is_batched = False

    def batch_H_and_state(
        self, H: TDOperatorLike, state: OperatorLike, state_to_dm: bool = False
    ) -> tuple[Tensor, Tensor]:
        """Batch Hamiltonian and state (state vector or density matrix)."""
        # handle H batching
        if isinstance(H, get_args(OperatorLike)):
            H = to_tensor(H, dtype=self.dtype, device=self.device, is_complex=True)
            if H.ndim == 2:
                H_batched = H[None, None, ...]  # (1, 1, n, n)
                b_H = 1
            else:
                H_batched = H[:, None, ...]  # (b_H, 1, n, n)
                b_H = H.size(0)
                self.H_is_batched = True
        elif isinstance(H, callable):
            H0 = H(0.0)

            # check type, dtype and device match
            error_message = (
                'For time-dependent problems, the Hamiltonian provided should be a'
                ' `torch.Tensor` of appropriate dtype and device. This avoids type'
                ' conversion or device transfer at every time step that would slow'
                ' down the solver. '
            )
            if not isinstance(H0, Tensor):
                raise TypeError(
                    error_message
                    + f'`H` is currently of type {type(H0)} instead of `torch.Tensor`.'
                )
            elif H0.dtype != self.dtype:
                raise TypeError(
                    error_message
                    + f'`H` is currently of dtype {H0.dtype} instead of {self.dtype}.'
                )
            elif H0.device != self.device:
                raise TypeError(
                    error_message
                    + f'`H` is currently on device {H0.device} instead of'
                    f' {self.device}.'
                )

            # apply batching dynamically
            if H0.ndim == 2:
                H_batched = lambda t: H(t)[None, None, ...]  # (1, 1, n, n)
                b_H = 1
            else:
                H_batched = lambda t: H(t)[:, None, ...]  # (b_H, 1, n, n)
                b_H = H0.size(0)
                self.H_is_batched = True

        # convert state to tensor and density matrix if needed
        state = to_tensor(state, dtype=self.dtype, device=self.device, is_complex=True)
        if is_ket(state) and state_to_dm:
            state = ket_to_dm(state)

        # handle state batching
        if state.ndim == 2:
            state_batched = state[None, None, ...]  # (1, 1, n, n)
        else:
            state_batched = state[None, :, ...]  # (1, b_state, n, n)
            self.state_is_batched = True
        state_batched = state_batched.repeat(b_H, 1, 1, 1)  # (b_H, b_state, n, n)

        return H_batched, state_batched

    def batch(self, operator: OperatorLike) -> Tensor:
        """Batch a given operator according to the Hamiltonian and state."""
        operator = to_tensor(
            operator, dtype=self.dtype, device=self.device, is_complex=True
        )
        return operator[None, ...] if operator.ndim == 2 else operator

    def unbatch(self, save: Tensor | None) -> Tensor | None:
        """Unbatch saved tensors."""
        if save is None:
            return None

        if not self.state_is_batched:
            save = save.squeeze(1)
        if not self.H_is_batched:
            save = save.squeeze(0)

        return save
