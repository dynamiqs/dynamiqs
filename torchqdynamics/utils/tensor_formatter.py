from __future__ import annotations

import torch
from torch import Tensor

from .tensor_types import (
    OperatorLike,
    TDOperatorLike,
    tdtensor_get_ndim,
    tdtensor_get_size,
    tdtensor_unsqueeze,
    to_tdtensor,
    to_tensor,
)
from .utils import is_ket, ket_to_dm


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
        # convert Hamiltonian to `TDTensor`
        H = to_tdtensor(H, dtype=self.dtype, device=self.device, is_complex=True)

        # handle Hamiltonian batching
        if tdtensor_get_ndim(H) == 2:
            H_batched = tdtensor_unsqueeze(H, (0, 0))  # (1, 1, n, n)
            b_H = 1
        else:
            H_batched = tdtensor_unsqueeze(H, (1,))
            b_H = tdtensor_get_size(H, 0)  # (b_H, 1, n, n)
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
