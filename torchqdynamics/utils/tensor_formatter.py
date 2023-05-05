from __future__ import annotations

import torch
from torch import Tensor

from .td_tensor import TDTensor, to_tdtensor
from .tensor_types import OperatorLike, TDOperatorLike, to_tensor
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

    def format_H_and_state(
        self, H: TDOperatorLike, state: OperatorLike, state_to_dm: bool = False
    ) -> tuple[TDTensor, Tensor]:
        """Convert and batch Hamiltonian and state (state vector or density matrix)."""
        # convert Hamiltonian to `TDTensor`
        H = to_tdtensor(H, dtype=self.dtype, device=self.device, is_complex=True)

        # handle Hamiltonian batching
        if H.dim() == 2:  # (n, n)
            H = H.unsqueeze(0)  # (1, n, n)
        else:  # (b_H, n, n)
            self.H_is_batched = True
        H = H.unsqueeze(1)  # (b_H, 1, n, n)
        b_H = H.size(0)

        # convert state to tensor and density matrix if needed
        state = to_tensor(state, dtype=self.dtype, device=self.device, is_complex=True)
        if is_ket(state) and state_to_dm:
            state = ket_to_dm(state)

        # handle state batching
        if state.ndim == 2:  # (n, n)
            state = state.unsqueeze(0)  # (1, n, n)
        else:  # (b_state, n, n)
            self.state_is_batched = True
        state = state.unsqueeze(0)  # (1, b_state, n, n)
        state = state.repeat(b_H, 1, 1, 1)  # (b_H, b_state, n, n)

        return H, state

    def format(self, operator: OperatorLike) -> Tensor:
        """Convert and batch a given operator according to the Hamiltonian and state."""
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
