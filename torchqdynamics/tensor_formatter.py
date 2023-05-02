from __future__ import annotations

from typing import Tuple

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
    ) -> Tuple[Tensor, Tensor]:
        """Batches H and the state (psi or rho)."""

        # convert H to a tensor and batch by default
        H = to_tensor(H, dtype=self.dtype, device=self.device, is_complex=True)
        if H.ndim == 2:
            H_batched = H[None, ...]
        else:
            H_batched = H
            self.H_is_batched = True

        # convert state to a tensor and density matrix and batch by default
        state = to_tensor(state, dtype=self.dtype, device=self.device, is_complex=True)
        if is_ket(state) and state_to_dm:
            state = ket_to_dm(state)
        b_H = H_batched.size(0)
        if state.ndim == 2:
            state_batched = state[None, ...]
        else:
            state_batched = state
            self.state_is_batched = True
        state_batched = state_batched[None, ...].repeat(
            b_H, 1, 1, 1
        )  # (b_H, b_state0, n, n)

        return H_batched, state_batched

    def batch(self, operator: OperatorLike) -> Tensor:
        operator = to_tensor(
            operator, dtype=self.dtype, device=self.device, is_complex=True
        )
        return operator[None, ...] if operator.ndim == 2 else operator

    def unbatch(self, save: Tensor | None) -> Tensor | None:
        if save is None:
            return None

        if not self.state_is_batched:
            save = save.squeeze(1)
        if not self.H_is_batched:
            save = save.squeeze(0)

        return save
