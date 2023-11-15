from __future__ import annotations

from abc import ABC

import torch


class System(ABC):
    pass


class ClosedSystem(System):
    def __init__(self):
        self.H = None
        self.y0 = None
        self.tsave = None

    def to(self, dtype: torch.dtype, device: torch.device):
        self.y0 = self.y0.to(dtype=dtype, device=device)
        self.tsave = self.tsave.to(dtype=dtype, device=device)


class OpenSystem(ClosedSystem):
    def __init__(self):
        super().__init__()
        self.jump_ops = None

    def to(self, dtype: torch.dtype, device: torch.device):
        super().to(dtype=dtype, device=device)
        self.jump_ops = [op.to(dtype=dtype, device=device) for op in self.jump_ops]
