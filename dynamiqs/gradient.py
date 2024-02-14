from __future__ import annotations

import equinox as eqx


class Gradient(eqx.Module):
    pass


class Autograd(Gradient):
    pass


class CheckpointAutograd(Gradient):
    checkpoints: int | None = None
