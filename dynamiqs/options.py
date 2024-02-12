from __future__ import annotations

import equinox as eqx
from jaxtyping import Scalar


class Options(eqx.Module):
    save_states: bool = True
    verbose: bool = True
    cartesian_batching: bool = True
    t0: Scalar | None = None  # defaults to tsave[0]
