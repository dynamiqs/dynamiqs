from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torchdiffeq


def mesolve(
    H: torch.Tensor,
    Ls: List[torch.Tensor],
    rho0: torch.Tensor,
    tlist: np.ndarray,
    solver: str = 'rk4',
    options: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    if options is None:
        options = {}

    if solver == 'rk4':
        if not 'step_size' in options:
            options['step_size'] = 0.001
        return _mesolve_torchdiffeq(H, Ls, rho0, tlist, options)

    raise ValueError('invalid solver (supported: \'rk4\')')


def _dissipator(L: torch.tensor, Ldag: torch.tensor, rho: torch.tensor) -> torch.tensor:
    # [speedup] by using the precomputed jump operator adjoint
    return L @ rho @ Ldag - 0.5 * Ldag @ L @ rho - 0.5 * rho @ Ldag @ L


def _lindbladian(
    H: torch.Tensor,
    Ls: torch.Tensor,
    Lsdag: torch.Tensor,
    rho: torch.Tensor,
) -> torch.Tensor:
    # [speedup] by using the precomputed jump operators adjoint
    return -1j * (H @ rho - rho @ H) + _dissipator(Ls, Lsdag, rho).sum(0)


def _mesolve_torchdiffeq(
    H: torch.Tensor,
    Ls: List[torch.Tensor],
    rho0: torch.Tensor,
    tlist: np.ndarray,
    options: Dict[str, Any],
) -> torch.Tensor:
    Ls = torch.stack(Ls)
    # [speedup] by precomputing the jump operators adjoint
    Lsdag = Ls.adjoint().resolve_conj()

    t = torch.from_numpy(tlist)
    options = {'step_size': options['step_size']}

    # This function takes as an argument the current state $\rho$, and
    # returns $d\rho/dt$ by computing the Lindbladian applied to $\rho$.
    f = lambda t, rho: _lindbladian(H, Ls, Lsdag, rho)

    return torchdiffeq.odeint(f, rho0, t, method='rk4', options=options)
