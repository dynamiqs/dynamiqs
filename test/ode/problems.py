import math

import torch
import torch.nn as nn
from torch.linalg import matrix_exp as expm

# --------------------------------------------------------------------------------------
#     Test problems for regular solver
# --------------------------------------------------------------------------------------


class RabiODE(nn.Module):
    """Test problem for complex-valued ODEs, corresponding to Rabi oscillations
    of a quantum two-level system."""
    def __init__(self, dtype):
        super().__init__()
        torch.manual_seed(0)
        self.dtype = DTYPE_EQUIV[dtype]
        self.tspan = torch.linspace(0., 2.0, 10)
        self.cst = nn.Parameter(torch.randn(1, dtype=dtype))
        yr = torch.rand(2, 2, dtype=self.dtype)
        self.y0 = nn.Parameter(0.5 * (yr + yr.adjoint()) / torch.real(yr.trace()))
        self.atol, self.rtol = 1e-8, 1e-6
        self.test_tol = 3e-3
        self.sx = torch.tensor([[0, 1], [1, 0]], dtype=self.dtype)
        self.si = torch.tensor([[1, 0], [0, 1]], dtype=self.dtype)

    def forward(self, t, y):
        yt = -1.0j * self.cst * self.sx @ y
        yt += yt.adjoint()
        return yt

    def forward_adj(self, t, a):
        at = -1.0j * self.cst * self.sx @ a
        at += at.adjoint()
        return at

    def solution(self):
        U_t = torch.cos(self.cst * self.tspan)[:, None, None] * self.si
        U_t += -1j * torch.sin(self.cst * self.tspan)[:, None, None] * self.sx
        return U_t @ self.y0 @ U_t.adjoint()


class SineODE(nn.Module):
    """Test problem with a scalar sine problem (taken from torchdiffeq).
    dy/dt = f(t,y) = 2y/t + t^4 sin(2t) - t^2 + 4t^3;
    da/dt = -2a / t y(t) = -t^4 cos(2t) / 2 + t^3 sin(2t) / 2
    + t^2 cos(2t) / 4 + 2t^4 - t^3 + (pi - 1/4) t^2"""
    def __init__(self, dtype):
        super().__init__()
        self.dtype = dtype
        self.tspan = torch.linspace(1., 8., 10)
        self.cst = nn.Parameter(torch.tensor(math.pi - 0.25).to(self.dtype))
        self.y0 = self.sol_t(self.tspan[:1]).to(self.dtype)
        self.atol, self.rtol = 1e-6, 1e-6
        self.test_tol = 1e-4

    def forward(self, t, y):
        return 2 * y / t + t**4 * torch.sin(2 * t) - t**2 + 4 * t**3

    def forward_adj(self, t, a):
        return

    def sol_t(self, t):
        return (
            (0.25 * t**2 - 0.5 * t**4) * torch.cos(2 * t) +
            0.5 * t**3 * torch.sin(2 * t) + 2 * t**4 - t**3 + self.cst * t**2
        )

    def solution(self):
        return self.sol_t(self.tspan).unsqueeze(1)


class LinearODE(torch.nn.Module):
    """Test problem with a 2-D matrix problem (taken from torchdiffeq).
    dy/dt = A @ y
    """
    def __init__(self, dtype, dim=10):
        super().__init__()
        torch.manual_seed(0)
        self.dtype = dtype
        self.tspan = torch.linspace(0., 1., 10)
        self.cst = nn.Parameter(torch.rand(1, dtype=self.dtype))
        self.y0 = nn.Parameter(torch.ones(dim, 1, dtype=self.dtype))
        self.atol, self.rtol = 1e-8, 1e-8
        self.test_tol = 1e-2
        self.dim = dim
        U = 0.1 * torch.randn(dim, dim)
        self.A = nn.Parameter((U + U.T).to(self.dtype))

    def forward(self, t, y):
        return self.cst * self.A @ y

    def forward_adj(self, t, a):
        return -self.cst * self.A.T @ a

    def solution(self):
        sol = expm(self.cst * self.tspan[:, None, None] * self.A) @ self.y0
        return sol


# --------------------------------------------------------------------------------------
#     Test problems for outsourced solver
# --------------------------------------------------------------------------------------


class LinearODE_OUT(torch.nn.Module):
    """Test problem with a 2-D matrix problem (taken from torchdiffeq).
    dy/dt = A @ y
    """
    def __init__(self, dtype, dim=10):
        super().__init__()
        torch.manual_seed(0)
        self.dtype = dtype
        self.tspan = torch.linspace(0., 1., 1000)
        self.save_at = torch.linspace(0., 1., 10)
        self.cst = nn.Parameter(torch.rand(1, dtype=self.dtype))
        self.y0 = nn.Parameter(torch.ones(dim, 1, dtype=self.dtype))
        self.atol, self.rtol = 1e-8, 1e-8
        self.test_tol = 1e-3
        self.dim = dim
        U = 0.1 * torch.randn(dim, dim)
        self.A = nn.Parameter((U + U.T).to(self.dtype))

    def forward(self, t, dt, y):
        Ay = dt * self.cst * self.A @ y
        return y + Ay + 0.5 * dt * self.cst * self.A @ Ay

    def forward_adj(self, t, dt, a):
        Aa = dt * self.cst * self.A.T @ a
        return a - Aa + 0.5 * dt * self.cst * self.A.T @ Aa

    def solution(self):
        sol = expm(self.cst * self.save_at[:, None, None] * self.A) @ self.y0
        return sol


# --------------------------------------------------------------------------------------
#     Dictionnaries
# --------------------------------------------------------------------------------------

DTYPE_EQUIV = {
    torch.float16: torch.complex32,
    torch.float32: torch.complex64,
    torch.float64: torch.complex128
}
PROBLEMS = {'rabi': RabiODE, 'sine': SineODE, 'linear': LinearODE}
PROBLEMS_OUT = {'linear': LinearODE_OUT}
