import torch
import torch.nn as nn
from .tableaus import construct_dopri5
from math import sqrt

# -------------------------------------------------------------------------------------------------
#     Solver common functions
# -------------------------------------------------------------------------------------------------

DTYPE_EQUIV = {
    torch.complex32:  torch.float16, 
    torch.complex64:  torch.float32, 
    torch.complex128: torch.float64,
    torch.float16:    torch.float16,
    torch.float32:    torch.float32,
    torch.float64:    torch.float64}

def init_step(f, f0, y0, t0, order, atol, rtol):
    scale = atol + torch.abs(y0) * rtol
    d0, d1 = hairer_norm(y0 / scale), hairer_norm(f0 / scale)

    if d0 < 1e-5 or d1 < 1e-5:
        h0 = torch.tensor(1e-6, dtype=t0.dtype, device=t0.device)
    else:
        h0 = 0.01 * d0 / d1

    x_new = y0 + h0 * f0
    f_new = f(t0 + h0, x_new)
    d2 = hairer_norm((f_new - f0) / scale) / h0
    if d1 <= 1e-15 and d2 <= 1e-15:
        h1 = torch.max(torch.tensor(1e-6, dtype=t0.dtype, device=t0.device), h0 * 1e-3)
    else:
        h1 = (0.01 / max(d1, d2)) ** (1. / float(order + 1))
    dt = torch.min(100 * h0, h1).to(t0)
    return dt

def hairer_norm(tensor):
    return torch.linalg.norm(tensor) / sqrt(tensor.numel())

@torch.no_grad()
def adapt_step(dt, error_ratio, safety, min_factor, max_factor, order):
    if error_ratio == 0: return dt * max_factor
    if error_ratio < 1: min_factor = torch.ones_like(dt)
    exponent = torch.tensor(order, dtype=dt.dtype, device=dt.device).reciprocal()
    factor = torch.min(max_factor, torch.max(safety / error_ratio ** exponent, min_factor))
    return dt * factor

# -------------------------------------------------------------------------------------------------
#     Solver classes
# -------------------------------------------------------------------------------------------------

class BaseSolver(nn.Module):
    def __init__(self, order=None, min_factor:float=0.2, 
                 max_factor:float=10, safety:float=0.9):
        super().__init__()
        self.order = order
        self.min_factor = torch.tensor([min_factor])
        self.max_factor = torch.tensor([max_factor])
        self.safety = torch.tensor([safety])
        self.tableau = None

    def sync_device_dtype(self, y0, tspan):
        """Ensures `y0`, `tspan`, `tableau` and other solver tensors are on the same device with compatible dtypes"""
        # Check compatilibity of `y0`` and `tspan`` types.
        t_dtype = DTYPE_EQUIV[y0.dtype]
        if tspan.dtype != t_dtype:
            self.t_dtype = t_dtype
            tspan = tspan.to(t_dtype)
   
        # Send all solver objects to compatible types and devices
        device = y0.device
        tspan = tspan.to(device)
        self.safety = self.safety.to(device)
        self.min_factor = self.min_factor.to(device)
        self.max_factor = self.max_factor.to(device)
        if self.tableau is not None:
            alpha, beta, csol, cerr = self.tableau
            self.tableau = alpha.to(tspan), beta.to(y0), csol.to(y0), cerr.to(y0)
        return y0, tspan

    def step(self, f, f0, y0, t0, dt):
        raise NotImplementedError("Stepping rule not implemented for the solver")


class DormandPrince45(BaseSolver):
    def __init__(self, y_dtype=torch.complex64, t_dtype=torch.float32):
        super().__init__(order=5)
        self.y_dtype = y_dtype
        self.t_dtype = t_dtype
        self.tableau = construct_dopri5(self.y_dtype, self.t_dtype)
        self.solver_type = 'rk'

    def step(self, f, f0, y0, t0, dt):
        # Import butcher tableau
        alpha, beta, csol, cerr = self.tableau
        
        # Compute runge kutta values
        k = torch.zeros((7,) + f0.shape, dtype=f0.dtype)
        k[0] = f0
        sum_dims = (...,) + (None,)*f0.dim()
        for i in range(1, 7):
            ti = t0 + alpha[i-1] * dt
            yi = y0 + dt.type(self.y_dtype) * (beta[i-1, :i][sum_dims] * k[:i]).sum(0)
            k[i] = f(ti, yi)

        # Compute results
        y1 = y0 + dt.type(self.y_dtype) * (csol[:6][sum_dims] * k[:6]).sum(0)
        y1_err = dt.type(self.y_dtype) * (cerr[sum_dims] * k).sum(0)
        f1 = k[-1]
        return f1, y1, y1_err, k

class Outsource(BaseSolver):
    def __init__(self, y_dtype=torch.complex64, t_dtype=torch.float32):
        super().__init__()
        self.y_dtype = y_dtype
        self.t_dtype = t_dtype
        self.tableau = None
        self.solver_type = 'out'

    def step(self, f, y0, t0, dt):
        return f(t0, dt, y0)

# -------------------------------------------------------------------------------------------------
#     Solver utility functions
# -------------------------------------------------------------------------------------------------

SOLVER_DICT = {
    'dopri5': DormandPrince45, 'DormandPrince45': DormandPrince45, 'DormandPrince5': DormandPrince45,
    'outsource': Outsource, 'out': Outsource}

SOLVERTYPE_DICT = {'dopri5': 'rk', 'DormandPrince45': 'rk', 'DormandPrince5': 'rk', 
                   'outsource': 'out', 'out': 'out'}

def str_to_solver(solver_name, y_dtype=torch.complex64):
    """Transforms string specifying desired solver into an instance of the Solver class."""
    solver = SOLVER_DICT[solver_name]
    t_dtype = DTYPE_EQUIV[y_dtype]
    return solver(y_dtype, t_dtype)

def solver_to_solvertype(solver):
    if isinstance(solver, str):
        return SOLVERTYPE_DICT[solver]
    else:
        return solver.solver_type