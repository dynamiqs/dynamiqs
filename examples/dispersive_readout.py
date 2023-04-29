from math import pi, sqrt

import torch

import torchqdynamics as tq

"""
H = chi adag @ a @ sigmaz + Omega(t) (a + adag)

"""

# units
ns = 1.0
MHz = 2 * pi * 1e-3

# parameters
chi = 10 * MHz
kappa = 30 * MHz
Omega = torch.tensor(10 * MHz, requires_grad=True)
N = 40

# operators
a, b = tq.destroy(N, 2)
adag, bdag = tq.create(N, 2)
sz = tq.tensprod(tq.eye(N), tq.sigmaz())

H = chi * adag @ a @ sz + Omega * (a + adag)
jump_op = sqrt(kappa) * a

# initial state
rho0 = tq.fock_dm((N, 2), (0, 0))
tsave = torch.linspace(0.0, 10 * ns, 11)

# mesolve
tq.mesolve(H, jump_op, tsave, rho0)
