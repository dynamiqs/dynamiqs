from .system import System
import dynamiqs as dq
import torch
from math import pi, sqrt, exp, cos
from scipy.special import erf

# units
GHz, MHz, kHz = 2 * pi, 2 * pi * 1e-3, 2 * pi * 1e-6
ns, us = 1.0, 1e3


class TransmonGate(System):
    def __init__(self, N = 5, Ec=315 * MHz, Ej=16 * GHz, ng = 0.0, kappa = 10 * kHz, T = 15 * ns, T0 = 3 * ns, num_charge=300):
        self.N = N
        self.Ec = Ec
        self.Ej = Ej
        self.ng = ng
        self.kappa = kappa
        self.num_charge = num_charge
        self.T0 = T0
        self.T = T
        self.eps_0 = pi / (self.int_gaussian(T) - T * self.gaussian(T))

        self.diagonalize_transmon()

    def diagonalize_transmon(self):
        """diagonalize the transmon Hamiltonian"""
        # charge basis dimension
        N_charge = 2 * self.num_charge + 1

        # charge operator
        charge = torch.arange(-self.num_charge, self.num_charge + 1)
        charge = torch.diag(charge) - torch.eye(N_charge) * self.ng

        # flux operator
        ones = torch.ones(N_charge - 1)
        cosphi = 0.5 * (torch.diag(ones, 1) + torch.diag(ones, -1))

        # static transmon hamiltonian
        H0 = 4 * self.Ec * charge @ charge - self.Ej * cosphi

        # diagonalize H0 and truncate
        evals, evecs = torch.linalg.eigh(H0)
        self.H0 = torch.diag(evals[: self.N] - evals[0]).to(torch.complex64)

        # transmon frequency
        self.omega_t = self.H0[1, 1].real.item()

        # get charge operator in truncated basis
        U = evecs[:, : self.N]  # change of basis matrix
        self.charge = (U.mH @ charge @ U).to(torch.complex64)

    def gaussian(self, t):
        return exp(-(t - 0.5 * self.T) ** 2 / (2 * self.T0 ** 2))

    def int_gaussian(self, t):
        return sqrt(2 * pi) * self.T0 * erf(t / (2 * sqrt(2) * self.T0))

    def eps(self, t):
        return self.eps_0 * (self.gaussian(t) - self.gaussian(0))

    def H(self, t):
        return self.H0 + self.eps(t) * cos(self.omega_t * t) * self.charge

    @property
    def jump_ops(self):
        return [sqrt(self.kappa) * torch.triu(self.charge)]

    @property
    def psi0(self):
        return dq.fock(self.N, 0)

    @property
    def tsave(self):
        return torch.linspace(0, self.T, 101)

    def to(self, dtype, device):
        super().to(dtype, device)
        self.H0 = self.H0.to(dtype, device)
        self.charge = self.charge.to(dtype, device)
