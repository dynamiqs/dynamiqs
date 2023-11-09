from __future__ import annotations

from math import cos, exp, pi, sqrt

import torch
from scipy.special import erf
from torch import Tensor

import dynamiqs as dq
from dynamiqs.utils.tensor_types import get_cdtype

from .system import ClosedSystem, OpenSystem

# units
GHz, MHz, kHz = 2 * pi, 2 * pi * 1e-3, 2 * pi * 1e-6
ns, us = 1.0, 1e3


class TransmonGate(ClosedSystem):
    def __init__(
        self,
        N: int = 5,
        num_tslots: int = 100,
        Ec: float = 315 * MHz,
        Ej: float = 16 * GHz,
        ng: float = 0.0,
        kappa: float = 10 * kHz,
        T: float = 15 * ns,
        T0: float = 3 * ns,
        num_charge: int = 300,
    ):
        # register attributes
        self.N = N
        self.num_tslots = num_tslots
        self.Ec = Ec
        self.Ej = Ej
        self.ng = ng
        self.kappa = kappa
        self.T = T
        self.T0 = T0
        self.num_charge = num_charge

        # compute pulse amplitude
        self.eps_0 = pi / (self.gaussian_integ(T) - T * self.gaussian(T))

        # initialize diagonal transmon Hamiltonian
        self.transmon_hamiltonian()

    def transmon_hamiltonian(self):
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
        self.H0 = torch.diag(evals[: self.N] - evals[0]).to(get_cdtype())

        # transmon frequency
        self.omega_t = self.H0[1, 1].real.item()

        # get charge operator in truncated basis
        U = evecs[:, : self.N]  # change of basis matrix
        self.charge = (U.mH @ charge @ U).to(get_cdtype())

    def gaussian(self, t: float) -> float:
        return exp(-((t - 0.5 * self.T) ** 2) / (2 * self.T0**2))

    def gaussian_integ(self, t: float) -> float:
        return sqrt(2 * pi) * self.T0 * erf(t / (2 * sqrt(2) * self.T0))

    def eps(self, t: float) -> float:
        return self.eps_0 * (self.gaussian(t) - self.gaussian(0))

    def H(self, t: float) -> Tensor:
        return self.H0 + self.eps(t) * cos(self.omega_t * t) * self.charge

    @property
    def psi0(self) -> Tensor:
        return dq.fock(self.N, 0)

    @property
    def tsave(self) -> Tensor:
        return torch.linspace(0, self.T, self.num_tslots + 1)

    def to(self, dtype: torch.dtype, device: torch.device):
        super().to(dtype=dtype, device=device)
        self.H0 = self.H0.to(dtype=dtype, device=device)
        self.charge = self.charge.to(dtype=dtype, device=device)


class OpenTransmonGate(TransmonGate, OpenSystem):
    @property
    def jump_ops(self) -> list[Tensor]:
        return [sqrt(self.kappa) * torch.triu(self.charge)]
