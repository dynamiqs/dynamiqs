from __future__ import annotations

import diffrax as dx
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ..core.abstract_solver import SMESolver
from ..core.diffrax_solver import AdaptiveSolver, DiffraxSolver, EulerSolver
from ..result import Saved, SMESaved
from ..utils.utils.general import dag, trace, tracemm


# state for the diffrax solver for SMEs
class Y(eqx.Module):
    rho: Array
    dYt: Array


class MeasurementTerm(dx.ControlTerm):
    def prod(self, vf: dx.VF, control: dx.Control) -> dx.Y:
        dW = control
        rho = (vf.rho * dW[:, None, None]).sum(0)  # (n, n)
        return Y(rho, dW)


class SMEDiffraxSolver(DiffraxSolver, SMESolver):
    wiener: dx.VirtualBrownianTree

    def __init__(self, *args):
        # === pass all init arguments to `BaseSolver`
        super().__init__(*args)

        # === define save function to save measurement results
        fn = lambda t, y, args: y.dYt  # noqa: ARG005
        save_c = dx.SubSaveAt(ts=self.tmeas, fn=fn)  # save measurement results
        self.saveat.subs.append(save_c)

        # === define initial augmented state
        self.y0 = Y(self.y0, jnp.empty(len(self.etas)))

        # === define wiener process
        self.wiener = dx.VirtualBrownianTree(
            self.t0, self.t1, tol=1e-3, shape=(len(self.etas),), key=self.key
        )  # todo: fix hard-coded tol

    @property
    def terms(self) -> dx.AbstractTerm:
        # === define deterministic term
        # This contains everything before the "dt" in the SME:
        # - lindblad term for drho
        # - sqrt(eta) Tr[(L+Ld) @ rho] for dYt
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            # state
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            H = self.H(t)
            tmp = (-1j * H - 0.5 * LdL) @ y.rho + 0.5 * (Ls @ y.rho @ Lsd).sum(0)
            rho = tmp + dag(tmp)

            # signal
            Lms = jnp.stack([L(t) for L in self.Lms])
            tr_Lms_rho = tracemm(Lms, y.rho)
            dYt = jnp.sqrt(self.etas) * (tr_Lms_rho + tr_Lms_rho.conj()).real  # (nLm,)

            return Y(rho, dYt)

        lindblad_term = dx.ODETerm(vector_field)

        # === define stochastic term
        # This contains everything before the "dWt" in the SME:
        # - measurement backaction term for drho
        # - simply dWt for dYt
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            # \sqrt\eta (L @ rho + rho @ Ld - Tr[L @ rho + rho @ Ld] rho) dWt
            Lms = jnp.stack([L(t) for L in self.Lms])
            Lms_rho = Lms @ y.rho
            tmp = Lms_rho + dag(Lms_rho)
            tr = trace(tmp).real

            # state
            etas = self.etas[:, None, None]  # (nLm, n, n)
            tr = tr[:, None, None]  # (nLm, n, n)
            rho = jnp.sqrt(etas) * (tmp - tr * y.rho)  # (nLm, n, n)

            # signal
            dYt = jnp.empty(len(self.etas))

            return Y(rho, dYt)

        control = self.wiener
        measurement_term = MeasurementTerm(vector_field, control)

        # === combine and return both terms
        return dx.MultiTerm(lindblad_term, measurement_term)

    def solution_to_saved(self, ys: PyTree) -> Saved:
        # === collect and return results
        save_a, save_b, save_c = ys
        saved, ylast, integrated_dYt = save_a, save_b, save_c

        # Diffrax integrates the state from t0 to t1. In this case, the state is
        # (rho, dYt). So we recover the signal by simply diffing the resulting array.
        Isave = jnp.diff(integrated_dYt, axis=0)

        saved = SMESaved(saved.ysave, saved.Esave, saved.extra, Isave, self.wiener)
        return self.collect_saved(saved, ylast)


class SMEEuler(SMEDiffraxSolver, EulerSolver):
    pass


class SMEMilstein(SMEDiffraxSolver, AdaptiveSolver):
    diffrax_solver = dx.HalfSolver(dx.ItoMilstein())
