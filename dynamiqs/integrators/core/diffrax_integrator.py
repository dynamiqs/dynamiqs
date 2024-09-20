from __future__ import annotations

import warnings
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ...gradient import Autograd, CheckpointAutograd
from ...result import Saved, SMESolveSaved
from ...utils.quantum_utils.general import dag, trace, tracemm
from .abstract_integrator import BaseIntegrator, MEIntegrator, SMEIntegrator


class DiffraxIntegrator(BaseIntegrator):
    # Subclasses should implement:
    # - the attributes: stepsize_controller, dt0, max_steps, diffrax_solver, terms
    # - the methods: result, infos

    stepsize_controller: dx.AbstractVar[dx.AbstractStepSizeController]
    dt0: dx.AbstractVar[float | None]
    max_steps: dx.AbstractVar[int]
    diffrax_solver: dx.AbstractVar[dx.AbstractSolver]
    terms: dx.AbstractVar[dx.AbstractTerm]
    saveat: dx.SaveAt

    def __init__(self, *args):
        # pass all init arguments to `BaseIntegrator`
        super().__init__(*args)

        # prepare saveat argument
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        save_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
        save_b = dx.SubSaveAt(t1=True)  # save last state
        self.saveat = dx.SaveAt(subs=[save_a, save_b])

    def run(self) -> PyTree:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)
            # TODO: remove once https://github.com/patrick-kidger/diffrax/issues/445 is
            # closed
            warnings.simplefilter('ignore', FutureWarning)

            # === prepare adjoint argument
            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, CheckpointAutograd):
                adjoint = dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.DirectAdjoint()

            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.ts[-1],
                dt0=self.dt0,
                y0=self.y0,
                saveat=self.saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )

        # === collect and return results
        saved = self.solution_to_saved(solution.ys)
        return self.result(saved, infos=self.infos(solution.stats))

    def solution_to_saved(self, ys: PyTree) -> Saved:
        # === collect and return results
        save_a, save_b = ys
        saved, ylast = save_a, save_b
        return self.collect_saved(saved, ylast)

    @abstractmethod
    def infos(self, stats: dict[str, Array]) -> PyTree:
        pass


class FixedStepIntegrator(DiffraxIntegrator):
    # Subclasses should implement:
    # - the attributes: diffrax_solver, terms
    # - the methods: result

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    stepsize_controller: dx.AbstractStepSizeController = dx.ConstantStepSize()
    max_steps: int = 100_000  # TODO: fix hard-coded max_steps

    @property
    def dt0(self) -> float:
        return self.solver.dt

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(stats['num_steps'])


class EulerIntegrator(FixedStepIntegrator):
    diffrax_solver: dx.AbstractSolver = dx.Euler()


class AdaptiveStepIntegrator(DiffraxIntegrator):
    # Subclasses should implement:
    # - the attributes: diffrax_solver, terms
    # - the methods: result

    class Infos(eqx.Module):
        nsteps: Array
        naccepted: Array
        nrejected: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                return (
                    f'avg. {self.nsteps.mean():.1f} steps ({self.naccepted.mean():.1f}'
                    f' accepted, {self.nrejected.mean():.1f} rejected) | infos shape'
                    f' {self.nsteps.shape}'
                )
            return (
                f'{self.nsteps} steps ({self.naccepted} accepted,'
                f' {self.nrejected} rejected)'
            )

    dt0 = None

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
            jump_ts=self.discontinuity_ts,
        )

    @property
    def max_steps(self) -> int:
        return self.solver.max_steps

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(
            stats['num_steps'], stats['num_accepted_steps'], stats['num_rejected_steps']
        )


class Dopri5Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Dopri5()


class Dopri8Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Dopri8()


class Tsit5Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Tsit5()


class Kvaerno3Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Kvaerno3()


class Kvaerno5Integrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.Kvaerno5()


class MilsteinIntegrator(AdaptiveStepIntegrator):
    diffrax_solver = dx.HalfSolver(dx.ItoMilstein())


class SEDiffraxIntegrator(DiffraxIntegrator):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class MEDiffraxIntegrator(DiffraxIntegrator, MEIntegrator):
    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation for a single loss channel is:
        # (1) drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        # An alternative but similar equation is:
        # (2) drho/dt = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        # While (1) and (2) are equivalent assuming that rho is hermitian, they differ
        # once you take into account numerical errors.
        # Decomposing rho = rho_s + rho_a with Hermitian rho_s and anti-Hermitian rho_a,
        # we get that:
        #  - if rho evolves according to (1), both rho_s and rho_a also evolve
        #    according to (1);
        #  - if rho evolves according to (2), rho_s evolves closely to (1) up
        #    to a constant error that depends on rho_a (which is small up to numerical
        #    precision), while rho_a is strictly constant.
        # In practice, we still use (2) because it involves less matrix multiplications,
        # and is thus more efficient numerically with only a negligible numerical error
        # induced on the dynamics.

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            tmp = (-1j * self.H(t) - 0.5 * LdL) @ y + 0.5 * (Ls @ y @ Lsd).sum(0)
            return tmp + dag(tmp)

        return dx.ODETerm(vector_field)


# state for the diffrax solver for SMEs
class Y(eqx.Module):
    rho: Array
    dYt: Array


class MeasurementTerm(dx.ControlTerm):
    def prod(self, vf: dx.VF, control: dx.Control) -> dx.Y:
        dW = control
        rho = (vf.rho * dW[:, None, None]).sum(0)  # (n, n)
        return Y(rho, dW)


class SMEDiffraxIntegrator(DiffraxIntegrator, SMEIntegrator):
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
        def vector_field_deterministic(t, y, _):  # noqa: ANN001, ANN202
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

        lindblad_term = dx.ODETerm(vector_field_deterministic)

        # === define stochastic term
        # This contains everything before the "dWt" in the SME:
        # - measurement backaction term for drho
        # - simply dWt for dYt
        def vector_field_stochastic(t, y, _):  # noqa: ANN001, ANN202
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
            # we use jnp.empty because this quantity is ignored later, in prod() we
            # throw the stochastic part away because there is no signal-dependent term
            # in the stochastic part, it's just dYt = deterministic + dWt
            dYt = jnp.empty(len(self.etas))

            return Y(rho, dYt)

        control = self.wiener
        measurement_term = MeasurementTerm(vector_field_stochastic, control)

        # === combine and return both terms
        return dx.MultiTerm(lindblad_term, measurement_term)

    def solution_to_saved(self, ys: PyTree) -> Saved:
        # === collect and return results
        save_a, save_b, save_c = ys
        saved, ylast, integrated_dYt = save_a, save_b, save_c

        # Diffrax integrates the state from t0 to t1. In this case, the state is
        # (rho, dYt). So we recover the signal by simply diffing the resulting array.
        Jsave = jnp.diff(integrated_dYt, axis=0)

        saved = SMESolveSaved(saved.ysave, saved.Esave, saved.extra, Jsave)
        return self.collect_saved(saved, ylast)
