from __future__ import annotations

import warnings
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ...gradient import Autograd, CheckpointAutograd
from ...utils.quantum_utils.general import dag
from .abstract_integrator import BaseIntegrator, MEIntegrator


class DiffraxIntegrator(BaseIntegrator):
    # Subclasses should implement:
    # - the attributes: stepsize_controller, dt0, max_steps, diffrax_solver, terms
    # - the methods: result, infos

    stepsize_controller: dx.AbstractVar[dx.AbstractStepSizeController]
    dt0: dx.AbstractVar[float | None]
    max_steps: dx.AbstractVar[int]
    diffrax_solver: dx.AbstractVar[dx.AbstractSolver]
    terms: dx.AbstractVar[dx.AbstractTerm]

    def __init__(self, *args):
        # pass all init arguments to `BaseIntegrator`
        super().__init__(*args)

    def run(self) -> PyTree:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)
            # TODO: remove once https://github.com/patrick-kidger/diffrax/issues/445 is
            # closed
            warnings.simplefilter('ignore', FutureWarning)

            # === prepare diffrax arguments
            fn = lambda t, y, args: self.save(y)  # noqa: ARG005
            subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
            subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
            saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

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
                t1=self.t1,
                dt0=self.dt0,
                y0=self.y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )

        # === collect and return results
        saved = self.postprocess_saved(*solution.ys)
        return self.result(saved, infos=self.infos(solution.stats))

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
