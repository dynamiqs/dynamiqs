"""Step 11 - is the residual 2nd copy from re-wrapping dq.mesolve in an outer jit,
or is it really there when dq.mesolve is called exactly as it calls itself?

Every memory_analysis() measurement in Steps 7-9 used a pattern like:

    def f(rho0):
        return dq.mesolve(H, Ls, rho0, ts, ...)
    jax.jit(f).lower(rho0).compile().memory_analysis()

But dq.mesolve's own top level (mesolve.py) already does exactly this:

    f = jax.jit(_vectorized_mesolve, static_argnames=(...))
    return f(H, Ls, rho0, tsave, exp_ops, method, gradient, options)

Wrapping THAT again in an outer jax.jit does not add a second compiled unit -- JAX
traces through an already-jitted call and inlines it, so "jit(f)" and dq.mesolve's own
"f" collapse into ONE compiled program. That single program is not the one that runs
when a script calls dq.mesolve(...) directly (no user jit at all): there, dq.mesolve's
own internal jax.jit(_vectorized_mesolve) IS the top-level, externally-visible
boundary, and its outputs must be genuine host-visible buffers, not values a further
jit is free to fuse away.

This compiles _vectorized_mesolve EXACTLY as dq.mesolve calls it -- same static
argnames, same argument list, no additional outer jit -- and reads memory_analysis()
on that. If this reproduces ~2.045 R (not ~1.0 R), the extra copy is a top-level jit
output-materialisation cost, not anything left in the error-check patch.
"""

import jax

jax.config.update("jax_enable_x64", True)

import dynamiqs as dq  # noqa: E402
import equinox as eqx  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from diffrax._solution import is_okay  # noqa: E402
from dynamiqs.integrators.apis.mesolve import _vectorized_mesolve  # noqa: E402
from dynamiqs.integrators.core.diffrax_integrator import DiffraxIntegrator  # noqa: E402
from dynamiqs.options import Options  # noqa: E402
from dynamiqs.integrators._utils import astimeqarray  # noqa: E402
from dynamiqs.qarrays.utils import asqarray  # noqa: E402
from typing import cast  # noqa: E402
import diffrax as dx  # noqa: E402

MIB = 1024**2
N = 64
R = 16 * N * N
ALPHA, K2, K1, DELTA, T_FINAL = 2.0, 1.0, 0.05, 0.2, 0.1


def diffeqsolve_no_throw(self, t0, t1, y0, saveat, event=None, dtmax=None):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', UserWarning)
        stepsize_controller = self.stepsize_controller
        if dtmax is not None:
            get_dtmax = (
                (lambda c: c.controller.dtmax)
                if isinstance(stepsize_controller, dx.ClipStepSizeController)
                else (lambda c: c.dtmax)
            )
            stepsize_controller = eqx.tree_at(
                get_dtmax, stepsize_controller, dtmax, is_leaf=lambda x: x is None
            )
        return dx.diffeqsolve(
            self.terms, self.diffrax_solver, t0=t0, t1=t1, dt0=self.dt0, y0=y0,
            saveat=saveat, stepsize_controller=stepsize_controller,
            adjoint=self.adjoint, event=event, max_steps=self.max_steps,
            throw=False,
            progress_meter=self.options.progress_meter.to_diffrax(),
        )


def run_patched(self):
    fn = lambda t, y, args: self.save(y)  # noqa: ARG005
    subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)
    subsaveat_b = dx.SubSaveAt(t1=True)
    saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])
    solution = self.diffeqsolve(self.t0, self.t1, self.y0, saveat)
    ys = cast(tuple, solution.ys)
    saved = self.postprocess_saved(*ys)
    result = self.result(saved, infos=self.infos(solution.stats))
    failed = jnp.invert(is_okay(solution.result))
    result = eqx.tree_at(
        lambda r: r.infos.nsteps, result,
        solution.result.error_if(result.infos.nsteps, failed),
    )
    return result


PATCH = None  # "none" | "patched"


def build():
    a = dq.destroy(N)
    H = DELTA * dq.dag(a) @ a
    Ls = [np.sqrt(K2) * (a @ a - ALPHA**2 * dq.eye(N)), np.sqrt(K1) * a]
    return H, Ls, dq.fock_dm(N, 0).asdense()


def compile_exactly_like_mesolve(nt, patched):
    """Reproduces mesolve.py's own top-level jit call, verbatim, no extra wrapper."""
    if patched:
        DiffraxIntegrator.diffeqsolve = diffeqsolve_no_throw
        DiffraxIntegrator.run = run_patched
    else:
        DiffraxIntegrator.diffeqsolve = _ORIG_DIFFEQSOLVE
        DiffraxIntegrator.run = _ORIG_RUN

    import jax as _jax
    _jax.clear_caches()   # avoid reusing a trace from the OTHER variant at the same nt
    H, Ls, rho0 = build()
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in Ls]
    rho0 = asqarray(rho0)
    ts = jnp.linspace(0.0, T_FINAL, nt)
    options = Options(save_states=True).initialise()
    method = dq.method.Tsit5(rtol=1e-6, atol=1e-6, max_steps=100_000)

    # exactly mesolve.py's own dispatch: no DiffusiveMonteCarlo/JumpMonteCarlo branch
    f = jax.jit(_vectorized_mesolve, static_argnames=('gradient', 'options'))
    return f.lower(H, Ls, rho0, ts, None, method, None, options).compile()


_ORIG_DIFFEQSOLVE = DiffraxIntegrator.diffeqsolve
_ORIG_RUN = DiffraxIntegrator.run

for patched, label in ((False, "baseline (throw=True, as shipped)"),
                       (True, "patched (throw=False + scalar error_if)")):
    print(f"\n### {label}   -- compiled EXACTLY as mesolve.py's own top-level jit, "
          f"NO extra outer jit", flush=True)
    xs, outs, temps = [], [], []
    for nt in (10, 20, 40, 80, 160):
        m = compile_exactly_like_mesolve(nt, patched).memory_analysis()
        xs.append(nt)
        outs.append(m.output_size_in_bytes)
        temps.append(m.temp_size_in_bytes)
        print(f"  ntsave={nt:>4}  output={m.output_size_in_bytes:>12,}  "
              f"temp={m.temp_size_in_bytes:>12,}  total/traj="
              f"{(m.output_size_in_bytes + m.temp_size_in_bytes) / (nt * R):.4f}",
              flush=True)
    x = np.array(xs, float)
    A = np.column_stack([np.ones_like(x), x])
    for name, y in (("output", outs), ("temp", temps),
                    ("total", np.array(outs) + np.array(temps))):
        b, *_ = np.linalg.lstsq(A, np.array(y, float), rcond=None)
        print(f"  {name:<7s} = {b[0]:>12,.0f} B + {b[1] / R:>7.4f} R per saved state",
              flush=True)

DiffraxIntegrator.diffeqsolve = _ORIG_DIFFEQSOLVE
DiffraxIntegrator.run = _ORIG_RUN

# --- dump mode: one module, patched, with HLO dumped for inspection ---
import os as _os
if _os.environ.get("STEP11_DUMP"):
    DUMP = _os.environ["STEP11_DUMP"]
    _os.environ["XLA_FLAGS"] = f"--xla_dump_to={DUMP}"
    c = compile_exactly_like_mesolve(40, True)
    print(f"\ndumped to {DUMP}: {c.memory_analysis()}")

