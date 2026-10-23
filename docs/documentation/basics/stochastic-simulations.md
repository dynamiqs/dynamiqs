# Stochastic simulations

This tutorial explains how to simulate quantum systems that are continuously measured. The evolution of such a system, conditioned on the measurement record, is random: it is described by a **stochastic trajectory**. Dynamiqs provides four stochastic solvers, depending on the type of detector and on whether all the information leaking out of the system is measured.

```python
import jax
import jax.numpy as jnp
import optimistix as optx
from matplotlib import pyplot as plt
import dynamiqs as dq

dq.plot.mplstyle(dpi=150)  # set custom matplotlib style
```

## Unravelings of the master equation

Consider a system coupled to its environment through the loss channels $\{L_k\}$. On average, its state follows the Lindblad master equation (see the [Open systems](open-systems.md) tutorial). If the signal leaking through a loss channel is measured by a detector, we gain information on the system, and its state conditioned on the measurement record evolves according to a stochastic differential equation. There are many ways to measure the same loss channel, called **unravelings** of the master equation. Dynamiqs supports the two most common ones:

- **Jump unraveling**: a detector clicks at random times, e.g. a photodetector counting the photons emitted by the system. Each click applies the jump $\rho\to L_k\rho L_k^\dag/\tr{L_k\rho L_k^\dag}$ to the state. The measurement record of channel $k$ is the list of click times, with a click occurring during $\dt$ with probability
    $$
        \mathbb{P}[\dd N_k = 1] = \left(\theta_k + \eta_k\tr{L_k\rho L_k^\dag}\right)\dt.
    $$
- **Diffusive unraveling**: a detector outputs a continuous noisy signal, e.g. the current of a homodyne detector. The state evolves continuously under the action of a Gaussian noise. The measurement record of channel $k$ is the signal
    $$
        \dd Y_k = \sqrt{\eta_k}\tr{(L_k+L_k^\dag)\rho}\dt + \dd W_k,
    $$
    where $\dd W_k$ are independent Wiener processes.

In these expressions, $0\leq\eta_k\leq1$ is the **efficiency** of the detector measuring channel $k$, and $\theta_k\geq0$ is the **dark count rate** of a jump detector (the rate of clicks unrelated to the system). A channel with $\eta_k=0$ is not measured at all: it is a purely dissipative loss channel.

If all channels are measured with perfect efficiency ($\eta_k=1$, $\theta_k=0$), no information is lost and an initially pure state stays pure. Its evolution is then described by a **stochastic Schrödinger equation** (SSE) for the state vector $\ket{\psi(t)}$. Otherwise, the state must be described by a density matrix $\rho(t)$ which follows a **stochastic master equation** (SME). Simulating an SSE is cheaper than simulating an SME, because a state vector has $n$ coefficients instead of $n^2$ for a density matrix. The four solvers are:

| | Jump detector | Diffusive detector |
| --- | --- | --- |
| **SSE**: perfect detection, state vector | [`dq.jssesolve()`][dynamiqs.jssesolve] | [`dq.dssesolve()`][dynamiqs.dssesolve] |
| **SME**: finite efficiency, density matrix | [`dq.jsmesolve()`][dynamiqs.jsmesolve] | [`dq.dsmesolve()`][dynamiqs.dsmesolve] |

The exact equations solved are given in the API documentation of each solver.

## Simulating trajectories

The stochastic solvers take the same arguments as [`dq.mesolve()`][dynamiqs.mesolve], with a few differences:

- `keys`: a list of [PRNG keys](https://docs.jax.dev/en/latest/random-numbers.html), one per trajectory, used to sample the noise. The number of keys sets the number of trajectories, and the same keys always give the same trajectories.
- `method`: there is no default method, you must choose one among the methods supported by each solver:

    | Solver | Supported methods |
    | --- | --- |
    | [`dq.jssesolve()`][dynamiqs.jssesolve] | [`EulerJump`][dynamiqs.method.EulerJump], [`Event`][dynamiqs.method.Event] |
    | [`dq.jsmesolve()`][dynamiqs.jsmesolve] | [`EulerJump`][dynamiqs.method.EulerJump] |
    | [`dq.dssesolve()`][dynamiqs.dssesolve] | [`EulerMaruyama`][dynamiqs.method.EulerMaruyama], [`Rouchon1`][dynamiqs.method.Rouchon1] |
    | [`dq.dsmesolve()`][dynamiqs.dsmesolve] | [`EulerMaruyama`][dynamiqs.method.EulerMaruyama], [`Rouchon1`][dynamiqs.method.Rouchon1] |

    [`EulerJump`][dynamiqs.method.EulerJump], [`EulerMaruyama`][dynamiqs.method.EulerMaruyama] and [`Rouchon1`][dynamiqs.method.Rouchon1] are fixed step size methods. [`Rouchon1`][dynamiqs.method.Rouchon1] writes each step as a Kraus map, which keeps the density matrix positive. [`Event`][dynamiqs.method.Event] samples the click times exactly, by integrating the evolution between clicks with an adaptive step size ODE method and stopping at the next click. With an adaptive method, you must control the precision of the click times by setting either a maximum step size `dtmax` or a `root_finder` that refines each click time.
- `thetas` and `etas` (SME solvers only): the dark count rate (jump SME only) and the efficiency of each loss channel, with one value per jump operator.

!!! Warning "Save times"
    The fixed step size methods only support linearly spaced `tsave` with values that are exact multiples of the step size `dt`. Moreover, to JIT-compile a function that calls a stochastic solver with these methods, `tsave` must be passed as a tuple.

### Jump trajectories

Let us consider a qubit starting in $\ket{g}$, driven with Rabi frequency $\Omega$ and decaying with rate $\gamma$, whose emitted photons are counted by a perfect photodetector:
$$
    H = \frac{\Omega}{2}\sigma_x, \qquad L = \sqrt{\gamma}\,\sigma_-.
$$
We simulate 500 trajectories with [`dq.jssesolve()`][dynamiqs.jssesolve] and the [`Event`][dynamiqs.method.Event] method, using a Newton root finder from the [Optimistix](https://docs.kidger.site/optimistix/) library to locate the click times:

```python
# define Hamiltonian, jump operators, initial state
omega, gamma = 3.0, 1.0
H = 0.5 * omega * dq.sigmax()
jump_ops = [jnp.sqrt(gamma) * dq.sigmam()]
psi0 = dq.ground()

# define save times and expectation operators
tsave = jnp.linspace(0.0, 10.0, 201)
exp_ops = [dq.sigmaz()]

# define one PRNG key per trajectory
ntrajs = 500
keys = jax.random.split(jax.random.key(42), ntrajs)

# simulate trajectories
method = dq.method.Event(root_finder=optx.Newton(rtol=1e-5, atol=1e-5))
result = dq.jssesolve(
    H, jump_ops, psi0, tsave, keys, exp_ops=exp_ops, method=method, nmaxclick=100
)
```

The result is a [`dq.JSSESolveResult`][dynamiqs.JSSESolveResult]. It holds the states and expectation values of each trajectory, and the times at which the detector clicked:

```pycon
>>> result.states.shape  # (ntrajs, ntsave, n, 1)
(500, 201, 2, 1)
>>> result.expects.shape  # (ntrajs, len(exp_ops), ntsave)
(500, 1, 201)
>>> result.clicktimes.shape  # (ntrajs, len(jump_ops), nmaxclick)
(500, 1, 100)
>>> result.nclicks.shape  # (ntrajs, len(jump_ops))
(500, 1)
```

The click times are stored in a fixed-size array padded with `jnp.nan`, whose size is set by the `nmaxclick` argument. Choose it larger than the maximum number of clicks you expect in a trajectory. The number of clicks of each trajectory is given by `result.nclicks`.

Each trajectory undergoes Rabi oscillations, interrupted at each click by a jump to the ground state. Averaging the trajectories recovers the solution of the Lindblad master equation:

```python
# individual trajectories
for i in range(3):
    plt.plot(tsave, result.expects[i, 0].real, lw=1.5, label=f'Trajectory {i + 1}')

# average of the trajectories
mean_sz = result.mean_expects()[0].real
plt.plot(tsave, mean_sz, color='black', label='Average')

# Lindblad solution
result_me = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, progress_meter=False)
plt.plot(tsave, result_me.expects[0].real, 'k--', label='Lindblad')

plt.gca().set(xlabel=r'time $t$', ylabel=r'$\langle\sigma_z\rangle$', ylim=(-1.05, 1.6))
plt.legend(ncol=3, loc='upper center')
```

<!-- invisible-code-block: python
renderfig('stochastic-jump-qubit-trajs')
-->

![stochastic-jump-qubit-trajs](../../figs_docs/stochastic-jump-qubit-trajs.png){.fig}

### Diffusive trajectories

Let us now monitor the same qubit with a homodyne detector of efficiency $\eta=0.8$, and add a pure dephasing channel $L_\varphi = \sqrt{\gamma_\varphi/2}\,\sigma_z$ which is not measured ($\eta_\varphi=0$). Information is lost, so we use [`dq.dsmesolve()`][dynamiqs.dsmesolve], which takes the efficiency of each loss channel:

```python
# define jump operators and their measurement efficiency
gamma_phi = 0.5
jump_ops = [jnp.sqrt(gamma) * dq.sigmam(), jnp.sqrt(gamma_phi / 2) * dq.sigmaz()]
etas = [0.8, 0.0]

# simulate 100 trajectories
keys = jax.random.split(jax.random.key(42), 100)
tsave = jnp.linspace(0.0, 5.0, 51)
method = dq.method.Rouchon1(dt=1e-3)
result = dq.dsmesolve(
    H, jump_ops, etas, psi0, tsave, keys, exp_ops=exp_ops, method=method
)
```

The result is a [`dq.DSMESolveResult`][dynamiqs.DSMESolveResult]. The states are now density matrices, and the measurement record of each measured channel is saved in `result.measurements`:

```pycon
>>> result.states.shape  # (ntrajs, ntsave, n, n)
(100, 51, 2, 2)
>>> result.measurements.shape  # (ntrajs, nLm, ntsave - 1)
(100, 1, 50)
```

Only the channels with non-zero efficiency are returned, so `nLm=1` here. The measurements are time-averaged over each interval $[t_n, t_{n+1}[$ defined by `tsave`:
$$
    I_k(t_n, t_{n+1}) = \frac{1}{t_{n+1}-t_n}\int_{t_n}^{t_{n+1}} \dd Y_k(t),
$$
so there are `len(tsave) - 1` values per trajectory and per measured channel.

## Results

The results of the stochastic solvers are [`dq.JSSESolveResult`][dynamiqs.JSSESolveResult], [`dq.JSMESolveResult`][dynamiqs.JSMESolveResult], [`dq.DSSESolveResult`][dynamiqs.DSSESolveResult] and [`dq.DSMESolveResult`][dynamiqs.DSMESolveResult]. Their main attributes are:

| Attribute | Shape | Description |
| --- | --- | --- |
| `states` | _(..., ntrajs, nsave, n, 1)_ or _(..., ntrajs, nsave, n, n)_ | State of each trajectory. |
| `expects` | _(..., ntrajs, len(exp_ops), ntsave)_ | Expectation values of each trajectory. |
| `clicktimes` (jump) | _(..., ntrajs, nLm, nmaxclick)_ | Click times, padded with `jnp.nan`. |
| `nclicks` (jump) | _(..., ntrajs, nLm)_ | Number of clicks. |
| `measurements` (diffusive) | _(..., ntrajs, nLm, ntsave - 1)_ | Time-averaged measurements. |
| `mean_states()` | _(..., nsave, n, n)_ | Density matrix averaged over the trajectories. |
| `mean_expects()` | _(..., len(exp_ops), ntsave)_ | Expectation values averaged over the trajectories. |

where `nLm` is the number of measured loss channels (all of them for the SSE solvers), and `...` are the batch dimensions. The stochastic solvers support batching like the other solvers (see the [Batching simulations](batching-simulations.md) tutorial), and the trajectories axis comes right after the batch dimensions. Each batch element samples different trajectories from the same keys. For example, to run the previous simulation for three Rabi frequencies:

```pycon
>>> H = [0.5 * omega * dq.sigmax() for omega in [1.0, 2.0, 3.0]]
>>> result = dq.dsmesolve(
...     H, jump_ops, etas, psi0, tsave, keys, exp_ops=exp_ops, method=method
... )
>>> result.states.shape  # (3, ntrajs, ntsave, n, n)
(3, 100, 51, 2, 2)
>>> result.mean_expects().shape  # (3, len(exp_ops), ntsave)
(3, 1, 51)
```

## Recovering the Lindblad evolution

Averaging the trajectories over the measurement outcomes, with `result.mean_states()` or `result.mean_expects()`, recovers the solution of the Lindblad master equation, with a statistical error that decreases as $1/\sqrt{\text{ntrajs}}$. If you are only interested in this average, and not in the individual trajectories or the measurement record, use [`dq.mesolve()`][dynamiqs.mesolve] with the [`JumpMonteCarlo`][dynamiqs.method.JumpMonteCarlo] or [`DiffusiveMonteCarlo`][dynamiqs.method.DiffusiveMonteCarlo] method. They sample trajectories of the jump or diffusive SSE and return only the averaged states and expectation values, in a [`dq.MESolveResult`][dynamiqs.MESolveResult]:

```python
H = 0.5 * omega * dq.sigmax()
method = dq.method.DiffusiveMonteCarlo(keys, dq.method.Rouchon1(dt=1e-3))
result = dq.mesolve(
    H, jump_ops, psi0, tsave, exp_ops=exp_ops, method=method, progress_meter=False
)
```

Sampling trajectories of state vectors can be cheaper than integrating the master equation for large Hilbert spaces, see the [Open systems](open-systems.md#sampling-trajectories) tutorial.

## Simulating the measurement record only

The measurement record of an imperfect detector has the same statistics as the record of a perfect detector degraded afterwards:

- for a jump detector, each click is kept with probability $\eta_k$, and false clicks are added at the dark count rate $\theta_k$,
- for a diffusive detector, $\dd Y_k = \sqrt{\eta_k}\,\dd\tilde Y_k + \sqrt{1-\eta_k}\,\dd W_k$, where $\dd\tilde Y_k$ is the perfect record and $\dd W_k$ is an independent Wiener process.

If you are only interested in the measurement record, you can thus simulate the cheaper SSE, and post-process its record with [`dq.clicktimes_sse_to_sme()`][dynamiqs.clicktimes_sse_to_sme] or [`dq.measurements_sse_to_sme()`][dynamiqs.measurements_sse_to_sme]. For the previous diffusive example:

```pycon
>>> method = dq.method.Rouchon1(dt=1e-3)
>>> result_sse = dq.dssesolve(H, jump_ops, psi0, tsave, keys, method=method)
>>> result_sse.measurements.shape  # all channels are measured
(100, 2, 50)
>>> key = jax.random.key(0)  # PRNG key for the added noise
>>> measurements = dq.measurements_sse_to_sme(result_sse.measurements, tsave, etas, key)
>>> measurements.shape  # same statistics as the record of dq.dsmesolve()
(100, 1, 50)
```

This gives a significant speedup for large systems. Note however that the states of `result_sse` are conditioned on the record of the perfect detector: they are not the states conditioned on the degraded record, which require solving the SME.

## Going further

The [Continuous jump measurement](../advanced_examples/continuous-jump-measurement.md) and [Continuous diffusive measurement](../advanced_examples/continuous-diffusive-measurement.md) examples study the trajectories of monitored qubits and oscillators in more depth, including the effect of imperfect detection. The API documentation of each solver gives the equations solved and all the available options.
