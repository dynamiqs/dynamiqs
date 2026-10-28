# Continuous diffusive measurement

In this example, we simulate stochastic trajectories of quantum systems that are continuously measured by a diffusive detector. We explain how to use [`dq.dssesolve()`][dynamiqs.dssesolve] to simulate trajectories modelled by the diffusive SSE, and [`dq.dsmesolve()`][dynamiqs.dsmesolve] to simulate trajectories modelled by the diffusive SME. For an introduction to the stochastic solvers, see the [Stochastic simulations](../basics/stochastic-simulations.md) tutorial.

```python
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import dynamiqs as dq

dq.plot.mplstyle(dpi=150)  # set custom matplotlib style
```

## Qubit

We consider a qubit starting from $\ket{\psi_0}=\ket+_x=(\ket0+\ket1)/\sqrt2$, with no unitary dynamics $H=0$ and a single jump operator $L=\sigma_z$ which is continuously measured by a diffusive detector. This realizes a weak continuous measurement of $\sigma_z$, so we expect each stochastic trajectory to be progressively projected onto one of the $\sigma_z$ eigenstates.

Let's begin with a perfectly efficient detector. We start with a pure state, and we measure the jump operator with perfect efficiency, so we don't lose any information about the system's state. As a result, it remains pure at all times. We use [`dq.dssesolve()`][dynamiqs.dssesolve] to simulate these quantum trajectories.

### Simulation

```python
# define Hamiltonian, jump operators, initial state
H = dq.zeros(2)
jump_ops = [dq.sigmaz()]
psi0 = (dq.ground() + dq.excited()).unit()

# define save times
tsave = np.linspace(0, 1.0, 101)
delta_t = tsave[1] - tsave[0]

# define a certain number of PRNG keys, one for each trajectory
key = jax.random.key(20)
ntrajs = 5
keys = jax.random.split(key, ntrajs)

# simulate trajectories
method = dq.method.EulerMaruyama(dt=1e-3)
result = dq.dssesolve(H, jump_ops, psi0, tsave, keys, method=method)
```

```pycon
>>> print(result)
==== DSSESolveResult ====
Method       : EulerMaruyama
Infos        : 1000 steps | infos shape (5,)
States       : QArray complex64 (5, 101, 2, 1) | 7.9 Kb
Measurements : Array float32 (5, 1, 100) | 2.0 Kb
>>> result.states.shape  # (ntrajs, ntsave, n, 1)
(5, 101, 2, 1)
>>> result.measurements.shape  # (ntrajs, len(jump_ops), ntsave-1)
(5, 1, 100)
```

The measurement record of each detector is returned as the time-averaged current $I^{[t,t+\Delta t)} = (Y_{t+\Delta t} - Y_t)/\Delta t$ over each interval of `tsave`, so there are `ntsave-1` values per detector and per trajectory. Here, the record is defined by $\dd Y_t = 2\braket{\sigma_z}_t \dt + \dd W_t$, since $L + L^\dagger = 2\sigma_z$.

### Individual trajectories

```python
Iks = result.measurements[:, 0]
```

```python
for Ik in Iks:
    plt.plot(tsave[:-1] + delta_t / 2, Ik, lw=1.5)

plt.gca().set(
    title=f'Simulated measurements for {ntrajs} trajectories',
    xlabel=r'$t$',
    ylabel=r'$I^{[t,t+\Delta t)}$',
)
```

<!-- invisible-code-block: python
renderfig('diffusive-monitored-qubit-trajs')
-->

![plot_diffusive_monitored_qubit_trajs](../../figs_docs/diffusive-monitored-qubit-trajs.png){.fig}

Each current is plotted at the middle of its averaging interval. The individual currents look like pure noise: the white noise $\dd W_t/\dt$ averaged over $\Delta t = 0.01$ has a standard deviation $1/\sqrt{\Delta t} = 10$, much larger than the signal $2\braket{\sigma_z}_t$, which is bounded by $2$ in absolute value.

### Cumulative measurements

The signal becomes visible by integrating the record, $Y_t = \sum I^{[s,s+\Delta t)}\,\Delta t$, which averages out the noise.

```python
for Ik in Iks:
    Yt = jnp.cumsum(Ik) * delta_t
    plt.plot(tsave[1:], Yt, lw=1.5)

plt.plot(tsave, 2 * tsave, ls='--', lw=1.0, color='gray')
plt.plot(tsave, -2 * tsave, ls='--', lw=1.0, color='gray')
plt.gca().set(
    title=r'Integral of the measurement record',
    xlabel=r'$t$',
    ylabel=r'$Y_t=\int_0^t \mathrm{d}Y_s$',
)
```

<!-- invisible-code-block: python
renderfig('monitored-qubit-cumsum-trajs')
-->

![plot_monitored_qubit_cumsum_trajs](../../figs_docs/monitored-qubit-cumsum-trajs.png){.fig}

Once a trajectory has been projected onto an eigenstate of $\sigma_z$, its integrated record drifts with slope $\pm2$ (dashed lines), on top of diffusive fluctuations of typical size $\sqrt t$. The sign of the slope tells which eigenstate the qubit ends up in.

### Projection onto $\sigma_z$ eigenstates

```python
expects_all = dq.expect(dq.sigmaz(), result.states).real

for expects in expects_all:
    plt.plot(tsave, expects, lw=1.5)

plt.axhline(-1, ls='--', lw=1.0, color='gray')
plt.axhline(1, ls='--', lw=1.0, color='gray')
plt.gca().set(
    title=r'Time evolution of $\sigma_z$ expectation value',
    xlabel=r'$t$',
    ylabel=r'$\langle \sigma_z \rangle_t=\langle \psi_t | \sigma_z | \psi_t \rangle$',
    ylim=(-1.1, 1.1),
)
```

<!-- invisible-code-block: python
renderfig('monitored-qubit-sigmaz')
-->

![plot_monitored_qubit_sigmaz](../../figs_docs/monitored-qubit-sigmaz.png){.fig}

All trajectories start at $\braket{\sigma_z}=0$ and diffuse towards $+1$ or $-1$, where most of them have settled by $t=1$: the measurement progressively projects the qubit onto one of the eigenstates of $\sigma_z$. Each outcome is obtained with probability $|\braket{\pm_z|\psi_0}|^2 = 1/2$, and once reached, an eigenstate is left unchanged by the measurement.

### Imperfect detection

If the detection is imperfect, the system state is a density matrix. We use [`dq.dsmesolve()`][dynamiqs.dsmesolve] to simulate these quantum trajectories, with a measurement efficiency $\eta$ for each jump operator. The record now reads $\dd Y_t = 2\sqrt\eta\braket{\sigma_z}_t \dt + \dd W_t$.

```python
# define efficiencies
etas = [0.2]

# simulate trajectories
result = dq.dsmesolve(H, jump_ops, etas, psi0, tsave, keys, method=method)
```

```pycon
>>> print(result)
==== DSMESolveResult ====
Method       : EulerMaruyama
Infos        : 1000 steps | infos shape (5,)
States       : QArray complex64 (5, 101, 2, 2) | 15.8 Kb
Measurements : Array float32 (5, 1, 100) | 2.0 Kb
>>> result.states.shape  # (ntrajs, ntsave, n, n)
(5, 101, 2, 2)
>>> result.measurements.shape  # (ntrajs, nLm, ntsave-1)
(5, 1, 100)
```

Here `nLm` is the number of measured jump operators, i.e. those with a non-zero efficiency: no measurement record is returned for a purely dissipative loss channel ($\eta=0$).

```python
expects_all = dq.expect(dq.sigmaz(), result.states).real

for expects in expects_all:
    plt.plot(tsave, expects, lw=1.5)

plt.axhline(-1, ls='--', lw=1.0, color='gray')
plt.axhline(1, ls='--', lw=1.0, color='gray')
plt.gca().set(
    title=r'Time evolution of $\sigma_z$ expectation value',
    xlabel=r'$t$',
    ylabel=r'$\langle \sigma_z \rangle_t = \mathrm{Tr}[\sigma_z\rho_t]$',
    ylim=(-1.1, 1.1),
)
```

<!-- invisible-code-block: python
renderfig('monitored-qubit-sigmaz-eta')
-->

![plot_monitored_qubit_sigmaz_eta](../../figs_docs/monitored-qubit-sigmaz-eta.png){.fig}

With $\eta=0.2$, the record carries less information about the qubit, so the projection onto the $\sigma_z$ eigenstates is slower: by $t=1$, most trajectories are still far from $\pm1$. Meanwhile, the unmeasured fraction of the channel dephases the qubit at the same rate as before, so each trajectory is now a mixed state, with purity $\mathrm{Tr}[\rho_t^2]<1$. Given enough time, the trajectories are still projected onto one of the eigenstates, which are left unchanged by both the measurement and the dephasing.

## Quantum harmonic oscillator

We consider a quantum harmonic oscillator starting from the coherent state $\ket{\alpha_0}$, with Hamiltonian $H=\omega a^\dagger a$ and a single loss channel $\sqrt\kappa a$ which is continuously measured by heterodyne detection with efficiency $\eta$. Heterodyne detection splits the output field into two diffusive detectors, which we model with the two jump operators $L_X=\sqrt{\kappa/2}\,a$ and $L_P=-i\sqrt{\kappa/2}\,a$, resulting in two diffusive measurement records $I_X$ and $I_P$. Defining the quadratures $X=(a+a^\dagger)/2$ and $P=(a-a^\dagger)/(2i)$, these records read
$$
    \dd Y_X = \sqrt{2\eta\kappa}\braket{X}_t \dt + \dd W_X, \qquad
    \dd Y_P = \sqrt{2\eta\kappa}\braket{P}_t \dt + \dd W_P.
$$

In this example the measurement backaction vanishes: because $a\ket\alpha=\alpha\ket\alpha$, a coherent state is left unchanged by the measurement, and every trajectory follows the same deterministic evolution $\ket{\alpha(t)}$ with $\alpha(t)=\alpha_0 e^{-\kappa t/2} e^{-i\omega t}$. Only the measurement records are random.

### Simulation

```python
# define Hamiltonian, jump operators, efficiencies, initial state
n = 16
a = dq.destroy(n)
kappa = 1.0
omega = 10.0
alpha0 = 2.0
H = omega * a.dag() @ a
jump_ops = [jnp.sqrt(kappa / 2) * a, jnp.sqrt(kappa / 2) * (-1j * a)]
etas = [1.0, 1.0]
psi0 = dq.coherent(n, alpha0)

# define save times
tsave = np.linspace(0, 1 / kappa, 101)
delta_t = tsave[1] - tsave[0]

# define a certain number of PRNG keys, one for each trajectory
key = jax.random.key(42)
ntrajs = 1000
keys = jax.random.split(key, ntrajs)

# simulate trajectories
method = dq.method.EulerMaruyama(dt=1e-3)
result = dq.dsmesolve(
    H, jump_ops, etas, psi0, tsave, keys, method=method, save_states=False
)
```

```pycon
>>> print(result)
==== DSMESolveResult ====
Method       : EulerMaruyama
Infos        : 1000 steps | infos shape (1000,)
States       : QArray complex64 (1000, 1, 16, 16) | 2.0 Mb
Measurements : Array float32 (1000, 2, 100) | 781.2 Kb
>>> result.measurements.shape  # (ntrajs, nLm, ntsave-1)
(1000, 2, 100)
```

### Individual trajectories

We rescale each record by $\sqrt{2\eta\kappa}$, so that its expectation value is the corresponding quadrature.

```python
Iks_x = result.measurements[:, 0] / np.sqrt(2 * etas[0] * kappa)
Iks_p = result.measurements[:, 1] / np.sqrt(2 * etas[1] * kappa)
tmid = tsave[:-1] + delta_t / 2  # middle of each averaging interval
```

```python
fig, axs = dq.plot.grid(2, 2, w=6, h=2, sharexy=True)
ax0, ax1 = list(axs)

for Ik_x in Iks_x[:3]:
    ax0.plot(tmid, Ik_x, lw=1.5)

ax0.set(
    title='Simulated measurements for 3 trajectories',
    ylabel=r'$I_X^{[t,t+\Delta t)}/\sqrt{2\eta\kappa}$',
)

for Ik_p in Iks_p[:3]:
    ax1.plot(tmid, Ik_p, lw=1.5)

ax1.set(xlabel=r'$t$', ylabel=r'$I_P^{[t,t+\Delta t)}/\sqrt{2\eta\kappa}$')
```

<!-- invisible-code-block: python
renderfig('monitored-oscillator-IxIp')
-->

![plot_monitored_oscillator_IxIp](../../figs_docs/monitored-oscillator-IxIp.png){.fig}

As for the qubit, the individual records are dominated by the measurement noise, of standard deviation $1/\sqrt{2\eta\kappa\Delta t}\approx7$ after rescaling, and the quadratures, of amplitude at most $|\alpha_0|=2$, cannot be read from a single trajectory at this time resolution.

### Averaged trajectories

```python
plt.figure()
plt.plot(tmid, Iks_x.mean(0), label=r'$\mathbb{E}[I_X]/\sqrt{2\eta\kappa}$')
plt.plot(tmid, Iks_p.mean(0), label=r'$\mathbb{E}[I_P]/\sqrt{2\eta\kappa}$')

alpha = alpha0 * jnp.exp(-kappa / 2 * tsave) * jnp.exp(-1j * omega * tsave)
plt.plot(tsave, alpha.real, label=r'$\mathrm{Tr}[X\rho_t]$', ls='--', color='gray')
plt.plot(tsave, alpha.imag, label=r'$\mathrm{Tr}[P\rho_t]$', ls='--', color='gray')

plt.gca().set(
    title=r'Measurement averaged over all trajectories and theory prediction',
    xlabel=r'$t$',
    ylim=(-2.5, 2.5),
)
plt.legend()
```

<!-- invisible-code-block: python
renderfig('monitored-oscillator-mean')
-->

![plot_monitored_oscillator_mean](../../figs_docs/monitored-oscillator-mean.png){.fig}

Averaging over the $1000$ trajectories reduces the noise by a factor $\sqrt{1000}\approx32$, and the averaged records follow the predicted quadratures $\braket{X}_t=\mathrm{Re}\,\alpha(t)$ and $\braket{P}_t=\mathrm{Im}\,\alpha(t)$: they rotate at frequency $\omega$ and decay at rate $\kappa/2$. The remaining fluctuations around the theory curves are of order $7/\sqrt{1000}\approx0.2$, as expected.
