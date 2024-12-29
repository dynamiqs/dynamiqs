# Continuous measurement of a qubit

!!! Warning "Work in progress."
    This tutorial is under construction, this is a draft version.

In this example, we will simulate the stochastic trajectories from the continuous diffusive measurement of a qubit.

We consider a qubit starting from $\ket{\psi_0}=\ket+_x=(\ket0+\ket1)/\sqrt2$, with no unitary dynamic $H=0$ and a single loss operator $L=\sigma_z$ which is continuously measured by a diffusive detector with efficiency $\eta$. We expect stochastic trajectories to project the qubit onto either of the $\sigma_z$ eigenstates.

```python
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import dynamiqs as dq

dq.plot.mplstyle(dpi=150)  # set custom matplotlib style
```

## Simulation

```python
# define Hamiltonian, jump operators, efficiencies, initial state
H = dq.zero(2)
jump_ops = [dq.sigmaz()]
etas = [1.0]
psi0 = (dq.ground() + dq.excited()).unit()

# define save times
nbins = 100
delta_t = 1/nbins
tsave = np.linspace(0, 1.0, nbins+1)

# define a certain number of PRNG key, one for each trajectory
key = jax.random.PRNGKey(20)
ntrajs = 5
keys = jax.random.split(key, ntrajs)

# simulate trajectories
solver = dq.solver.EulerMaruyama(dt=1e-3)
result = dq.dsmesolve(H, jump_ops, etas, psi0, tsave, keys, solver)
print(result)
```

```text title="Output"
==== DSMESolveResult ====
Solver       : EulerMaruyama
Infos        : 1000 steps | infos shape (5,)
States       : QArray complex64 (5, 101, 2, 2) | 15.8 Kb
Measurements : Array float32 (5, 1, 100) | 2.0 Kb
```

```pycon
>>> result.states.shape  # (ntrajs, ntsave, n, n)
(5, 101, 2, 2)
>>> result.measurements.shape  # (ntrajs, nLm, ntsave-1)
(5, 1, 100)
```

## Individual trajectories

```python
Iks = result.measurements[:, 0]
```

```python
for Ik in Iks:
    plt.plot(tsave[:-1], Ik, lw=1.5)
    plt.gca().set(
        title=rf'Simulated measurements for 5 trajectories',
        xlabel=r'$t$',
        ylabel=r'$I^{[t,t+\Delta t)}$',
    )

renderfig('monitored-qubit-trajs')
```

![plot_monitored_qubit_trajs](../../figs_docs/monitored-qubit-trajs.png){.fig}

## Cumulative measurements

```python
for Ik in Iks:
    cumsum_Ik = jnp.cumsum(Ik)
    plt.plot(tsave[1:], cumsum_Ik, lw=1.5)
    plt.axhline(0, ls='--', lw=1.0, color='gray')
    plt.gca().set(
        title=r'Integral of the measurement record',
        xlabel=r'$t$',
        ylabel=r'$Y_t=\int_0^t \mathrm{d}Y_s$',
    )

renderfig('monitored-qubit-cumsum-trajs')
```

![plot_monitored_qubit_cumsum_trajs](../../figs_docs/monitored-qubit-cumsum-trajs.png){.fig}

## Projection onto $\sigma_z$ eigenstate

```python
expects_all = dq.expect(dq.sigmaz(), result.states).real

for expects in expects_all:
    plt.plot(tsave, expects, lw=1.5)
    plt.axhline(-1, ls='--', lw=1.0, color='gray')
    plt.axhline(1, ls='--', lw=1.0, color='gray')
    plt.gca().set(
        title=r'Time evolution of $\sigma_z$ expectation value',
        xlabel=r'$t$',
        ylabel=r'$\mathrm{Tr}[\sigma_z\rho_t]$',
        ylim=(-1.1, 1.1),
    )

renderfig('monitored-qubit-sigmaz')
```

![plot_monitored_qubit_sigmaz](../../figs_docs/monitored-qubit-sigmaz.png){.fig}
