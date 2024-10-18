# Monitored qubit

First example to convince the SME-practionner that the solver is correct. Qubit starting from $\ket+$ with $H=0$ and monitored $L=\sigma_z$ (we expect diffusive projection onto one of $\sigma_z$ eigenvalues).

```python
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import dynamiqs as dq
```

```python
H = dq.zero(2)
psi0 = dq.unit(dq.basis(2, 0) + dq.basis(2, 1))
jump_ops = [1.0 * dq.sigmaz()]
etas = [1.0]
tsave = np.linspace(0, 1.0, 101)
key = jax.random.PRNGKey(42)
ntrajs = 6
keys = jax.random.split(key, ntrajs)

solver = dq.solver.Euler(dt=1e-2)
result = dq.dsmesolve(H, jump_ops, etas, psi0, tsave, keys, solver=solver)
print(result)
```

```text title="Output"
==== SMESolveResult ====
Solver       : Euler
Infos        : 100 steps | infos shape (6,)
States       : Array complex64 (6, 101, 2, 2) | 18.9 Kb
Measurements : Array float32 (6, 1, 100) | 2.3 Kb
```

```pycon
>>> result.states.shape  # (ntrajs, ntsave, n, n)
(6, 101, 2, 2)
>>> result.measurements.shape  # (ntrajs, nLm, ntsave-1)
(6, 1, 100)
```

```python
Iks_all = result.measurements[:, 0]
delta_t = tsave[1] - tsave[0]
```

```python
for Iks in Iks_all:
    plt.plot(tsave[:-1], Iks, lw=1.5)
    plt.gca().set(
        title=rf'Instantaneous signal (binned on $\Delta t={delta_t}$)',
        xlabel=r'$t$',
        ylabel=r'$I_t$',
        ylim=(-0.4, 0.4),
    )

renderfig('monitored-qubit-trajs')
```

![plot_monitored_qubit_trajs](/figs_docs/monitored-qubit-trajs.png){.fig}

```python
for Iks in Iks_all:
    cumsum_Iks = jnp.cumsum(Iks)
    plt.plot(tsave[1:], cumsum_Iks, lw=1.5)
    plt.axhline(0, ls='--', lw=1.0, color='gray')
    plt.gca().set(
        title=r'Time integrated signal',
        xlabel=r'$t$',
        ylabel=r'$\int_0^t \mathrm{d}Y_u$',
        ylim=(-4.5, 4.5),
    )

renderfig('monitored-qubit-cumsum-trajs')
```

![plot_monitored_qubit_cumsum_trajs](/figs_docs/monitored-qubit-cumsum-trajs.png){.fig}

```python
expects_all = dq.expect(dq.sigmaz(), result.states).real
```

```python
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

![plot_monitored_qubit_sigmaz](/figs_docs/monitored-qubit-sigmaz.png){.fig}


<!-- average like a mad man -->
