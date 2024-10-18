# Monitored oscillator

Cavity starting from $\ket\alpha$ with $H=\Delta a^\dagger a$ and monitored $L_X = \sqrt\kappa a$ and $L_P = \sqrt\kappa (-ia)$ (we expect no measurement backaction and average signal trajectories following the state expectation values)

```python
# imports
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import dynamiqs as dq
```

```python
n = 16
a = dq.destroy(n)
H = 10 * dq.dag(a) @ a
psi0 = dq.coherent(n, 2.0)
jump_ops = [a, -1j * a]
etas = [1.0, 1.0]
tsave = np.linspace(0, 1.0, 101)
key = jax.random.PRNGKey(42)
ntrajs = 1000
keys = jax.random.split(key, ntrajs)

solver = dq.solver.Euler(dt=1e-3)
result = dq.dsmesolve(H, jump_ops, etas, psi0, tsave, keys, solver=solver)
print(result)
```

```text title="Output"
==== SMESolveResult ====
Solver       : Euler
Infos        : 1000 steps | infos shape (1000,)
States       : Array complex64 (1000, 101, 16, 16) | 197.3 Mb
Measurements : Array float32 (1000, 2, 100) | 781.2 Kb
```

```pycon
>>> result.states.shape  # (ntrajs, ntsave, n, n)
(1000, 101, 16, 16)
>>> result.measurements.shape  # (ntrajs, nLm, ntsave-1)
(1000, 2, 100)
```

```python
delta_t = tsave[1] - tsave[0]
Iks_x_all = result.measurements[:, 0] / delta_t / 2
Iks_p_all = result.measurements[:, 1] / delta_t / 2
```

```python
fig, axs = dq.plot.grid(2, 2, w=6, h=3, sharexy=True)
ax0, ax1 = list(axs)

for Iks_x in Iks_x_all[:3]:
    ax0.plot(tsave[:-1], Iks_x, lw=1.5)
    ax0.set(
        title=rf'Instantaneous $X$ quadrature signal for 3 trajectories (binned on $\Delta t={delta_t}$)',
        xlabel=r'$t$',
        ylabel=r'$I_t^X$',
        ylim=(-15, 15),
    )

for Iks_p in Iks_p_all[:3]:
    ax1.plot(tsave[:-1], Iks_p, lw=1.5)
    ax1.set(
        title=rf'Instantaneous $P$ quadrature signal for 3 trajectories (binned on $\Delta t={delta_t}$)',
        xlabel=r'$t$',
        ylabel=r'$I_t^P$',
        ylim=(-15, 15),
    )

renderfig('monitored-oscillator-IxIp1')
```

![plot_monitored_oscillator_IxIp1](/figs_docs/monitored-oscillator-IxIp1.png){.fig}

```python
plt.plot(tsave[1:], jnp.mean(Iks_x_all, axis=0), label=r'$\mathbb{E}[I_t^X]$')
plt.plot(tsave[1:], jnp.mean(Iks_p_all, axis=0), label=r'$\mathbb{E}[I_t^P]$')
plt.gca().set(
    title=r'$X$ quadrature signal averaged over all trajectories',
    xlabel=r'$t$',
    # ylabel=r'$\mathbb{E}[I_t^X]$',
    ylim=(-2.5, 2.5),
)
plt.legend()
renderfig('monitored-oscillator-IxIp')
```

![plot_monitored_oscillator_IxIp](/figs_docs/monitored-oscillator-IxIp.png){.fig}

```python
exp_ops = (dq.position(n), dq.momentum(n))
expects_all = dq.expect(exp_ops, result.states[0]).real

plt.plot(tsave, expects_all[0], lw=1.5, label=rf'$\mathrm{{Tr}}[X\rho_t]$',)
plt.plot(tsave, expects_all[1], lw=1.5, label=rf'$\mathrm{{Tr}}[P\rho_t]$',)

# plt.plot(tsave[1:], jnp.mean(Iks_x_all, axis=0), label=r'$\mathbb{E}[I_t^X]$')
# plt.plot(tsave[1:], jnp.mean(Iks_p_all, axis=0), label=r'$\mathbb{E}[I_t^P]$')

plt.gca().set(
    title=rf'Time evolution of $X$ expectation value',
    xlabel=r'$t$',
    # ylabel=rf'$\mathrm{{Tr}}[{name}\rho_t]$',
    ylim=(-2.5, 2.5),
)
plt.legend()

renderfig('monitored-oscillator-xp')
```

![plot_monitored_oscillator_xp](/figs_docs/monitored-oscillator-xp.png){.fig}


<!-- average like a mad man -->
