# Continuous jump measurement

In this example, we simulate stochastic trajectories of quantum systems that are continuously measured by a jump detector. We explain how to use [`dq.jssesolve()`][dynamiqs.jssesolve] to simulate trajectories modelled by the jump SSE, and [`dq.jsmesolve()`][dynamiqs.jsmesolve] to simulate trajectories modelled by the jump SME. For an introduction to the stochastic solvers, see the [Stochastic simulations](../basics/stochastic-simulations.md) tutorial.

```python
import jax
import jax.numpy as jnp
import numpy as np
from matplotlib import pyplot as plt
import dynamiqs as dq

dq.plot.mplstyle(dpi=150)  # set custom matplotlib style
```

## Qubit

We consider a qubit starting from the excited state $\ket{\psi_0}=\ket{e}$, with no unitary dynamics $H=0$ and two jump operators $L_-=\sigma_-$ (loss) and $L_+=\sigma_+$ (gain) which are continuously measured by jump detectors. Each click of the first detector brings the qubit to the ground state, and each click of the second one brings it back to the excited state, so we expect stochastic trajectories jumping between the excited and ground states.

Let's begin with perfectly efficient detectors. We start with a pure state, and we measure both jump operators with perfect efficiency, so we don't lose any information about the system's state. As a result, it remains pure at all times. We use [`dq.jssesolve()`][dynamiqs.jssesolve] to simulate these quantum trajectories.

### Simulation

```python
# define Hamiltonian, jump operators, initial state
H = dq.zeros(2)
jump_ops = [dq.sigmam(), dq.sigmap()]
psi0 = dq.excited()

# define save times
tsave = np.linspace(0, 2.0, 101)

# define a certain number of PRNG keys, one for each trajectory
key = jax.random.key(0)
ntrajs = 3
keys = jax.random.split(key, ntrajs)

# simulate trajectories
method = dq.method.EulerJump(dt=1e-3)
result = dq.jssesolve(H, jump_ops, psi0, tsave, keys, method=method)
```

```pycon
>>> print(result)
==== JSSESolveResult ====
Method     : EulerJump
Infos      : 2000 steps | infos shape (3,)
States     : QArray complex64 (3, 101, 2, 1) | 4.7 Kb
Clicktimes : Array float32 (3, 2, 10000) | 234.4 Kb
>>> result.states.shape  # (ntrajs, ntsave, n, 1)
(3, 101, 2, 1)
>>> result.clicktimes.shape  # (ntrajs, len(jump_ops), nmaxclick)
(3, 2, 10000)
```

The measurement record of each detector is returned as the list of its click times. Since the number of clicks varies from one trajectory to another, `result.clicktimes` is a fixed-size buffer of length `nmaxclick`, padded with `nan` after the last click, and `result.nclicks` holds the number of clicks of each detector. The buffer size defaults to `nmaxclick=10_000`, which is why `result.clicktimes` is much larger than the states here. It can be reduced with the `nmaxclick` argument, as long as it stays above the maximum number of clicks expected in a trajectory.

### Individual trajectories

```python
_, axs = dq.plot.grid(ntrajs, ntrajs, w=5, h=3, sharexy=True)

for i in range(ntrajs):
    ax = next(axs)
    dq.plot.xyz(result.states[i], times=tsave, ax=ax)
    ax.set(
        title=f'Trajectory {i + 1}, number of clicks: {result.nclicks[i].sum()}',
        xlabel=r'$t$',
        ylabel='Pauli expectation values',
    )
```

<!-- invisible-code-block: python
renderfig('jump-monitored-qubit-trajs')
-->

![plot_jump_monitored_qubit_trajs](../../figs_docs/jump-monitored-qubit-trajs.png){.fig}

Each trajectory alternates between $\braket{\sigma_z}=+1$ and $\braket{\sigma_z}=-1$, with a flip at each click. Between two clicks, nothing happens: because $L_-^\dagger L_- + L_+^\dagger L_+ = I$, the total click rate is $1$ whatever the state, so the absence of clicks carries no information about the qubit. The number of clicks over the duration $t=2$ thus follows a Poisson distribution of mean $2$.

### Imperfect detection

If the detection is imperfect, the system state is a density matrix. We use [`dq.jsmesolve()`][dynamiqs.jsmesolve] to simulate these quantum trajectories. For jump measurement we can model two classes of imperfections: false clicks (accounted by the dark count rate $\theta$) and missed clicks (accounted by the efficiency $\eta$).

```python
# define dark count rates
thetas = [0.0, 0.0]
# define efficiencies
etas = [0.5, 0.5]

# simulate trajectories
result = dq.jsmesolve(H, jump_ops, thetas, etas, psi0, tsave, keys, method=method)
```

```pycon
>>> print(result)
==== JSMESolveResult ====
Method     : EulerJump
Infos      : 2000 steps | infos shape (3,)
States     : QArray complex64 (3, 101, 2, 2) | 9.5 Kb
Clicktimes : Array float32 (3, 2, 10000) | 234.4 Kb
>>> result.states.shape  # (ntrajs, ntsave, n, n)
(3, 101, 2, 2)
>>> result.clicktimes.shape  # (ntrajs, nLm, nmaxclick)
(3, 2, 10000)
```

Here `nLm` is the number of measured jump operators, i.e. those with a non-zero efficiency: no click times are returned for a purely dissipative loss channel ($\eta=0$).

```python
_, axs = dq.plot.grid(ntrajs, ntrajs, w=5, h=3, sharexy=True)

for i in range(ntrajs):
    ax = next(axs)
    dq.plot.xyz(result.states[i], times=tsave, ax=ax)
    ax.set(
        title=f'Trajectory {i + 1}, number of clicks: {result.nclicks[i].sum()}',
        xlabel=r'$t$',
        ylabel='Pauli expectation values',
    )
```

<!-- invisible-code-block: python
renderfig('jump-monitored-qubit-eta-trajs')
-->

![plot_jump_monitored_qubit_eta_trajs](../../figs_docs/jump-monitored-qubit-eta-trajs.png){.fig}

With $\eta=0.5$, half of the jumps go undetected. Between two detected clicks, the qubit therefore relaxes towards the maximally mixed state, with $\braket{\sigma_z}$ decaying exponentially at rate $2(1-\eta)=1$, as seen on the trajectory without any detected click. Each detected click projects the qubit back onto a pure state: the ground state for a click of the first detector, and the excited state for a click of the second one.

## Quantum harmonic oscillator

We consider a quantum harmonic oscillator starting from the Fock state $\ket{n_0}$, with Hamiltonian $H=0$ and a single jump operator $L=\sqrt\kappa a$ which is continuously measured by photodetection. Each click removes one photon, so there are at most $n_0$ clicks per trajectory, and we set `nmaxclick` accordingly.

### Simulation

```python
# define Hamiltonian, jump operators, initial state
n = 16
a = dq.destroy(n)
kappa = 1.0
n0 = 10
H = dq.zeros(n)
jump_ops = [jnp.sqrt(kappa) * a]
psi0 = dq.fock(n, n0)

# define save times
tsave = np.linspace(0, 2.0, 101)

# define a certain number of PRNG keys, one for each trajectory
key = jax.random.key(42)
ntrajs = 100
keys = jax.random.split(key, ntrajs)

# simulate trajectories
method = dq.method.EulerJump(dt=1e-3)
exp_ops = [a.dag() @ a]
result = dq.jssesolve(
    H,
    jump_ops,
    psi0,
    tsave,
    keys,
    method=method,
    exp_ops=exp_ops,
    save_states=False,
    nmaxclick=n0,
)
```

```pycon
>>> print(result)
==== JSSESolveResult ====
Method     : EulerJump
Infos      : 2000 steps | infos shape (100,)
States     : QArray complex64 (100, 1, 16, 1) | 12.5 Kb
Expects    : Array complex64 (100, 1, 101) | 78.9 Kb
Clicktimes : Array float32 (100, 1, 10) | 3.9 Kb
```

### Individual and averaged trajectories

```python
exps_adaga = result.expects[:, 0, :].real
```

```python
kw = dict(lw=1.5, alpha=0.8)

# individual trajectories
for i, exp in enumerate(exps_adaga[:4]):
    plt.plot(tsave, exp, label=f'Trajectory {i + 1}', **kw)

# average trajectory
plt.plot(tsave, exps_adaga.mean(0), label='Average trajectory', color='gray', **kw)

# Lindblad solution
plt.plot(
    tsave,
    n0 * np.exp(-kappa * tsave),
    ls='--',
    label=r'Lindblad, $n_0 e^{-\kappa t}$',
    color='gray',
    **kw,
)

plt.gca().set(xlabel=r'$t$', ylabel=r'$\langle a^\dagger a \rangle_t$')
plt.legend()
```

<!-- invisible-code-block: python
renderfig('jump-monitored-oscillator-trajs')
-->

![plot_jump_monitored_oscillator_trajs](../../figs_docs/jump-monitored-oscillator-trajs.png){.fig}

Each trajectory is a staircase: the photon number stays constant between clicks, since the no-click evolution leaves a Fock state unchanged, and decreases by one at each click, with a click rate $\kappa\braket{a^\dagger a}_t$ that slows down as the oscillator empties. Averaging over the $100$ trajectories recovers the smooth exponential decay predicted by the Lindblad master equation, $\braket{a^\dagger a}_t = n_0 e^{-\kappa t}$, up to statistical fluctuations that shrink as $1/\sqrt{\text{ntrajs}}$.
