# Steady states

An open quantum system driven at constant parameters relaxes to a stationary state, its **steady state**. This tutorial defines it, explains the two ways to compute it, and shows how to compute it with [`dq.mesteadystate()`][dynamiqs.mesteadystate] instead of running [`dq.mesolve()`][dynamiqs.mesolve] to long times.

```python
import jax
import jax.numpy as jnp
from matplotlib import pyplot as plt
import dynamiqs as dq

dq.plot.mplstyle(dpi=150)  # set custom matplotlib style
```

## The steady state

A **steady state** $\rho_\infty$ is a density matrix left unchanged by the Lindblad master equation $\dd\rho/\dt = \mathcal{L}(\rho)$ (see the [Open systems](open-systems.md) tutorial):
$$
    \mathcal{L}(\rho_\infty) = -i[H, \rho_\infty] + \sum_{k=1}^N \left(L_k \rho_\infty L_k^\dag - \frac{1}{2} L_k^\dag L_k \rho_\infty - \frac{1}{2} \rho_\infty L_k^\dag L_k\right) = 0.
$$
In finite dimension, a steady state always exists. When it is unique, every initial state converges to it, at a rate set by the **Liouvillian gap**: the smallest non-zero $|\mathrm{Re}\,\lambda|$ among the eigenvalues $\lambda$ of $\mathcal{L}$.

!!! Example "Example for a two-level system"
    For $H=\frac{\omega}{2}\sigma_z$ and $L=\sqrt\gamma\sigma_-$, the unique steady state is the ground state $\rho_\infty=\ket{g}\bra{g}$.

When the steady state is **not unique**, the system keeps a memory of its initial state through conserved quantities. A cat qubit stabilized by two-photon loss is an example: its steady states span a whole qubit.

## Computing the steady state numerically

### Integrating until the state stops changing

The first approach integrates the master equation from an initial state, as [`dq.mesolve()`][dynamiqs.mesolve] does, and stops once $\mathcal{L}(\rho)$ is small enough. The number of steps grows as the inverse of the Liouvillian gap. Implicit ODE methods suit this problem, because they take larger steps as the state settles, but each of their steps solves an $n^2\times n^2$ linear system.

^^Space complexity^^: $O(n^2)$ for explicit methods, $O(n^4)$ for implicit methods with a dense linear solver.

^^Time complexity^^: $O(n^3\times\text{number of time steps})$ for explicit methods, $O(n^6\times\text{number of time steps})$ for implicit methods with a dense linear solver.

### Solving the linear system

The second approach solves $\mathcal{L}(\rho)=0$ as a linear system. The Liouvillian is singular, so the system is first deflated with the condition $\tr{\rho}=1$, which makes it invertible when the steady state is unique. A dense solver costs $O(n^6)$ in time and $O(n^4)$ in memory, which is prohibitive beyond $n\approx 100$.

An iterative solver such as GMRES avoids this cost, because it only applies $\mathcal{L}$ to $n\times n$ matrices. It converges in few iterations with a good **preconditioner**, a cheap approximation of $\mathcal{L}^{-1}$. Dynamiqs uses the inverse of the no-jump part of the Liouvillian, $\mathcal{S}(X)=H_\mathrm{nh}X+XH_\mathrm{nh}^\dag$ with $H_\mathrm{nh}=-iH-\frac12\sum_kL_k^\dag L_k$, which is cheap to invert in the eigenbasis of $H_\mathrm{nh}$.

^^Space complexity^^: $O(n^2\times\text{Krylov space size})$.

^^Time complexity^^: $O(n^3\times\text{number of iterations})$.

## Using Dynamiqs

[`dq.mesteadystate()`][dynamiqs.mesteadystate] takes the same Hamiltonian, jump operators and expectation operators as [`dq.mesolve()`][dynamiqs.mesolve], but no initial state or save times. Consider a cavity with detuning $\Delta$, driven with amplitude $\epsilon$ and losing photons at rate $\kappa$:
$$
    H = \Delta a^\dag a + \epsilon(a + a^\dag), \qquad L = \sqrt\kappa\, a.
$$
Its unique steady state is the coherent state $\ket{\alpha}$ with $\alpha = -i\epsilon/(i\Delta+\kappa/2)$:

```pycon
>>> n = 16
>>> a = dq.destroy(n)
>>> delta, eps, kappa = 1.0, 0.5, 1.0
>>> H = delta * a.dag() @ a + eps * (a + a.dag())
>>> jump_ops = [jnp.sqrt(kappa) * a]
>>> exp_ops = [a.dag() @ a, a]
>>> result = dq.mesteadystate(H, jump_ops, exp_ops=exp_ops)
>>> print(result)
==== MESteadyStateResult ====
Method   : DiffraxSteadyState
Infos    : converged at t=18.5 | 32 steps (32 accepted, 0 rejected)
State    : QArray complex64 (16, 16) | 2.0 Kb
Expects  : Array complex64 (2,) | 0.0 Kb
Residual : Array float32 () | 0.0 Kb
>>> result.expects  # numerical expectation values
Array([ 0.2+0.j , -0.4-0.2j], dtype=complex64)
>>> alpha = -1j * eps / (1j * delta + 0.5 * kappa)
>>> print(f'{abs(alpha) ** 2:.3f}, {alpha:.3f}')  # analytical expectation values
0.200, -0.400-0.200j
```

The result holds the state in `result.state`, its expectation values in `result.expects`, and the root mean square of $\mathcal{L}(\rho)$ in `result.residual`. A computation that does not converge raises no error: it returns the last state, with `result.infos.converged` set to `False`. Always check this flag, which batched computations set for each element:

```pycon
>>> result.infos.converged
Array(True, dtype=bool)
```

### Choosing a method

The `method` argument selects one of two methods:

| Method | Approach | When to use it |
| --- | --- | --- |
| [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState] (default) | Integrates the master equation, with `Kvaerno5` by default, until the state stops changing. | Small systems, non-unique steady states, and systems on which `LyapunovGMRES` fails. |
| [`LyapunovGMRES`][dynamiqs.method.LyapunovGMRES] | Solves the deflated linear system with preconditioned GMRES. | Large systems with a unique steady state, especially with a small Liouvillian gap. |

Continuing the last example:

```pycon
>>> method = dq.method.LyapunovGMRES()
>>> result = dq.mesteadystate(H, jump_ops, exp_ops=exp_ops, method=method)
>>> print(result)
==== MESteadyStateResult ====
Method   : LyapunovGMRES
Infos    : converged after 1 GMRES cycles
State    : QArray complex64 (16, 16) | 2.0 Kb
Expects  : Array complex64 (2,) | 0.0 Kb
Residual : Array float32 () | 0.0 Kb
>>> result.expects
Array([ 0.2+0.j , -0.4-0.2j], dtype=complex64)
```

Both methods stop when $\mathrm{rms}(\mathcal{L}(\rho)) < \mathrm{atol} + \mathrm{rtol}\cdot\mathrm{rms}(\rho)$, with `rtol` and `atol` defaulting to `1e-5`. Because $\mathcal{L}(\rho)$ is a rate, the error on the state is about this tolerance divided by the Liouvillian gap. The defaults suit rates of order one: for a slowest rate of $10^{-3}$, lower both tolerances by the same factor.

!!! Warning "When `LyapunovGMRES` fails"
    The preconditioner is singular when a state annihilated by every $L_k$ is also an eigenstate of $H$, as for an undriven lossy cavity or a decaying qubit. The method then returns NaN, with `result.infos.converged` set to `False`:

    ```pycon
    >>> result = dq.mesteadystate(dq.sigmaz(), [dq.sigmam()], method=method)
    >>> result.infos.converged
    Array(False, dtype=bool)
    ```

    The preconditioner also loses accuracy when the eigenvectors of $H_\mathrm{nh}$ are ill-conditioned, typically for large driven or squeezed modes in single precision. The method then fails to converge, or its gradient returns NaN. For these systems, use [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState] or double precision (see [Devices and precision](devices-and-precision.md)).

### Non-unique steady states

When the steady state is not unique, the result depends on `rho0`, the initial state of the method. Its default, the maximally mixed state $I/n$, converges to a mixture of all the steady states. For example, two-photon loss $L=a^2$ conserves the photon number parity, and relaxes every state to a mixture of $\ket0$ and $\ket1$:

```pycon
>>> n = 8
>>> a = dq.destroy(n)
>>> H = dq.zeros(n)
>>> jump_ops = [a @ a]
>>> exp_ops = [a.dag() @ a]
>>> dq.mesteadystate(H, jump_ops, exp_ops=exp_ops).expects  # from I/n
Array([0.5+0.j], dtype=complex64)
>>> dq.mesteadystate(H, jump_ops, rho0=dq.fock(n, 3), exp_ops=exp_ops).expects
Array([1.+0.j], dtype=complex64)
```

From $I/n$, half of the population is in odd Fock states, which gives $(\ket0\bra0+\ket1\bra1)/2$. From $\ket3$, the steady state is $\ket1$. Only [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState] handles this case, because the linear system of [`LyapunovGMRES`][dynamiqs.method.LyapunovGMRES] is then singular.

## Batching steady state computations

The Hamiltonian, the jump operators and `rho0` can be batched as in [`dq.mesolve()`][dynamiqs.mesolve] (see the [Batching simulations](batching-simulations.md) tutorial). For example, the steady state photon number of a driven Kerr oscillator, $H = -\Delta a^\dag a + K a^{\dag2}a^2 + \epsilon(a + a^\dag)$, as a function of the detuning:

```python
n = 32
a = dq.destroy(n)
K, eps, kappa = 0.1, 1.0, 1.0
deltas = jnp.linspace(-2.0, 4.0, 61)

H = -deltas[:, None, None] * a.dag() @ a + K * a.dag() @ a.dag() @ a @ a
H = H + eps * (a + a.dag())
jump_ops = [jnp.sqrt(kappa) * a]
exp_ops = [a.dag() @ a]
method = dq.method.LyapunovGMRES()
result = dq.mesteadystate(H, jump_ops, exp_ops=exp_ops, method=method)

plt.plot(deltas, result.expects[:, 0].real)
plt.gca().set(xlabel=r'Detuning $\Delta$', ylabel=r'$\langle a^\dagger a\rangle$')
```

<!-- invisible-code-block: python
renderfig('steady-state-kerr-detuning')
-->

![steady-state-kerr-detuning](../../figs_docs/steady-state-kerr-detuning.png){.fig}

The Kerr term shifts the cavity frequency by about $2K\braket{a^\dag a}$, which pulls the resonance towards positive detunings and makes it asymmetric.

## Computing gradients

The default gradient, [`dq.gradient.Implicit`][dynamiqs.gradient.Implicit], differentiates the equation $\mathcal{L}(\rho_\infty)=0$ instead of the iterations that reached $\rho_\infty$. Its cost thus does not depend on the number of steps, and it supports both forward and reverse modes. For the driven cavity above, $\braket{a^\dag a}=|\alpha|^2$ is proportional to $\epsilon^2$, so its derivative is $2|\alpha|^2/\epsilon$:

```pycon
>>> n = 16
>>> a = dq.destroy(n)
>>> delta, eps, kappa = 1.0, 0.5, 1.0
>>> def photon_number(eps):
...     H = delta * a.dag() @ a + eps * (a + a.dag())
...     jump_ops = [jnp.sqrt(kappa) * a]
...     result = dq.mesteadystate(H, jump_ops, exp_ops=[a.dag() @ a])
...     return result.expects[0].real
>>> print(f'{jax.grad(photon_number)(eps):.3f}')  # numerical gradient
0.800
>>> alpha = -1j * eps / (1j * delta + 0.5 * kappa)
>>> print(f'{2 * abs(alpha) ** 2 / eps:.3f}')  # analytical gradient
0.800
```

The implicit gradient requires a unique steady state. For a non-unique one, use [`DiffraxSteadyState`][dynamiqs.method.DiffraxSteadyState] with a gradient through the time integration, such as [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed]. These gradients are less accurate at the same tolerance, because the integration stops with a transient left whose derivative decays more slowly. [`LyapunovGMRES`][dynamiqs.method.LyapunovGMRES] only supports `Implicit`, which it computes with the same preconditioned GMRES. See the [Performance](performance.md#compute-steady-states-with-lyapunovgmres) tutorial for the cost of each option.
