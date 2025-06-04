# Basic examples

First time using Dynamiqs? Below are a few basic examples to help you get started.

## Simulate a lossy quantum harmonic oscillator

This first example shows simulation of a lossy harmonic oscillator with Hamiltonian $H=\omega a^\dagger a$ and a single jump operator $L=\sqrt{\kappa} a$ from time $0$ to time $T$, starting from the initial coherent state $\ket{\alpha_0}$.

```python
import dynamiqs as dq
import jax.numpy as jnp

# parameters
n = 16          # Hilbert space dimension
omega = 1.0     # frequency
kappa = 0.1     # decay rate
alpha0 = 1.0    # initial coherent state amplitude
T = 2 * jnp.pi  # total evolution time (one full revolution)

# initialize operators, initial state and saving times
a = dq.destroy(n)
H = omega * a.dag() @ a
jump_ops = [jnp.sqrt(kappa) * a]
psi0 = dq.coherent(n, alpha0)
tsave = jnp.linspace(0, T, 101)

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave)
print(result)
```

```text title="Output"
|██████████| 100.0% ◆ elapsed 6.30ms ◆ remaining 0.00ms
==== MESolveResult ====
Method : Tsit5
Infos  : 40 steps (40 accepted, 0 rejected)
States : QArray complex64 (101, 16, 16) | 202.0 Kb
```

## Compute gradients with respect to some parameters

Suppose that in the above example, we want to compute the gradient of the number of photons in the final state at time $T$, $\bar{n} = \mathrm{Tr}[a^\dagger a \rho(T)]$, with respect to the frequency $\omega$, the decay rate $\kappa$ and the initial coherent state amplitude $\alpha_0$.

```python
import dynamiqs as dq
import jax.numpy as jnp
import jax

# parameters
n = 16          # Hilbert space dimension
omega = 1.0     # frequency
kappa = 0.1     # decay rate
alpha0 = 1.0    # initial coherent state amplitude
T = 2 * jnp.pi  # total evolution time (one full revolution)

def population(omega, kappa, alpha0):
    """Return the oscillator population after time evolution."""
    # initialize operators, initial state and saving times
    a = dq.destroy(n)
    H = omega * a.dag() @ a
    jump_ops = [jnp.sqrt(kappa) * a]
    psi0 = dq.coherent(n, alpha0)
    tsave = jnp.linspace(0, T, 101)

    # run simulation
    result = dq.mesolve(H, jump_ops, psi0, tsave)

    return dq.expect(dq.number(n), result.states[-1]).real

# compute gradient with respect to omega, kappa and alpha
grad_population = jax.grad(population, argnums=(0, 1, 2))
grads = grad_population(omega, kappa, alpha0)
print(f'Gradient w.r.t. omega : {grads[0]:.4f}')
print(f'Gradient w.r.t. kappa : {grads[1]:.4f}')
print(f'Gradient w.r.t. alpha0: {grads[2]:.4f}')
```

```text title="Output"
|██████████| 100.0% ◆ elapsed 5.94ms ◆ remaining 0.00ms
Gradient w.r.t. omega : 0.0000
Gradient w.r.t. kappa : -3.3520
Gradient w.r.t. alpha0: 1.0670
```

!!! Note
    On this specific example, we can verify the result analytically. The state remains a coherent state at all time with complex amplitude $\alpha(t) = \alpha_0 e^{-\kappa t/2} e^{-i\omega t}$, and the final photon number is thus $\bar{n} = |\alpha(T)|^2 = \alpha_0^2 e^{-\kappa T}$. We can then compute the gradient with respect to the three parameters $\theta = (\omega, \kappa, \alpha_0)$:

    $$
    \nabla_\theta\ \bar{n} = \begin{pmatrix}
      \partial\bar{n} / \partial\omega \\
      \partial\bar{n} / \partial\kappa \\
      \partial\bar{n} / \partial\alpha_0
    \end{pmatrix}
    = \begin{pmatrix}
      0\\
      -\alpha_0^2 T e^{-\kappa T} \\
      2 \alpha_0 e^{-\kappa T}
    \end{pmatrix}
    \approx \begin{pmatrix}
      0.0 \\
      -3.3520 \\
      1.0670
    \end{pmatrix}
    $$
