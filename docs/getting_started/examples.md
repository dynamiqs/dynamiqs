# Basic examples

First time using dynamiqs? Below are a few basic examples to help you get started. For a more complete set of tutorials, check out our [Tutorials](../tutorials/index.md) section.

## Simulate a lossy quantum harmonic oscillator

This first example shows simulation of a lossy harmonic oscillator with Hamiltonian $H=\omega a^\dagger a$ and a single jump operator $L=\sqrt{\kappa} a$.

```python
import dynamiqs as dq
import jax.numpy as jnp

# parameters
n = 128      # Hilbert space dimension
omega = 1.0  # frequency
kappa = 0.1  # decay rate
alpha = 1.0  # initial coherent state amplitude

# initialize operators, state and saving times
a = dq.destroy(n)
H = omega * dq.dag(a) @ a
jump_ops = [jnp.sqrt(kappa) * a]
psi0 = dq.coherent(n, alpha)
tsave = jnp.linspace(0, 1.0, 101)

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave)
print(result)
```

```text
==== Result ====
Solver  : Tsit5
States  : Array complex64 (101, 128, 128) | 12.62 Mb
```

## Compute gradients with respect to some parameters

Suppose that in the above example, we want to compute the gradient of the number of photons in the final state, $\bar{n} = \mathrm{Tr}[a^\dagger a \rho(t_f)]$, with respect to the decay rate $\kappa$ and the initial coherent state amplitude $\alpha$. For this computation, we will define the objects with dynamiqs:

```python
import dynamiqs as dq
import jax.numpy as jnp
import jax

# parameters
n = 128      # Hilbert space dimension
omega = 1.0  # frequency
kappa = 0.1  # decay rate
alpha = 1.0  # initial coherent state amplitude

def population(omega, kappa, alpha):
    """Return the population inside the cavity after time evolution."""
    # initialize operators, state and saving times
    a = dq.destroy(n)
    H = omega * dq.dag(a) @ a
    jump_ops = [jnp.sqrt(kappa) * a]
    psi0 = dq.coherent(n, alpha)
    tsave = jnp.linspace(0, 1.0, 101)

    # run simulation
    result = dq.mesolve(H, jump_ops, psi0, tsave)

    return dq.expect(dq.dag(a) @ a, result.states[-1]).real

# compute gradient with respect to kappa and alpha
grad_population = jax.grad(population, argnums=(1, 2))
print(grad_population(omega, kappa, alpha))
```

```text
(Array(-0.90483725, dtype=float32, weak_type=True), Array(1.8096755, dtype=float32, weak_type=True))
```
