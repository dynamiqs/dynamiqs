<h1 align="center">
    <img src="./docs/media/dynamiqs-logo.png" width="520" alt="dynamiqs library logo">
</h1>

[P. Guilmin](https://github.com/pierreguilmin), [R. Gautier](https://github.com/gautierronan), [A. Bocquet](https://github.com/abocquet), [E. Genois](https://github.com/eliegenois)

[![ci](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml?query=branch%3Amain)  ![python version](https://img.shields.io/badge/python-3.9%2B-blue) [![chat](https://badgen.net/badge/icon/on%20slack?icon=slack&label=chat&color=orange)](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA) [![license: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-yellow)](https://github.com/dynamiqs/dynamiqs/blob/main/LICENSE) [![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-performance quantum systems simulation with JAX.

The **dynamiqs** library enables GPU simulation of large quantum systems, and computation of gradients based on the evolved quantum state. Differentiable solvers are available for the Schrödinger equation, the Lindblad master equation, and the stochastic master equation. The library is fully built on JAX and can efficiently run on CPUs and GPUs.

> [!WARNING]
> This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

Some exciting features of dynamiqs include:

- Running simulations on **GPUs** and **TPUs**, with a significant speedup for large Hilbert space dimensions.
- **Batching** many simulations of different Hamiltonians, jump operators or initial states to run them concurrently.
- Exploring solvers **tailored to quantum** simulations that preserve the properties of the state, such as trace and positivity.
- Computing **gradients** of any function of the evolved quantum state with respect to any parameter of the Hamiltonian, jump operators, or initial state.

We hope that this library will prove beneficial to the community for e.g. simulations of large quantum systems, batched simulations of time-varying problems, gradient-based parameter estimation, or large-scale quantum optimal control.

## Installation

We will soon make a first release of the library on PyPi. In the meantime, you can install directly from source:

```shell
pip install git+https://github.com/dynamiqs/dynamiqs.git
```

## Examples

### Simulate a lossy quantum harmonic oscillator

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

### Compute gradients with respect to some parameters

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

## Let's talk!

If you're curious, have questions or suggestions, wish to contribute or simply want to say hello, please don't hesitate to engage with us, we're always happy to chat! You can join the community on Slack via [this invite link](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA), open an issue on GitHub, or contact the lead developer via email at <pierreguilmin@gmail.com>.

## Contributing

We warmly welcome all contributions. Please refer to [CONTRIBUTING.md](https://github.com/dynamiqs/dynamiqs/blob/main/CONTRIBUTING.md) for detailed instructions.
