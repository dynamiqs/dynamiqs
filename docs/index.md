<h1 align="center">
    <img src="./media/dynamiqs_logo.png" width="520" alt="dynamiqs library logo">
</h1>

[P. Guilmin](https://github.com/pierreguilmin), [R. Gautier](https://github.com/gautierronan), [A. Bocquet](https://github.com/abocquet), [E. Genois](https://github.com/eliegenois)

[![ci](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml)  ![python version](https://img.shields.io/badge/python-3.8%2B-blue) [![chat](https://badgen.net/badge/icon/on%20slack?icon=slack&label=chat&color=orange)](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA) [![license: GPLv3](https://img.shields.io/badge/license-GPLv3-yellow)](https://github.com/dynamiqs/dynamiqs/blob/main/LICENSE) [![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

High-performance quantum systems simulation with PyTorch.

The **dynamiqs** library enables GPU simulation of large quantum systems, and computation of gradients based on the evolved quantum state. Differentiable solvers are available for the Schrödinger Equation, the Lindblad Master Equation, and the Stochastic Master Equation. The library is fully built on PyTorch and can efficiently run on CPUs and GPUs.

:hammer_and_wrench: This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

Some exciting features of dynamiqs include:

- Running simulations on GPUs, with a significant speedup for large Hilbert space dimensions.
- Batching many simulations of different Hamiltonians or initial states to run them concurrently.
- Exploring quantum-specific solvers that preserve the properties of the state, such as trace and positivity.
- Computing gradients of any function of the evolved quantum state with respect to any parameter of the Hamiltonian, jump operators, or initial state.
- Using the library as a drop-in replacement for [QuTiP](https://qutip.org/) by directly passing QuTiP-defined quantum objects to our solvers.
- Implementing your own solvers with ease by subclassing our base solver class and focusing directly on the solver logic.
- Enjoy reading our carefully crafted documentation on our website: <https://www.dynamiqs.org>.

We hope that this library will prove beneficial to the community for e.g. simulations of large quantum systems, gradient-based parameter estimation, or large-scale quantum optimal control.

## Installation

We will soon make a first release of the library on PyPi. In the meantime, you can clone the repository locally and install directly from source:

```shell
git clone https://github.com/dynamiqs/dynamiqs.git
pip install -e dynamiqs
```

## Examples

### Simulate a lossy quantum harmonic oscillator

This first example shows simulation of a lossy harmonic oscillator with Hamiltonian $H=\omega a^\dagger a$ and a single jump operator $L=\sqrt{\kappa} a$ using QuTiP-defined objects:

```python
import dynamiqs as dq
import numpy as np
import qutip as qt
import torch

# parameters
n = 128       # Hilbert space dimension
omega = 1.0   # frequency
kappa = 0.1   # decay rate
alpha0 = 1.0  # initial coherent state amplitude

# QuTiP operators, initial state and saving times
a = qt.destroy(n)
H = omega * a.dag() * a
jump_ops = [np.sqrt(kappa) * a]
rho0 = qt.coherent_dm(n, alpha0)
t_save = np.linspace(0, 1.0, 101)

# uncomment the next line to run the simulation on GPU
# torch.set_default_device('gpu')

# simulation
result = dq.mesolve(H, jump_ops, rho0, t_save)
print(result)
```

Output:

```shell
|██████████| 100.0% - time 00:00/00:00
==== Result ====
Method       : Dopri5
Start        : 2023-07-07 10:26:13
End          : 2023-07-07 10:26:13
Total time   : 0.53 s
states       : Tensor (101, 128, 128) | 12.62 Mb
```

### Compute gradients with respect to some parameters

Suppose that in the above example, we want to compute the gradient of the number of photons in the final state, $\bar{n} = \mathrm{Tr}[a^\dagger a \rho(t_f)]$, with respect to the decay rate $\kappa$ and the initial coherent state amplitude $\alpha_0$. For this computation, we will define the objects with dynamiqs:

```python
import dynamiqs as dq
import numpy as np
import torch

# parameters
n = 128
omega = 1.0
kappa = torch.tensor([0.1], requires_grad=True)
alpha0 = torch.tensor([1.0], requires_grad=True)

# dynamiqs operators, initial state and saving times
a = dq.destroy(n)
H = omega * a.mH @ a
jump_ops = [torch.sqrt(kappa) * a]
rho0 = dq.coherent_dm(n, alpha0)
t_save = np.linspace(0, 1.0, 101)

# uncomment the next line to run the simulation on GPU
# torch.set_default_device('gpu')

# simulation
options = dq.options.Dopri5(gradient_alg='autograd', verbose=False)
result = dq.mesolve(H, jump_ops, rho0, t_save, options=options)

# gradient computation
loss = dq.expect(a.mH @ a, result.states[-1])  # Tr[a^dag a rho]
grads = torch.autograd.grad(loss, (kappa, alpha0))
print(
    f'gradient wrt to kappa  : {grads[0]}\n'
    f'gradient wrt to alpha0 : {grads[1]}'
)
```

Output:

```shell
gradient wrt to kappa  : tensor([-0.9048])
gradient wrt to alpha0 : tensor([1.8097])
```

## Let's discuss

If you're curious, have questions or suggestions, wish to contribute or simply want to say hello, please don't hesitate to engage with us, we're always happy to chat! You can join the community on Slack via [this invite link](https://join.slack.com/t/dynamiqs-org/shared_invite/zt-1z4mw08mo-qDLoNx19JBRtKzXlmlFYLA), open an issue on GitHub, or contact the lead developer via email at <pierreguilmin@gmail.com>.

## Contributing

We warmly welcome all contributions. Please refer to [CONTRIBUTING.md](https://github.com/dynamiqs/dynamiqs/blob/main/CONTRIBUTING.md) for detailed instructions.
