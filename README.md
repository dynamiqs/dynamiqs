<h1 align="center">
    <img src="./docs/media/dynamiqs_logo.png" width="520" alt="dynamiqs library logo">
</h1>

[![ci](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml/badge.svg)](https://github.com/dynamiqs/dynamiqs/actions/workflows/ci.yml)  ![python version](https://img.shields.io/badge/python-3.8%2B-blue) [![license: GPLv3](https://img.shields.io/badge/license-GPLv3-yellow)](./LICENSE) [![code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<!-- PyTorch implementation of differentiable quantum dynamics solvers. -->
<!-- Differentiable Quantum Solvers -->

High-performance quantum systems simulation with PyTorch.

The **dynamiqs** library enables simulation of large quantum systems on GPUs, and the computation of the gradient of any function depending on the evolved quantum state. We provide differentiable solvers for the Schrödinger Equation, the Lindblad Master Equation, and the Stochastic Master Equation. All of these solvers are implemented using PyTorch and can run on either a CPU or a GPU.

:hammer_and_wrench: This library is under active development and while the APIs and solvers are still finding their footing, we're working hard to make it worth the wait. Check back soon for the grand opening!

Here are the main features we're proud of:

- Run your simulation on GPUs, to gain a significant speedup for large Hilbert space dimensions.
- Batch your simulations to run them concurrently on multiple different Hamiltonians or initial states.
- Explore quantum-specific solvers that preserve the properties of the state, such as positivity and trace.
- Compute the gradient of any function that depends on the quantum state at any given time with respect to any parameter of the Hamiltonian, Lindblad jump operators, or initial state.
- Choose to compute the gradient using fast automatic differentiation for smaller systems, or opt for the adjoint method for constant memory cost.
- Use the library as a drop-in replacement for [QuTiP](https://qutip.org/) by directly passing QuTiP-defined quantum objects to our solvers.
- Implement your own solvers with ease by subclassing our base solver class and focusing directly on the solver logic.
- Enjoy reading our carefully crafted documentation on our website: <https://www.dynamiqs.org>.
- Benefit from a well-structured library with a rigorous test suite, continuous integration and best coding practices (and a team of friendly enthusiast developers :nerd_face:).

We hope that this library will prove beneficial to the community for simulations of large quantum systems like coupled bosonic modes, for gradient-based parameter estimation through experimental data fitting, and for efficient quantum optimal control.

## Installation

We will soon make a first release of the library on PyPi. In the meantime, you can clone the repository locally and install directly from source:

```shell
git clone https://github.com/dynamiqs/dynamiqs.git
pip install -e dynamiqs
```

## Examples

### Simulate a leaky bosonic mode

Let's simulate a leaky cavity with Hamiltonian $H=\Delta a^\dagger a$ and a single Lindblad jump operator $L=\sqrt{\kappa} a$, using QuTiP to define these operators:

```python
import dynamiqs as dq
import numpy as np
import qutip as qt
import torch

# parameters
n = 128
delta = 1.0
kappa = 0.5
alpha0 = 1.0

# QuTiP operators, initial state and saving times
a = qt.destroy(n)
H = a.dag() * a
jump_ops = [np.sqrt(kappa) * a]
rho0 = qt.coherent(n, alpha0)
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
Start        : 2023-07-06 22:52:30
End          : 2023-07-06 22:52:31
Total time   : 0.61 s
states       : Tensor (101, 128, 128) | 12.62 Mb
```

### Compute the gradient with respect to some parameters

Suppose that in the above example, we want to compute the gradient of the final state photon number expectation value $\text{loss}=\braket{a^\dagger a}=\mathrm{Tr}[a^\dagger a \rho]$ with respect to the decay rate $\kappa$ and the initial coherent state amplitude $\alpha_0$. For this computation, we will define the objects with dynamiqs:

```python
import dynamiqs as dq
import numpy as np
import torch

# parameters
n = 128
delta = 1.0
kappa = torch.tensor([0.5], requires_grad=True)
alpha0 = torch.tensor([1.0], requires_grad=True)

# dynamiqs operators, initial state and saving times
a = dq.destroy(n)
H = a.mH @ a
jump_ops = [torch.sqrt(kappa) * a]
rho0 = dq.coherent(n, alpha0)
t_save = np.linspace(0, 1.0, 101)

# uncomment the next line to run the simulation on GPU
# torch.set_default_device('gpu')

# simulation
options = dq.options.Dopri5(gradient='autograd', verbose=False)
result = dq.mesolve(H, jump_ops, rho0, t_save, options=options)

# gradient computation
loss = dq.expect(a.mH @ a, result.states[-1])  # Tr[a^dag a rho]
grads = torch.autograd.grad(loss, (kappa, alpha0))
print(grads)
```

Output:

```shell
(tensor([-0.6065]), tensor([1.2131]))
```

## Performance

While we have not yet conducted a rigorous benchmark of the library, we have some preliminary results. For instance, simulating two coupled dissipative bosonic modes each truncated at 32 photons takes approximately **2 minutes** with QuTiP, compared to **4 seconds** with dynamiqs (when executed on a standard GPU).

## Let's discuss

If you're curious, have questions or suggestions, wish to contribute or simply want to say hello, please don't hesitate to engage with us, we're always happy to chat! For now, you can open an issue on GitHub, or contact the lead developer via email at <pierreguilmin@gmail.com>.

## Contributing

We warmly welcome all contributions. Please refer to [CONTRIBUTING.md](CONTRIBUTING.md) for detailed instructions.
