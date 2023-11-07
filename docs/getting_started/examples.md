# Basic examples

First time using dynamiqs? Below are a few basic examples to help you get started. For a more complete set of tutorials, check out our [Tutorials](../tutorials/index.md) section.

## Simulate a lossy quantum harmonic oscillator

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
tsave = np.linspace(0, 1.0, 101)

# uncomment the next line to run the simulation on GPU
# torch.set_default_device('gpu')

# simulation
result = dq.mesolve(H, jump_ops, rho0, tsave)
print(result)
```

```text
|██████████| 100.0% - time 00:00/00:00
==== Result ====
Method       : Dopri5
Start        : 2023-09-10 16:57:34
End          : 2023-09-10 16:57:35
Total time   : 0.48 s
states       : Tensor (101, 128, 128) | 12.62 Mb
```

## Compute gradients with respect to some parameters

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
tsave = np.linspace(0, 1.0, 101)

# uncomment the next line to run the simulation on GPU
# torch.set_default_device('gpu')

# simulation
result = dq.mesolve(
    H, jump_ops, rho0, tsave,
    gradient=dq.gradient.Autograd(),
    options=dict(verbose=False),
)

# gradient computation
loss = dq.expect(a.mH @ a, result.states[-1]).real  # Tr[a^dag a rho]
grads = torch.autograd.grad(loss, (kappa, alpha0))
print(
    f'gradient wrt to kappa  : {grads[0]}\n'
    f'gradient wrt to alpha0 : {grads[1]}'
)
```

```text
gradient wrt to kappa  : tensor([-0.9048])
gradient wrt to alpha0 : tensor([1.8097])
```
