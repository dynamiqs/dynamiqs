# Batching simulations

Batching can be used to **run multiple independent simulations simultaneously**, and can dramatically speedup simulations, especially on GPUs. In this tutorial, we explain how to batch quantum simulations in dynamiqs.

***

```python
import torch
import dynamiqs as dq
import timeit
from math import pi, sqrt
```

## Batching in dynamiqs

To simulate multiple Hamiltonians, you can pass a list of Hamiltonians for the argument `H` to [`dq.sesolve()`](../python_api/solvers/sesolve.md), [`dq.mesolve()`](../python_api/solvers/mesolve.md) or [`dq.smesolve()`](../python_api/solvers/smesolve.md). You can also pass a list of initial states for the argument `psi0` (or `rho0` for open systems) to simulate multiple initial states. In this case, we say that the simulation is *batched*.

!!! Note "Result of a batched simulation"
    When a simulation is batched in dynamiqs, the result of the simulation is a batched tensor (a multi-dimensional array) that contains all the individual simulations results. The resulting `states` object has shape `(bH?, bstate?, nt, n, m)` where

    - `bH` is the number of Hamiltonians,
    - `bstate` is the number of initial states,
    - `nt` is the number of saved states,
    - `n` is the Hilbert space dimension,
    - `m=1` for closed systems and `m=n` for open systems.

    The `?` in the shape `(bH?, bstate?, nt, n, n)` indicates that the dimension is only present if the simulation is batched over Hamiltonians or initial states.

For instance, let's simulate the Schrödinger equation on multiple initial states:

```python
# initial states
g = dq.fock(2, 0)
e = dq.fock(2, 1)
plus = dq.unit(g + e)
minus = dq.unit(g - e)
psi0 = [g, e, plus, minus]  # shape (4, 2, 1)

H = dq.sigmaz()
tsave = torch.linspace(0, 1, 11)  # shape (11)
exp_ops = [dq.sigmaz()]  # shape (1, 2, 2)
result = dq.sesolve(H, psi0, tsave, exp_ops=exp_ops)
```

```pycon
>>> result.states.shape
torch.Size([4, 11, 2, 1])
```

The returned `states` Tensor has shape `(4, 11, 2, 1)` where `4` is the number of initial states, `11` is the number of saved states (the length of `tsave`) and `(2, 1)` is the shape of a single state.

```pycon
>>> result.expects.shape
torch.Size([4, 1, 11])
```

Similarly, `expects` has shape `(4, 1, 11)` where `4` is the number of initial states, `1` is the number of `exp_ops` operators (a single one here) and `11` is the number of saved expectation values (the length of `tsave`).

!!! Note "Creating a batched tensor with PyTorch"
    To directly create a batched PyTorch tensor, use `torch.stack`:

    ```python
    H0 = dq.sigmaz()
    H1 = dq.sigmax()
    H = torch.stack([H0 + 0.1 * H1, H0 + 0.2 * H1, H0 + 0.3 * H1])  # shape (3, 2, 2)
    ```

    or PyTorch broadcasting semantics:

    ```python
    amplitudes = torch.linspace(0.1, 0.3, 3)
    H = H0 + amplitudes[:, None, None] * H1  # shape (3, 2, 2)
    ```

## Batching over stochastic trajectories (SME)

For the diffusive stochastic master equation solver, many stochastic trajectories must often be solved to obtain faithful statistics of the evolved density matrix. In this case, dynamiqs also provides batching over trajectories to run them simultaneously. This is performed automatically by setting the value of the `ntrajs` argument in [`dq.smesolve()`](../python_api/solvers/smesolve.md). The resulting `states` object has shape `(bH?, brho?, ntrajs, nt, n, n)`.

## Why batching?

When batching multiple simulations, the state is not a 2-D tensor that evolves in time but a N-D tensor which holds each independent simulation. This allows running **multiple simulations simultaneously** with great efficiency, especially on GPUs. Moreover, it usually simplifies the subsequent analysis of the simulation, because all the results are gathered in a single large array.

Common use cases for batching include:

- simulating a system with different values of a parameter (e.g. a drive amplitude),
- simulate a system with different initial states (e.g. for gate tomography),
- perform optimisation using multiple starting points with random initial guesses (for parameters fitting or quantum optimal control).

Let's quickly benchmark the speedup obtained by batching a simple simulation:

```python
# Hilbert space size
n = 16

# Hamiltonians
amplitudes = torch.linspace(-1, 1, 11)
a = dq.destroy(n)
H = amplitudes[:, None, None] * (a + a.mH)  # shape (11, 16, 16)

# jump operator
jump_ops = [sqrt(0.1) * a]

# initial states
angles = torch.linspace(0, 2 * pi, 11)
alphas = 2.0 * torch.exp(1j * angles)
rho0 = torch.stack([dq.coherent_dm(n, a) for a in alphas])  # shape (11, 16, 16)

# time vector
tsave = torch.linspace(0, 1, 11)

def run_unbatched(device):
    options=dict(device=device, verbose=False)
    for i in range(H.shape[0]):
        for j in range(rho0.shape[0]):
            dq.mesolve(H[i], jump_ops, rho0[j], tsave, options=options)

def run_batched(device):
    options=dict(device=device, verbose=False)
    dq.mesolve(H, jump_ops, rho0, tsave, options=options)
```

So we want to run a total of `11 * 11 = 121` simulations. Let's compare how long it takes to run them unbatched vs batched on CPU[^1]:
[^1]: Apple M1 chip with 8-core CPU.

% skip: start

```pycon
>>> %timeit run_unbatched('cpu')
1.78 s ± 184 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
>>> %timeit run_batched('cpu')
284 ms ± 129 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

Even with this simple example, we gain a **factor x6** in speedup just from batching.

The result is even more striking on GPU[^2]:
[^2]: NVIDIA GeForce RTX 4090.

```pycon
>>> %timeit run_unbatched('cuda')
1.51 s ± 1.98 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
>>> %timeit run_batched('cuda')
20.1 ms ± 78.7 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
```

On the GPU, because we save costly data transfers with the CPU and do N-D matrices multiplications, we gain a **factor x75** in speedup!

% skip: end
