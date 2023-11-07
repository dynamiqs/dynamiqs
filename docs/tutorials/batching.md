# Batching simulations

Batching refers to the process of grouping similar tasks that require similar resources to streamline their completion and improve efficiency. In the context of quantum simulations, batching can be used to run multiple simulations concurrently, and can dramatically speedup simulations. In this tutorial, we explain how to batch quantum simulations in dynamiqs.

```python
import torch
import dynamiqs as dq
import timeit
from math import pi, sqrt
```

***

## Batching over Hamiltonians

To batch over multiple Hamiltonians, you can pass a Tensor of shape `(b, n, n)` as the `H` argument of [`sesolve`](/python_api/solvers/sesolve.html), [`mesolve`](/python_api/solvers/mesolve.html) or [`smesolve`](/python_api/solvers/smesolve.html). In this case, `n` is the Hilbert space dimension, and `b` the batching dimension.

In general, this can be achieved using `torch.stack`. For instance:
```python
>>> H0 = dq.sigmaz()
>>> H1 = dq.sigmax()
>>> H = torch.stack([H0 + 0.1 * H1, H0 + 0.2 * H1, H0 + 0.3 * H1])
>>> H.shape
torch.Size([3, 2, 2])

```

Similar constructions can be applied with the use of PyTorch broadcasting semantics. The same batched Hamiltonian as before can be defined with:
```python
>>> amplitudes = torch.linspace(0.1, 0.3, 3)
>>> H = H0 + amplitudes[:, None, None] * H1
>>> H.shape
torch.Size([3, 2, 2])

```

## Batching over initial states

To batch over multiple initial states, you can pass a Tensor of shape `(b, n, m)` as the `psi0` or `rho0` argument of simulation functions. In this case, `n` and `m` are the Hilbert space dimension (with `m=1` for a ket and `m=n` for a density matrix), and `b` the batching dimension.

Similarly as with Hamiltonians, this can be achieved using `torch.stack`, broadcasting semantics or batchable utility functions. For instance:
```python
>>> psi0 = torch.stack([dq.fock(2, 0),
...                     dq.fock(2, 1),
...                     dq.unit(dq.fock(2, 0) + dq.fock(2, 1)),
...                     dq.unit(dq.fock(2, 0) - dq.fock(2, 1))])
>>> psi0.shape
torch.Size([4, 2, 1])
```
or
```python
>>> alphas = torch.linspace(0, 1, 3)
>>> rho0 = dq.coherent_dm(20, alphas)
>>> rho0.shape
torch.Size([3, 20, 20])

```

## Batching over stochastic trajectories (SME)

For the diffusive stochastic master equation solver, many stochastic trajectories must often be solved to obtain faithful statistics of the evolved density matrix. In this case, dynamiqs also provides batching over trajectories. This is performed automatically by passing a non-zero `ntrajs` to `smesolve` optional arguments. See the [API documentation](/python_api/solvers/smesolve.html) for more details.

## Results of a batched simulation

When a simulation is batched in dynamiqs, the result of the simulation is also returned with batched tensors. In this case, the batching dimensions are returned with the following order: `(bH, brho, ntrajs, ...)`. For any of these three batching dimensions to the Tensor, if batching is not performed, the dimension is removed and a Tensor with one less dimension is returned.

For instance, for the simulation of a Schrödinger equation batched over Hamiltonian and initial state, we have:
```python
>>> amplitudes = torch.linspace(0.1, 0.3, 3)
>>> H = dq.sigmaz() + amplitudes[:, None, None] * dq.sigmax()
>>> psi0 = torch.stack([dq.fock(2, 0),
...                     dq.fock(2, 1),
...                     dq.unit(dq.fock(2, 0) + dq.fock(2, 1)),
...                     dq.unit(dq.fock(2, 0) - dq.fock(2, 1))])
>>> tsave = torch.linspace(0, 1, 10)
>>> exp_ops = [dq.sigmaz()]
>>> result = dq.sesolve(H, psi0, tsave, exp_ops=exp_ops)
>>> result.states.shape
torch.Size([3, 4, 10, 2, 1])
>>> result.expects.shape
torch.Size([3, 4, 1, 10])
```
The returned `states` Tensor has shape `(3, 4, 10, 2, 1)` where `(2, 1)` is the individual shape of each state, `(10,)` is the shape of the time vector, `(4,)` is the shape of the batched initial state, and `(3,)` is the shape of the batched Hamiltonian. Similarly, `expects` has the same batching dimensions, but with a dimension `(1,)` that corresponds to the single expectation value provided, and a dimension `(10,)` that corresponds to the time vector.


## Why batching?

Batching can be used to run multiple simulations concurrently, and can dramatically speedup simulations. This is particularly useful when running simulations on GPUs, where the overhead of moving data to and from the GPU can be significant. In this case, batching can be used to reduce the number of data transfers to and from the GPU, and thus improve the overall performance of the simulation.

Even when running simulations on CPUs, batching can be used to improve the performance of simulations.

```python
# Hilbert space size
N = 20

# Batched Hamiltonian
amplitudes = torch.linspace(-5, 5, 10)
a = dq.destroy(N)
H0 = a.mH @ a
H1 = a + a.mH
H = H0 + amplitudes[:, None, None] * H1

# Batched initial state
angles = torch.linspace(0, 2 * pi, 20)
alphas = 2.0 * torch.exp(1j * angles)
psi0 = dq.coherent(N, alphas)

# Jump operator
gamma = 0.1
jump_ops = [sqrt(gamma) * a]

# Time array
tsave = torch.linspace(0, 1, 30)

def run_batched():
    dq.mesolve(H, jump_ops, psi0, tsave)

def run_unbatched():
    for i in range(H.shape[0]):
        for j in range(psi0.shape[0]):
            dq.mesolve(H[i], jump_ops, psi0[j], tsave)

```
```text
%timeit run_batched()
# 1.69 s ± 7.32 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
%timeit run_unbatched()
# 6.9 s ± 187 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

```

Even in this simple example, we gain a factor 4 in speedup just from batching !
