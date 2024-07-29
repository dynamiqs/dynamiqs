# Batching simulations

Batching can be used to **run multiple independent simulations simultaneously**, and can dramatically speedup simulations, especially on GPUs. In this tutorial, we explain how to batch quantum simulations in dynamiqs.

```python
import dynamiqs as dq
import jax.numpy as jnp
import timeit
```

## Batching in dynamiqs

To simulate multiple Hamiltonians, you can pass an array of Hamiltonians for the argument `H` to [`dq.sesolve()`][dynamiqs.sesolve], [`dq.mesolve()`][dynamiqs.mesolve] or [`dq.smesolve()`][dynamiqs.smesolve]. You can also pass an array of initial states for the argument `psi0` (or `rho0` for open systems) to simulate multiple initial states. In this case, we say that the simulation is *batched*.

!!! Note "Result of a batched simulation"
    When a simulation is batched in dynamiqs, the result of the simulation is a batched array (a multi-dimensional array) that contains all the individual simulations results. The resulting `states` object has shape `(nH?, nstate?, ntsave, n, m)` where

    - `nH` is the number of Hamiltonians,
    - `nstate` is the number of initial states,
    - `ntsave` is the number of saved states,
    - `n` is the Hilbert space dimension,
    - `m=1` for closed systems and `m=n` for open systems.

    The `?` in the shape `(nH?, nstate?, ntsave, n, n)` indicates that the dimension is only present if the simulation is batched over Hamiltonians or initial states.

For instance, let's simulate the Schrödinger equation on multiple initial states:

```python
# initial states
g = dq.fock(2, 0)
e = dq.fock(2, 1)
plus = dq.unit(g + e)
minus = dq.unit(g - e)
psi0 = [g, e, plus, minus]  # shape (4, 2, 1)

H = dq.sigmaz()
tsave = jnp.linspace(0, 1, 11)  # shape (11)
exp_ops = [dq.sigmaz()]  # shape (1, 2, 2)
result = dq.sesolve(H, psi0, tsave, exp_ops=exp_ops)
print(result)
```

```
==== SEResult ====
Solver  : Tsit5
States  : Array complex64 (4, 11, 2, 1) | 0.69 Kb
Expects : Array complex64 (4, 1, 11) | 0.34 Kb
Infos   : avg. 6.0 steps (6.0 accepted, 0.0 rejected) | infos shape (4,)
```

The returned `states` array has shape `(4, 11, 2, 1)` where `4` is the number of initial states, `11` is the number of saved states (the length of `tsave`) and `(2, 1)` is the shape of a single state.

Similarly, `expects` has shape `(4, 1, 11)` where `4` is the number of initial states, `1` is the number of `exp_ops` operators (a single one here) and `11` is the number of saved expectation values (the length of `tsave`).

!!! Note "Creating a batched JAX array"
    To directly create a JAX batched array, use `jnp.stack`:

    ```python
    H0 = dq.sigmaz()
    H1 = dq.sigmax()
    H = jnp.stack([H0 + 0.1 * H1, H0 + 0.2 * H1, H0 + 0.3 * H1])  # shape (3, 2, 2)
    ```

    or JAX broadcasting semantics:

    ```python
    amplitudes = jnp.linspace(0.1, 0.3, 3)
    H = H0 + amplitudes[:, None, None] * H1  # shape (3, 2, 2)
    ```

<!-- remove until smesolve is written again
## Batching over stochastic trajectories (SME)

For the diffusive stochastic master equation solver, many stochastic trajectories must often be solved to obtain faithful statistics of the evolved density matrix. In this case, dynamiqs also provides batching over trajectories to run them simultaneously. This is performed automatically by setting the value of the `ntrajs` argument in [`dq.smesolve()`][dynamiqs.smesolve]. The resulting `states` object has shape `(bH?, brho?, ntrajs, ntsave, n, n)`.

-->

## Why batching?

When batching multiple simulations, the state is not a 2-D array that evolves in time but a N-D array which holds each independent simulation. This allows running **multiple simulations simultaneously** with great efficiency, especially on GPUs. Moreover, it usually simplifies the subsequent analysis of the simulation, because all the results are gathered in a single large array.

Common use cases for batching include:

- simulating a system with different values of a parameter (e.g. a drive amplitude),
- simulate a system with different initial states (e.g. for gate tomography),
- perform optimisation using multiple starting points with random initial guesses (for parameters fitting or quantum optimal control).
