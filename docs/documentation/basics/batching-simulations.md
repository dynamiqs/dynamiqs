# Batching simulations

Batching allows **running many independent simulations concurrently**. It can dramatically speedup simulations, especially on GPUs. In this tutorial, we explain how to batch quantum simulations in dynamiqs.

```python
import dynamiqs as dq
import jax.numpy as jnp
import timeit
```

## Batching in short

Batching in dynamiqs is achieved by passing arrays of Hamiltonians, initial states, or jump operators to the simulation functions. The result of a batched simulation is a batched array that contains all the individual simulations results.

??? Example "Example: a simple batched simulation"
    As a first example, let us simulate the Schrödinger equation for a set of 20 Hamiltonians and 2 initial states:

    ```python
    # sweep several Hamiltonians over some parameter delta
    deltas = jnp.linspace(0., 1., 20)
    H = deltas[:, None, None] * dq.sigmax()  # (20, 2, 2)

    # define several initial states, stored in a list
    psis = [dq.fock(2, 0), dq.fock(2, 1)]  # [(2, 2), (2, 2)]

    # run simulation
    tsave = jnp.linspace(0., 1., 10)
    result = dq.sesolve(H, psis, tsave)

    # the computed states are stored in a batched array of kets
    print(f'Shape of result.states: {result.states.shape}')
    ```

    ```
    Shape of result.states: (20, 2, 10, 2, 1)
    ```

    The resulting `states` array has shape `(20, 2, 10, 2, 1)` where `20` is the number of Hamiltonians, `2` is the number of initial states, `10` is the number of saved states, and `(2, 1)` is the shape of a single state.

There are two batching modes in dynamiqs:

 - **Cartesian batching**: the simulation is run for all possible combinations of Hamiltonians, jump operators and initial states. The returned batching shape is `(...H, ...L0, ...L1,  (...), ...psi0)` where `...x` is the batching shape of object `x`. This is the default mode.

    ??? Example "Example: cartesian batching"
        For example with [`dq.mesolve`][dynamiqs.mesolve], if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
        - `rho0` has shape _(7, n, n)_,

        then `states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.

 - **Flat batching**: the simulation is run for each set of Hamiltonians, jump operators and initial states. The returned batching shape is `jnp.broadcast_shapes(...H, ...L0, ...L1,  (...), ...psi0)` corresponding to the broadcasted shape of all input objects. This mode is activated by setting `cartesian_batching=False` in [`dq.Options`][dynamiqs.Options].

    ??? Example "Example: flat batching"
        For example with [`dq.mesolve`][dynamiqs.mesolve], if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
        - `rho0` has shape _(3, n, n)_,

        then `states` has shape _(2, 3, ntsave, n, n)_.


## Mastering batching

### Initializing a batched array

The most straightforward way to use batching is to pass arrays with extra batching dimensions to the simulation functions. In JAX, there are many ways to initialize such batched arrays. The most generic way is to use `jnp.stack`:

```python
# define several Hamiltonians
Hx, Hy, Hz = dq.sigmax(), dq.sigmay(), dq.sigmaz()

# stack them along the first axis
H = jnp.stack([Hx, Hy, Hz])  # (3, 2, 2)

# run simulation
psi0 = dq.fock(2, 0)
tsave = jnp.linspace(0., 1., 10)
result = dq.sesolve(H, psi0, tsave)
print(result.states.shape)
```
```text
(3, 10, 2, 1)
```

Often, it is also desired to sweep over a parameter. In this case, you can use JAX broadcasting semantics:

```python
# sweep several Hamiltonians over some parameter delta
deltas = jnp.linspace(0., 1., 20)
H = deltas[:, None, None] * dq.sigmax()  # (20, 2, 2)

# run simulation
result = dq.sesolve(H, psi0, tsave)
print(result.states.shape)
```
```text
(20, 10, 2, 1)
```

??? Note "Broadcasting semantics in JAX"
    JAX and NumPy broadcasting semantics are very powerful and allow you to write concise and efficient code. For more information, see the [NumPy documentation](https://numpy.org/doc/stable/user/basics.broadcasting.html).

Finally, dynamiqs also provides utility functions that can be used to initialize batched arrays. This is for instance the case of [`dq.coherent`][dynamiqs.coherent] and [`dq.fock`][dynamiqs.fock] which can be used to create batched coherent and Fock states respectively:

```python
# create a batched coherent state
alphas = jnp.array([1., 2., 3.])
psis = dq.coherent(16, alphas)  # (3, 16, 1)

# run simulation
H = dq.sigmax()
result = dq.sesolve(H, psis, tsave)
print(result.states.shape)
```
```text
(3, 10, 16, 1)
```

### Batching over a TimeArray

We have seen how to batch over time-independent objects, but how about time-dependent ones? For each type of [`TimeArray`][dynamiqs.TimeArray], there are different rules. In short,

- For a [`PWCTimeArray`][dynamiqs.pwc], batching is enabled over the `values` argument which should be of shape `(..., len(times) - 1)`. For instance:
  ```pycon
  >>> times = jnp.linspace(0., 1., 10)
  >>> values = jnp.arange(45).reshape(5, 9)
  >>> H = dq.pwc(times, values, dq.sigmax())
  >>> print(H.shape)
  (5, 2, 2)
  ```
- For a [`ModulatedTimeArray`][dynamiqs.modulated], batching is enabled over the function argument, such that a batched array of shape `(...,)` should be returned by this function. For instance:
  ```pycon
  >>> omegas = jnp.linspace(0., 1., 6)
  >>> f = lambda t: jnp.cos(omegas * t)
  >>> H = dq.modulated(f, dq.sigmax())
  >>> print(H.shape)
  (6, 2, 2)
  ```
- For a [`CallableTimeArray`][dynamiqs.timecallable], batching is also enabled over the function argument, which should now return a batched array of shape `(..., n, n)`. For instance:
  ```pycon
  >>> omegas = jnp.linspace(0., 1., 7)
  >>> f = lambda t: omegas[:, None, None] * dq.sigmax()
  >>> H = dq.timecallable(f)
  >>> print(H.shape)
  (7, 2, 2)
  ```

??? Note "Function with additional arguments"
    To define a modulated or callable time array with a function that takes arguments other than time (extra `*args` and `**kwargs`), you can use [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial). For example:
    ```pycon
    >>> import functools
    >>> def pulse(t, omega, amplitude=1.0):
    ...     return amplitude * jnp.cos(omega * t)
    >>> # create function with correct signature (t: float) -> Array
    >>> f = functools.partial(pulse, omega=1.0, amplitude=5.0)
    >>> H = dq.modulated(f, dq.sigmax())
    ```

### One function fits all: the `jnp.vectorize` trick

In some cases, the function that defines your `TimeArray` may involve many parameters which are either batched or not depending on the simulation at hand. It is then tedious to redefine slight variations of the same function throughout the code, to accomodate for which parameters are batched. To illustrate this issue, consider the following example:

```python
# unbatched
epsilon, omega = 1.0, 2.0
f = lambda t: epsilon * jnp.cos(omega * t)
H = dq.modulated(f, dq.sigmax())

# batched over epsilon
epsilons = jnp.linspace(0.1, 1.0, 10)
f = lambda t: epsilons * jnp.cos(omega * t)
H = dq.modulated(f, dq.sigmax())

# batched over omega
omegas = jnp.linspace(0.1, 2.0, 10)
f = lambda t: epsilon * jnp.cos(omegas * t)
H = dq.modulated(f, dq.sigmax())

# batched over epsilon and omega (flat)
f = lambda t: epsilons * jnp.cos(omegas * t)
H = dq.modulated(f, dq.sigmax())
```

Here, we had to define four different functions `f` to accomodate for the different batching scenarios. This can be cumbersome and error-prone. To avoid this, we can use the `jnp.vectorize` decorator together with [`jax.tree_util.Partial`](https://jax.readthedocs.io/en/latest/_autosummary/jax.tree_util.Partial.html) to define a single function that works in all cases:

```python
from jax.tree_util import Partial

# define a single function that works in all cases
@jnp.vectorize
def init_f(epsilon, omega):
    def f(t, epsilons, omegas):
        return epsilons * jnp.cos(omegas * t)
    return Partial(f, epsilons=epsilon, omegas=omega)

# unbatched
epsilon, omega = 1.0, 2.0
H = dq.modulated(init_f(epsilon, omega), dq.sigmax())

# batched over epsilon
epsilons = jnp.linspace(0.1, 1.0, 10)
H = dq.modulated(init_f(epsilons, omega), dq.sigmax())

# batched over omega
omegas = jnp.linspace(0.1, 2.0, 10)
H = dq.modulated(init_f(epsilon, omegas), dq.sigmax())

# batched over epsilon and omega (flat)
H = dq.modulated(init_f(epsilons, omegas), dq.sigmax())
```
Here, `init_f` is a function that returns a partially applied function `f` with the correct batching parameters. The `@jnp.vectorize` decorator ensures that `init_f` can be called with scalar or batched arguments. This trick is especially useful when the function `f` is complex and involves many parameters.

<!-- remove until smesolve is written again
### Batching over stochastic trajectories (SME)

For the diffusive stochastic master equation solver, many stochastic trajectories must often be solved to obtain faithful statistics of the evolved density matrix. In this case, dynamiqs also provides batching over trajectories to run them simultaneously. This is performed automatically by setting the value of the `ntrajs` argument in [`dq.smesolve()`][dynamiqs.smesolve]. The resulting `states` object has shape `(bH?, brho?, ntrajs, ntsave, n, n)`.

-->

## Why batching?

When batching multiple simulations, the state is not a 2-D array that evolves in time but a N-D array which holds each independent simulation. This allows running **multiple simulations simultaneously** with great efficiency, especially on GPUs. Moreover, it usually simplifies the subsequent analysis of the simulation, because all the results are gathered in a single large array.

Common use cases for batching include:

- simulating a system with different values of a parameter (e.g. a drive amplitude),
- simulating a system with different initial states (e.g. for gate tomography),
- performing an optimisation using multiple starting points with random initial guesses (for parameters fitting or quantum optimal control).

### Performance comparison

To illustrate the performance gain of batching, let us compare the time it takes to run a batched simulation with the time it takes to run the same simulation in a for loop. We will simulate a set of 3,000 Hamiltonians on a two-level system, and compare the time it takes to run the simulations in batched and unbatched mode.

```python
# some batched Hamiltonian
omegas = jnp.linspace(0.0, 10.0, 100)
epsilons = jnp.linspace(0.0, 1.0, 30)
H = omegas[:, None, None] * dq.sigmaz() + epsilons[:, None, None, None] * dq.sigmax()

# other simulation parameters
psi0 = dq.basis(2, 0)
tsave = jnp.linspace(0.0, 1.0, 50)

# run functions
def run_batched():
    return dq.sesolve(H, psi0, tsave)

def run_unbatched():
    results = []
    for i in range(len(omegas)):
        for j in range(len(epsilons)):
            result = dq.sesolve(H[i, j], psi0, tsave)
            results.append(result)
    return results

# time functions
%timeit run_batched()
%timeit run_unbatched()
```

```text
44.1 ms ± 2.66 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)
2.59 s ± 52.8 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)
```

As we can see, the batched simulation is **much faster** than the unbatched one. In this simple example, it is about 60 times faster. The performance gain will be even more significant for larger simulations or when running on a GPU.
