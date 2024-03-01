# Defining Hamiltonians

In this short tutorial, we explain how to define Hamiltonians in dynamiqs. There are currently four ways: using array-like objects for constant Hamiltonians, defining a function for time-dependent Hamiltonians, defining time-dependent coefficients as functions that precede constant Hamiltonians, or using a custom format for piecewise constant Hamiltonians.

!!! Warning "Differences with QuTiP"
    dynamiqs manipulates JAX arrays, which are different from QuTiP quantum objects. See in [The sharp bits 🔪](/getting_started/sharp-bits.html) page the main differences, briefly:

    - use `x + 2 * dq.eye(n)` instead of `x + 2`
    - use `x @ y` instead of `x * y`, and `dq.mpow(x, 4)` instead of `x**4`
    - use `dq.dag(x)`, `x.mT.conj()` instead of `x.dag()`

## Constant Hamiltonians

A constant Hamiltonian can be defined using **array-like objects**, i.e. NumPy and JAX arrays. In all cases, the Hamiltonian is then converted internally into a JAX array for differentiability and GPU support. It is also possible to directly use dynamiqs [utility functions](../python_api/index.md) for common Hamiltonians.

For instance, to define the Pauli Z operator $H = \sigma_z$, you can use any of the following syntaxes:

```python
# using NumPy arrays
import numpy as np
H = np.array([[1, 0], [0, -1]])

# using JAX arrays
import jax.numpy as jnp
H = jnp.array([[1, 0], [0, -1]])

# using dynamiqs
import dynamiqs as dq
H = dq.sigmaz()
```

## Time-dependent Hamiltonians

A time-dependent Hamiltonian can be defined using a Python function with signature `H(t: float, *args: ArrayLike) -> Array` that returns the Hamiltonian as a JAX array for any time `t`, which is then fed into `dq.timecallable`.

For instance, to define a time-dependent Hamiltonian $H = \sigma_z + \cos(t)\sigma_x$, you can use the following syntax:

```python
H = dq.timecallable(lambda t: dq.sigmaz() + jnp.cos(t) * dq.sigmax())
```

!!! Warning "Function returning non-array object"
    An error is raised if `H(t)` return a non-array object, or an array with a different `dtype` than the ones specified to the solver. This is enforced to avoid costly type or data type conversions at every time step of the numerical integration.

??? Note "Function with optional arguments"
    To define a time-dependent Hamiltonian with additional arguments, you can use the optional `args` parameter of `dq.timecallable`.
    ```python
    def _H(t, omega):
        return dq.sigmaz() + jnp.cos(omega * t) * dq.sigmax()
    omega = 1.0
    H = dq.timecallable(_H, args=(omega,))
    ```

    In addition, any array that you want to batch over should be passed as an extra argument to `dq.timecallable`. For instance,
    ```python
    def _H(t, omega):
        return dq.sigmaz() + jnp.cos(jnp.expand_dims(omega, (-1, -2)) * t) * dq.sigmax()
    omegas = jnp.linspace(0.0, 2.0, 10)
    H = dq.timecallable(_H, args=(omegas,))
    ```

## Piecewise constant Hamiltonians

!!! Warning "Work in Progress."
