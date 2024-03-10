# Time-dependent operators

In this short tutorial, we explain how to define time-dependent Hamiltonians – and more generally time-dependent operators – in dynamiqs. There are currently four formats: constant, piecewise constant, constant modulated by a time-dependent factor or arbitrary time-dependent defined by a function.

!!! Warning "Differences with QuTiP"
    dynamiqs manipulates JAX arrays, which are different from QuTiP quantum objects. See in [The sharp bits 🔪](/getting_started/sharp-bits.md) page for more details, briefly:

    - use `x + 2 * dq.eye(n)` instead of `x + 2`
    - use `x @ y` instead of `x * y`, and `dq.mpow(x, 4)` instead of `x**4`
    - use `dq.dag(x)`, `x.mT.conj()` instead of `x.dag()`

```python
import dynamiqs as dq
import jax.numpy as jnp
```

## The `TimeArray` type

In dynamiqs, time-dependent operators are defined using `TimeArray` objects. The core feature of such objects is that they can be called for arbitrary times, which computes and returns the underlying array at the given time.

```python
H = dq.constant(dq.sigmaz()) # initialize a constant TimeArray
print(type(H))
# dynamiqs.time_array.ConstantTimeArray
print(type(H(1.0)))
# jaxlib.xla_extension.ArrayImpl
print(H(1.0))
# Array([[ 1.+0.j,  0.+0.j],
#        [ 0.+0.j, -1.+0.j]], dtype=complex64)
```

`TimeArray` objects can also be arbitrarily summed together. The only requirement to do so is that the shape of the underlying arrays are broadcastable, and that they have the same floating point precision.

```python
H0 = dq.constant(dq.sigmaz()) # constant TimeArray
H1 = dq.modulated(lambda t: jnp.cos(2.0 * t), dq.sigmax()) # modulated TimeArray
H = H0 + H1
print(H(1.0))
# Array([[ 1.        +0.j, -0.41614684+0.j],
#        [-0.41614684+0.j, -1.        +0.j]], dtype=complex64)
```

Finally, a `TimeArray` also supports a subset of common utility functions, such as `.conj()`, `.shape` or `.reshape()`. More details can be found in the [Python API](../python_api/time_array/TimeArray.md).

Next, we show how to define a `TimeArray` in dynamiqs.

## Defining a `TimeArray`

### Constant operators

Constant operators can be defined using either **array-like objects** (e.g. Python lists, NumPy and JAX arrays, QuTiP Qobjs) or `ConstantTimeArray` objects. In all cases, the operator is then converted internally into the latter type for differentiability and GPU support. It is also possible to directly use dynamiqs [utility functions](../python_api/index.md) for common operators.

For instance, to define the Pauli Z operator $H = \sigma_z$, you can use any of the following syntaxes:

```python
# using Python lists
H = [[1, 0], [0, -1]]

# using NumPy arrays
import numpy as np
H = np.array([[1, 0], [0, -1]])

# using JAX arrays
import jax.numpy as jnp
H = jnp.array([[1, 0], [0, -1]])

# using QuTiP Qobjs
import qutip as qt
H = qt.sigmaz()

# using dynamiqs
import dynamiqs as dq
H = dq.sigmaz()
```

While such array-like objects can directly be passed to common dynamiqs functions, they can also be converted to a `TimeArray` using [`dq.constant()`](dynamiqs.constant).

### Piecewise constant operators

Piecewise constant (PWC) operators are defined using a set of three arrays: a set of times defining the boundaries of the intervals (of shape `(nv+1,)`), a set of constant values over each interval (of shape `(..., nv)`), and an array defining the operator (of shape `(n, m)`). These three arrays can then be fed to [`dq.pwc()`](dynamiqs.pwc) to instantiate the corresponding `PWCTimeArray`. Importantly, the `times` array must be sorted in ascending order, but does not need to be evenly spaced. When calling the time array at any given time, the returned array is the one corresponding to the interval in which the time falls.

```python
# define a PWC time array
times = jnp.array([0.0, 1.0, 2.0])
values = jnp.array([3.0, -2.0])
array = dq.sigmaz()
H = dq.pwc(times, values, array)

# call the time array at different times
print(H(0.5))
# [[ 3.+0.j  0.+0.j]
#  [ 0.+0.j -3.+0.j]]
print(H(1.5))
# [[-2.+0.j -0.+0.j]
#  [-0.+0.j  2.-0.j]]
print(H(1.0))
# [[-2.+0.j -0.+0.j]
#  [-0.+0.j  2.-0.j]]
print(H(-1.0))
# [[ 0.+0.j  0.+0.j]
#  [ 0.+0.j -0.+0.j]]
```

### Modulated operators

Modulated operators are defined using a Python function with signature `coeff(t: float, *args: ArrayLike) -> Array` that returns the time-dependent factor as a JAX array for any time `t`. The function can be passed to [`dq.modulated()`](dynamiqs.modulated) to obtain the corresponding `ModulatedTimeArray`. The operator is then modulated by the factor at every time step.

```python
# define a modulated time array
def coeff(t):
    return jnp.cos(2.0 * jnp.pi * t)
H = dq.modulated(coeff, dq.sigmax())

# call the time array at different times
print(H(0.5))
# [[-0.+0.j -1.+0.j]
#  [-1.+0.j -0.+0.j]]
print(H(1.0))
# [[0.+0.j 1.+0.j]
#  [1.+0.j 0.+0.j]]
```

??? Note "Function with optional arguments"
    To define a modulated time array with additional arguments, you can use the optional `args` parameter of [`dq.modulated()`](dynamiqs.modulated).
    ```python
    def coeff(t, omega):
        return jnp.cos(omega * t)

    omega = 1.0
    H = dq.modulated(coeff, dq.sigmax(), args=(omega,))
    ```

### Arbitrary time-dependent operators

A time-dependent operator can be defined using a Python function with signature `H(t: float, *args: ArrayLike) -> Array` that returns the operator as a JAX array for any time `t`. The function can be passed to [`dq.timecallable()`](dynamiqs.timecallable) to obtain a `CallableTimeArray` object.

For instance, to define the time-dependent Hamiltonian $H = \sigma_z + \cos(t)\sigma_x$, you can use the following syntax:

```python
H = dq.timecallable(lambda t: dq.sigmaz() + jnp.cos(t) * dq.sigmax())
```

!!! Warning "Function returning non-array object"
    An error is raised if `H(t)` returns a non-array object (including array-like objects such as QuTiP Qobjs or Python lists). This is enforced to avoid costly conversions at every time step of the numerical integration.

??? Note "Function with optional arguments"
    To define a callable time array with additional arguments, you can use the optional `args` parameter of [`dq.timecallable()`](dynamiqs.timecallable).

    ```python
    def _H(t, omega):
        return dq.sigmaz() + jnp.cos(omega * t) * dq.sigmax()
    omega = 1.0
    H = dq.timecallable(_H, args=(omega,))
    ```

## Batching and differentiating through a `TimeArray`

For modulated and callable time arrays, it is important that any array you want to batch over or differentiate against is passed as an extra argument to [`dq.timecallable()`](dynamiqs.timecallable) and [`dq.modulated()`](dynamiqs.modulated). Such requirements are very aligned with JAX philosophy, in which all internal arguments that are required at runtime should be explicitly provided.

Below is an example of how to define a time-dependent Hamiltonian that can be batched over and/or differentiated against the `omegas` array.

```python
H0 = dq.sigmaz()
H1 = dq.sigmax()
omegas = jnp.linspace(0.0, 2.0, 10)

# using a modulated time array
def coeff(t, omega):
    return jnp.cos(omega * t)
H = H0 + dq.modulated(coeff, H1, args=(omegas,))

# using a callable time array
def _H(t, omega):
    return H0 + jnp.cos(jnp.expand_dims(omega, (-1, -2)) * t) * H1
H = dq.timecallable(_H, args=(omegas,))
```
