# Time-dependent operators

In this short tutorial, we explain how to define time-dependent Hamiltonians – and more generally time-dependent operators – in dynamiqs. There are currently four supported formats: constant operator, piecewise constant operator, constant operator modulated by a time-dependent factor, or arbitrary time-dependent operator defined by a function.

!!! Warning "Differences with QuTiP"
    dynamiqs uses JAX arrays, which are different from QuTiP quantum objects. See [The sharp bits 🔪](../getting_started/sharp-bits.md) page for more details, briefly:

    - use `x + 2 * dq.eye(n)` instead of `x + 2`
    - use `x @ y` instead of `x * y`, and `dq.mpow(x, 4)` instead of `x**4`
    - use `dq.dag(x)` or `x.mT.conj()` instead of `x.dag()`

```python
import dynamiqs as dq
import jax.numpy as jnp
```

## The [`TimeArray`][dynamiqs.TimeArray] type

In dynamiqs, time-dependent operators are defined using [`TimeArray`][dynamiqs.TimeArray] objects. These objects can be called at arbitrary times, and return the corresponding array at that time:

```pycon
>>> H = dq.timecallable(lambda t: t * dq.sigmaz()) # initialize a callable time-array
>>> H(2.0)
Array([[ 2.+0.j,  0.+0.j],
       [ 0.+0.j, -2.+0.j]], dtype=complex64)
>>> H.shape
(2, 2)
```

Time-arrays support common arithmetic operations, for example we can add two time-arrays together:

```pycon
>>> H0 = dq.constant(dq.sigmaz()) # constant time-array
>>> H1 = dq.modulated(lambda t: jnp.cos(2.0 * t), dq.sigmax()) # modulated time-array
>>> H = H0 + H1
>>> H(1.0)
Array([[ 1.   +0.j, -0.416+0.j],
       [-0.416+0.j, -1.   +0.j]], dtype=complex64)
```

Finally, time-arrays also support common utility functions, such as `.conj()`, or `.reshape()`. More details can be found in the corresponding API page [`TimeArray`][dynamiqs.TimeArray].

## Defining a [`TimeArray`][dynamiqs.TimeArray]

### Constant operators

A constant operator is an operator of the form
$$
\hat O(t) = \hat O_0
$$
for any time $t$. In dynamiqs, constant operators can be defined either with **array-like objects** (e.g. Python lists, NumPy and JAX arrays, QuTiP Qobjs) or with `ConstantTimeArray` objects. In all cases, the operator is then converted internally into the latter type for differentiability and GPU support. It is also possible to directly use dynamiqs [utility functions](../python_api/index.md) for common operators. If you need to explicitely define a constant time array, you can use [`dq.constant()`](../python_api/time_array/constant.md).

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

# using a constant time array
H = dq.constant(dq.sigmaz())
```

### Piecewise constant operators

A piecewise constant (PWC) operator is an operator of the form
$$
 \hat O(t) = \left( \sum_{k=0}^{N-1} c_k w_{[t_k, t_{k+1}[}(t)\right) \hat O_0
$$
where $c_k$ are constant values, $w_{[t_k, t_{k+1}[}$ is the rectangular window function that is unity inside the interval and null otherwise, and $t_k$ are the boundaries of the intervals. In dynamiqs, PWC operators are defined using a set of three arrays:

- the set of times $[t_0, \ldots, t_N]$ defining the boundaries of the intervals, of shape _(N+1,)_,
- the set of constant values $[c_0, \ldots, c_{N-1}]$ over each interval, of shape _(..., N)_,
- the array defining the main operator $\hat O_0$, of shape _(n, m)_.

These three arrays can then be fed to [`dq.pwc()`](../python_api/time_array/pwc.md) to instantiate the corresponding `PWCTimeArray`. Importantly, the `times` array must be sorted in ascending order, but does not need to be evenly spaced. When calling the time array at any given time, the returned array is the one corresponding to the interval in which the time falls.

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

Modulated operators are operators of the form
$$
 \hat O(t) = f(t) \hat O_0
$$
where $f(t)$ is an arbitrary time-dependent factor. In dynamiqs, modulated operators are defined using:

- a Python function with signature `f(t: float, *args: ArrayLike) -> Array` that returns the time-dependent factor of shape _(...,)_ as a JAX array for any time $t$,
- the array defining the main operator $\hat O_0$, of shape _(n, m)_.

The function can be passed to [`dq.modulated()`](../python_api/time_array/modulated.md) to obtain the corresponding `ModulatedTimeArray`.

```python
# define a modulated time array
def f(t):
    return jnp.cos(2.0 * jnp.pi * t)
H = dq.modulated(f, dq.sigmax())

# call the time array at different times
print(H(0.5))
# [[-0.+0.j -1.+0.j]
#  [-1.+0.j -0.+0.j]]
print(H(1.0))
# [[0.+0.j 1.+0.j]
#  [1.+0.j 0.+0.j]]
```

??? Note "Function with optional arguments"
    To define a modulated time array with additional arguments, you can use the optional `args` parameter of [`dq.modulated()`](../python_api/time_array/modulated.md).
    ```python
    def f(t, omega):
        return jnp.cos(omega * t)

    omega = 1.0
    H = dq.modulated(f, dq.sigmax(), args=(omega,))
    ```

### Arbitrary time-dependent operators

Arbitrary time-dependent operators are operators of the form
$$
 \hat O(t) = f(t)
$$
In dynamiqs, arbitrary time-dependent operators are defined using a Python function with signature `f(t: float, *args: ArrayLike) -> Array` that returns the operator as a JAX array for any time $t$. The function can be passed to [`dq.timecallable()`](../python_api/time_array/timecallable.md) to obtain a `CallableTimeArray` object. For instance, to define the time-dependent Hamiltonian $H = \sigma_z + \cos(2\pi t)\sigma_x$, you can use the following syntax:

```python
# define a callable time array
def _H(t):
    return dq.sigmaz() + jnp.cos(2.0 * jnp.pi * t) * dq.sigmax()
H = dq.timecallable(_H)

# call the time array at different times
print(H(0.5))
# [[ 1.+0.j -1.+0.j]
#  [-1.+0.j -1.+0.j]]
print(H(1.0))
# [[ 1.+0.j  1.+0.j]
#  [ 1.+0.j -1.+0.j]]
```

!!! Warning "Function returning non-array object"
    An error is raised if `H(t)` returns a non-array object (including array-like objects such as QuTiP Qobjs or Python lists). This is enforced to avoid costly conversions at every time step of the numerical integration.

??? Note "Function with optional arguments"
    To define a callable time array with additional arguments, you can use the optional `args` parameter of [`dq.timecallable()`](../python_api/time_array/timecallable.md).

    ```python
    def _H(t, omega):
        return dq.sigmaz() + jnp.cos(omega * t) * dq.sigmax()
    omega = 1.0
    H = dq.timecallable(_H, args=(omega,))
    ```

## Batching and differentiating through a `TimeArray`

For modulated and callable time arrays, it is important that any array you want to batch over or differentiate against is passed as an extra argument to [`dq.timecallable()`](../python_api/time_array/timecallable.md) and [`dq.modulated()`](../python_api/time_array/modulated.md). Such requirements are very aligned with JAX philosophy, in which all internal arguments that are required at runtime should be explicitly provided.

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
