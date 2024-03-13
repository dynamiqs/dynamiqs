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
>>> H1 = dq.modulated(lambda t: jnp.cos(2.0 * jnp.pi * t), dq.sigmax()) # modulated time-array
>>> H = H0 + H1
>>> H(1.0)
Array([[ 1.+0.j,  1.+0.j],
       [ 1.+0.j, -1.+0.j]], dtype=complex64)
```

Finally, time-arrays also support common utility functions, such as `.conj()`, or `.reshape()`. More details can be found in the corresponding [`TimeArray`][dynamiqs.TimeArray] API page.

## Defining a [`TimeArray`][dynamiqs.TimeArray]

### Constant operators

A constant operator is defined by
$$
    O(t) = O_0
$$
for any time $t$, where $O_0$ is a constant operator.

In dynamiqs, constant operators can either be defined with **array-like objects** (e.g. Python lists, NumPy and JAX arrays, QuTiP Qobjs) or as a [`TimeArray`][dynamiqs.TimeArray] object (using the [`dq.constant()`][dynamiqs.constant] function).

!!! Notes
    Common operators are available as utility functions, see the list of available operators in the [Python API](../python_api/index.md#operators).

For instance, to define the Pauli $Z$ operator $H = \sigma_z$, you can use any of the following syntaxes:

```python
# Python lists
H = [[1, 0], [0, -1]]

# NumPy arrays
import numpy as np
H = np.array([[1, 0], [0, -1]])

# JAX arrays
import jax.numpy as jnp
H = jnp.array([[1, 0], [0, -1]])

# QuTiP Qobjs
import qutip as qt
H = qt.sigmaz()

# dynamiqs utility function
import dynamiqs as dq
H = dq.sigmaz()

# constant time array
H = dq.constant(dq.sigmaz())
```

### Piecewise constant operators

A piecewise constant (PWC) operator takes constant values over some time intervals. It is defined by
$$
    O(t) = \left(\sum_{k=0}^{N-1} c_k\; \Omega_{[t_k, t_{k+1}[}(t)\right) O_0
$$
where $c_k$ are constant values, $\Omega_{[t_k, t_{k+1}[}$ is the rectangular window function defined by $\Omega_{[t_a, t_b[}(t) = 1$ if $t \in [t_a, t_b[$ and $\Omega_{[t_a, t_b[}(t) = 0$ otherwise, and $O_0$ is a constant operator.

In dynamiqs, PWC operators are defined by three **array-like objects** (e.g. Python lists, NumPy and JAX arrays, QuTiP Qobjs):

- `times`: the time points $(t_0, \ldots, t_N)$ defining the boundaries of the time intervals, of shape _(N+1,)_,
- `values`: the constant values $(c_0, \ldots, c_{N-1})$ for each time interval, of shape _(..., N)_,
- `array`: the array defining the constant operator $O_0$, of shape _(n, n)_.

To construct a PWC operator, pass these three arguments to the [`dq.pwc()`][dynamiqs.pwc] function, which returns a [`TimeArray`][dynamiqs.TimeArray] object.

!!! Notes
    The argument `times` must be sorted in ascending order, but does not need to be evenly spaced. When calling the resulting time array object at time $t$, the returned array is the operator $c_k\ O_0$ corresponding to the interval $[t_k, t_{k+1}[$ in which the time $t$ falls. If $t$ does not belong to any time intervals, the returned array is null.

Let's define a PWC operator:
```pycon
>>> times = jnp.array([0.0, 1.0, 2.0])
>>> values = jnp.array([3.0, -2.0])
>>> array = dq.sigmaz()
>>> H = dq.pwc(times, values, array)
>>> type(H)
<class 'dynamiqs.time_array.PWCTimeArray'>
>>> H.shape
(2, 2)
```

The returned object can be called at different times:
```pycon
>>> H(-1.0)
Array([[ 0.+0.j,  0.+0.j],
       [ 0.+0.j, -0.+0.j]], dtype=complex64)
>>> H(0.0)
Array([[ 3.+0.j,  0.+0.j],
       [ 0.+0.j, -3.+0.j]], dtype=complex64)
>>> H(0.5)
Array([[ 3.+0.j,  0.+0.j],
       [ 0.+0.j, -3.+0.j]], dtype=complex64)
>>> H(1.0)
Array([[-2.+0.j, -0.+0.j],
       [-0.+0.j,  2.-0.j]], dtype=complex64)
>>> H(1.5)
Array([[-2.+0.j, -0.+0.j],
       [-0.+0.j,  2.-0.j]], dtype=complex64)
>>> H(2.0)
Array([[ 0.+0.j,  0.+0.j],
       [ 0.+0.j, -0.+0.j]], dtype=complex64)
```

### Modulated operators

A modulated operator is defined by
$$
    O(t) = f(t) O_0
$$
where $f(t)$ is an time-dependent scalar.

In dynamiqs, modulated operators are defined by:

- `f`: a Python function with signature `f(t: float, *args: ArrayLike) -> Array` that returns the modulating factor $f(t)$ for any time $t$, as an array of shape _(...)_,
- `array`: the array defining the constant operator $O_0$, of shape _(n, n)_.

To construct a modulated operator, pass these two arguments to the [`dq.modulated()`][dynamiqs.modulated] function, which returns a [`TimeArray`][dynamiqs.TimeArray] object.

Let's define the modulated operator $H=\cos(2\pi t)\sigma_x$:
```python
>>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
>>> H = dq.modulated(f, dq.sigmax())
>>> type(H)
<class 'dynamiqs.time_array.ModulatedTimeArray'>
>>> H.shape
(2, 2)
```

The returned object can be called at different times:
```pycon
>>> H(0.5)
Array([[-0.+0.j, -1.+0.j],
       [-1.+0.j, -0.+0.j]], dtype=complex64)
>>> H(1.0)
Array([[0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j]], dtype=complex64)
```

??? Note "Function with optional arguments"
    To define a modulated time array with additional arguments, you can use the optional `args` parameter of [`dq.modulated()`][dynamiqs.modulated]:
    ```python
    f = lambda t, omega: jnp.cos(omega * t)
    omega = 1.0
    H = dq.modulated(f, dq.sigmax(), args=(omega,))
    ```

### Arbitrary time-dependent operators

An arbitrary time-dependent operator is defined by
$$
    O(t) = f(t)
$$
where $f(t)$ is a time-dependent operator.

In dynamiqs, arbitrary time-dependent operators are defined by a Python function with signature `f(t: float, *args: ArrayLike) -> Array` that returns the operator $f(t)$ for any time $t$, as an array of shape _(..., n, n)_. To construct an arbitrary time-dependent operator, pass this argument to the [`dq.timecallable()`][dynamiqs.timecallable] function, which returns a [`TimeArray`][dynamiqs.TimeArray] object.

Let's define the arbitrary time-dependent operator $H=\begin{pmatrix}t & 0\\0 & 1 - t\end{pmatrix}$:
```pycon
>>> f = lambda t: jnp.array([[t, 0], [0, 1 - t]])
>>> H = dq.timecallable(f)
>>> type(H)
<class 'dynamiqs.time_array.CallableTimeArray'>
>>> H.shape
(2, 2)
```

The returned object can be called at different times:
```pycon
>>> H(0.5)
Array([[0.5, 0. ],
       [0. , 0.5]], dtype=float32)
>>> H(1.0)
Array([[1., 0.],
       [0., 0.]], dtype=float32)
```

!!! Warning "The function must return a JAX array"
    An error is raised if the function `f` does not return a JAX array. This error includes other array-like objects (e.g. Python lists, NumPy arrays or QuTiP Qobjs). This is enforced to avoid costly conversions at every time step of the numerical integration.

??? Note "Function with optional arguments"
    To define a callable time array with additional arguments, you can use the optional `args` parameter of [`dq.timecallable()`][dynamiqs.timecallable]:
    ```python
    f = lambda t, x: x * jnp.array([[t, 0], [0, 1 - t]])
    x = 1.0
    H = dq.timecallable(f, args=(x,))
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
