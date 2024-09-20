# Time-dependent operators

This tutorial explains how to define time-dependent Hamiltonians – and more generally time-dependent operators – in Dynamiqs. There are currently four supported formats: constant operator, piecewise constant operator, constant operator modulated by a time-dependent factor, or arbitrary time-dependent operator defined by a function.

!!! Warning "Differences with QuTiP"
    Dynamiqs uses JAX arrays, which are different from QuTiP quantum objects. See [The sharp bits 🔪](../getting_started/sharp-bits.md) page for more details, briefly:

    - use `x + 2 * dq.eye(n)` instead of `x + 2`
    - use `x @ y` instead of `x * y`, and `dq.powm(x, 4)` instead of `x**4`
    - use `dq.dag(x)` or `x.mT.conj()` instead of `x.dag()`

```python
import dynamiqs as dq
import jax.numpy as jnp
```

## The [`TimeArray`][dynamiqs.TimeArray] type

In Dynamiqs, time-dependent operators are defined with [`TimeArray`][dynamiqs.TimeArray] objects. These objects can be called at arbitrary times, and return the corresponding array at that time. For example to define the Hamiltonian
$$
    H_x(t)=\cos(2\pi t)\sigma_x
$$
```pycon
>>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
>>> Hx = dq.modulated(f, dq.sigmax())  # initialize a modulated time-array
>>> Hx(1.0)
Array([[0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j]], dtype=complex64)
>>> Hx.shape
(2, 2)
```

Time-arrays support common arithmetic operations with scalars, regular arrays and other time-array objects. For example to define the Hamiltonian
$$
    H(t) = \sigma_z + 2 H_x(t) - \sin(\pi t) \sigma_y
$$
```pycon
>>> g = lambda t: jnp.sin(jnp.pi * t)
>>> Hy = dq.modulated(g, dq.sigmay())
>>> H = dq.sigmaz() + 2 * Hx - Hy
>>> H(1.0)
Array([[ 1.+0.j,  2.-0.j],
       [ 2.+0.j, -1.+0.j]], dtype=complex64)
```

Finally, time-arrays also support common utility functions, such as `.conj()`, or `.reshape()`. More details can be found in the [`TimeArray`][dynamiqs.TimeArray] API page.

## Defining a [`TimeArray`][dynamiqs.TimeArray]

### Constant operators

A constant operator is defined by
$$
    O(t) = O_0
$$
for any time $t$, where $O_0$ is a constant operator.

In Dynamiqs, constant operators can either be defined with array-like objects or as [`TimeArray`][dynamiqs.TimeArray] objects (using the [`dq.constant()`][dynamiqs.constant] function).

!!! Note
    Common operators are available as utility functions, see the list of available operators in the [Python API](../../python_api/index.md#operators).

For instance, to define the Pauli $Z$ operator $H = \sigma_z$, you can use any of the following syntaxes:

=== "Dynamiqs"
    ```python
    import dynamiqs as dq
    H = dq.sigmaz()
    ```
=== "NumPy array"
    ```python
    import numpy as np
    H = np.array([[1, 0], [0, -1]])
    ```
=== "JAX array"
    ```python
    import jax.numpy as jnp
    H = jnp.array([[1, 0], [0, -1]])
    ```
=== "QuTiP Qobj"
    ```python
    import qutip as qt
    H = qt.sigmaz()
    ```
=== "Python list"
    ```python
    H = [[1, 0], [0, -1]]
    ```

### Piecewise constant operators

A piecewise constant (PWC) operator takes constant values over some time intervals. It is defined by
$$
    O(t) = \left(\sum_{k=0}^{N-1} c_k\; \Omega_{[t_k, t_{k+1}[}(t)\right) O_0
$$
where $c_k$ are constant values, $\Omega_{[t_k, t_{k+1}[}$ is the rectangular window function defined by $\Omega_{[t_a, t_b[}(t) = 1$ if $t \in [t_a, t_b[$ and $\Omega_{[t_a, t_b[}(t) = 0$ otherwise, and $O_0$ is a constant operator.

In Dynamiqs, PWC operators are defined by three array-like objects:

- `times`: the time points $(t_0, \ldots, t_N)$ defining the boundaries of the time intervals, of shape _(N+1,)_,
- `values`: the constant values $(c_0, \ldots, c_{N-1})$ for each time interval, of shape _(..., N)_,
- `array`: the array defining the constant operator $O_0$, of shape _(n, n)_.

To construct a PWC operator, pass these three arguments to the [`dq.pwc()`][dynamiqs.pwc] function, which returns a [`TimeArray`][dynamiqs.TimeArray] object. This object then returns an array with shape _(..., n, n)_ when called at any time $t$.

!!! Note
    The argument `times` must be sorted in ascending order, but does not need to be evenly spaced. When calling the resulting time-array object at time $t$, the returned array is the operator $c_k\ O_0$ corresponding to the interval $[t_k, t_{k+1}[$ in which the time $t$ falls. If $t$ does not belong to any time intervals, the returned array is null.

Let's define a PWC operator:
```pycon
>>> times = [0.0, 1.0, 2.0]
>>> values = [3.0, -2.0]
>>> array = dq.sigmaz()
>>> H = dq.pwc(times, values, array)
>>> H
PWCTimeArray(shape=(2, 2), dtype=complex64)
```

The returned object can be called at different times:
=== "$t = -1.0$"
    ```pycon
    >>> H(-1.0)
    Array([[ 0.+0.j,  0.+0.j],
           [ 0.+0.j, -0.+0.j]], dtype=complex64)
    ```
=== "$t=0.0$"
    ```pycon
    >>> H(0.0)
    Array([[ 3.+0.j,  0.+0.j],
           [ 0.+0.j, -3.+0.j]], dtype=complex64)
    ```
=== "$t=0.5$"
    ```pycon
    >>> H(0.5)
    Array([[ 3.+0.j,  0.+0.j],
           [ 0.+0.j, -3.+0.j]], dtype=complex64)
    ```
=== "$t=1.0$"
    ```pycon
    >>> H(1.0)
    Array([[-2.+0.j, -0.+0.j],
           [-0.+0.j,  2.-0.j]], dtype=complex64)
    ```
=== "$t=1.5$"
    ```pycon
    >>> H(1.5)
    Array([[-2.+0.j, -0.+0.j],
           [-0.+0.j,  2.-0.j]], dtype=complex64)
    ```
=== "$t=2.0$"
    ```pycon
    >>> H(2.0)
    Array([[ 0.+0.j,  0.+0.j],
           [ 0.+0.j, -0.+0.j]], dtype=complex64)
    ```

??? Note "Batching PWC operators"
    The batching of the returned time-array is specified by `values`. For example, to define a PWC operator batched over a parameter $\theta$:
    ```pycon
    >>> thetas = jnp.linspace(0, 1.0, 11)  # (11,)
    >>> times = [0.0, 1.0, 2.0]
    >>> values = thetas[:, None] * jnp.array([3.0, -2.0])  # (11, 2)
    >>> array = dq.sigmaz()
    >>> H = dq.pwc(times, values, array)
    >>> H.shape
    (11, 2, 2)
    ```

### Modulated operators

A modulated operator is defined by
$$
    O(t) = f(t) O_0
$$
where $f(t)$ is an time-dependent scalar.

In Dynamiqs, modulated operators are defined by:

- `f`: a Python function with signature `f(t: float) -> Scalar | Array` that returns the modulating factor $f(t)$ for any time $t$, as a scalar or an array of shape _(...)_,
- `array`: the array defining the constant operator $O_0$, of shape _(n, n)_.

To construct a modulated operator, pass these two arguments to the [`dq.modulated()`][dynamiqs.modulated] function, which returns a [`TimeArray`][dynamiqs.TimeArray] object. This object then returns an array with shape _(..., n, n)_ when called at any time $t$.

Let's define the modulated operator $H(t)=\cos(2\pi t)\sigma_x$:
```pycon
>>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
>>> H = dq.modulated(f, dq.sigmax())
>>> H
ModulatedTimeArray(shape=(2, 2), dtype=complex64)
```

The returned object can be called at different times:
=== "$t = 0.5$"
    ```pycon
    >>> H(0.5)
    Array([[-0.+0.j, -1.+0.j],
           [-1.+0.j, -0.+0.j]], dtype=complex64)
    ```
=== "$t=1.0$"
    ```pycon
    >>> H(1.0)
    Array([[0.+0.j, 1.+0.j],
           [1.+0.j, 0.+0.j]], dtype=complex64)
    ```

??? Note "Batching modulated operators"
    The batching of the returned time-array is specified by the array returned by `f`. For example, to define a modulated Hamiltonian $H(t)=\cos(\omega t)\sigma_x$ batched over the parameter $\omega$:
    ```pycon
    >>> omegas = jnp.linspace(0.0, 1.0, 11)  # (11,)
    >>> f = lambda t: jnp.cos(omegas * t)
    >>> H = dq.modulated(f, dq.sigmax())
    >>> H.shape
    (11, 2, 2)
    ```

??? Note "Function with additional arguments"
    To define a modulated operator with a function that takes arguments other than time (extra `*args` and `**kwargs`), you can use [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial). For example:
    ```pycon
    >>> import functools
    >>> def pulse(t, omega, amplitude=1.0):
    ...     return amplitude * jnp.cos(omega * t)
    >>> # create function with correct signature (t: float) -> Array
    >>> f = functools.partial(pulse, omega=1.0, amplitude=5.0)
    >>> H = dq.modulated(f, dq.sigmax())
    ```

??? Note "Discontinuous function"
    If there is a discontinuous jump in the function values, you should use the optional
    argument `discontinuity_ts` to enforce adaptive step size solvers to stop at these
    times (i.e., right before, and right after the jump).

### Arbitrary time-dependent operators

An arbitrary time-dependent operator is defined by
$$
    O(t) = f(t)
$$
where $f(t)$ is a time-dependent operator.

In Dynamiqs, arbitrary time-dependent operators are defined by:

- `f`: a Python function with signature `f(t: float) -> Array` that returns the operator $f(t)$ for any time $t$, as an array of shape _(..., n, n)_.

To construct an arbitrary time-dependent operator, pass this argument to the [`dq.timecallable()`][dynamiqs.timecallable] function, which returns a [`TimeArray`][dynamiqs.TimeArray] object. This object then returns an array with shape _(..., n, n)_ when called at any time $t$.

Let's define the arbitrary time-dependent operator $H(t)=\begin{pmatrix}t & 0\\0 & 1 - t\end{pmatrix}$:
```pycon
>>> f = lambda t: jnp.array([[t, 0], [0, 1 - t]])
>>> H = dq.timecallable(f)
>>> H
CallableTimeArray(shape=(2, 2), dtype=float32)
```

The returned object can be called at different times:
=== "$t = 0.5$"
    ```pycon
    >>> H(0.5)
    Array([[0.5, 0. ],
           [0. , 0.5]], dtype=float32)
    ```
=== "$t=1.0$"
    ```pycon
    >>> H(1.0)
    Array([[1., 0.],
           [0., 0.]], dtype=float32)
    ```

!!! Warning "The function `f` must return a JAX array (not an array-like object!)"
    An error is raised if the function `f` does not return a JAX array. This error concerns any other array-like objects. This is enforced to avoid costly conversions at every time step of the numerical integration.

??? Note "Batching arbitrary time-dependent operators"
    The batching of the returned time-array is specified by the array returned by `f`. For example, to define an arbitrary time-dependent operator batched over a parameter $\theta$:
    ```pycon
    >>> thetas = jnp.linspace(0, 1.0, 11)  # (11,)
    >>> f = lambda t: thetas[:, None, None] * jnp.array([[t, 0], [0, 1 - t]])
    >>> H = dq.timecallable(f)
    >>> H.shape
    (11, 2, 2)
    ```

??? Note "Function with additional arguments"
    To define an arbitrary time-dependent operator with a function that takes arguments other than time (extra `*args` and `**kwargs`), you can use [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial). For example:
    ```pycon
    >>> import functools
    >>> def func(t, a, amplitude=1.0):
    ...     return amplitude * jnp.array([[t, a], [a, 1 - t]])
    >>> # create function with correct signature (t: float) -> Array
    >>> f = functools.partial(func, a=1.0, amplitude=5.0)
    >>> H = dq.timecallable(f)
    ```

??? Note "Discontinuous function"
    If there is a discontinuous jump in the function values, you should use the optional
    argument `discontinuity_ts` to enforce adaptive step size solvers to stop at these
    times (i.e., right before, and right after the jump).
