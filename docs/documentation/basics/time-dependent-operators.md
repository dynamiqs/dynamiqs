# Time-dependent operators

This tutorial explains how to define time-dependent Hamiltonians – and more generally time-dependent operators – in Dynamiqs. There are currently four supported formats: constant operator, piecewise constant operator, constant operator modulated by a time-dependent factor, or arbitrary time-dependent operator defined by a function.

```python
import dynamiqs as dq
import jax.numpy as jnp
```

## The [`TimeQArray`][dynamiqs.TimeQArray] type

In Dynamiqs, time-dependent operators are defined with type [`TimeQArray`][dynamiqs.TimeQArray]. They can be called at arbitrary times, and return the corresponding qarray at that time. For example to define the Hamiltonian
$$
    H_x(t)=\cos(2\pi t)\sigma_x
$$
```pycon
>>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
>>> Hx = dq.modulated(f, dq.sigmax())  # initialize a modulated time-qarray
>>> Hx(1.0)
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
[[  ⋅    1.+0.j]
 [1.+0.j   ⋅   ]]
>>> Hx.shape
(2, 2)
```

Time-qarrays support common arithmetic operations with scalars, qarray-likes and other time-qarrays. For example to define the Hamiltonian
$$
    H(t) = \sigma_z + 2 H_x(t) - \sin(\pi t) \sigma_y
$$
```pycon
>>> g = lambda t: jnp.sin(jnp.pi * t)
>>> Hy = dq.modulated(g, dq.sigmay())
>>> H = dq.sigmaz() + 2 * Hx - Hy
>>> H(1.0)
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=3
[[ 1.+0.j  2.-0.j]
 [ 2.+0.j -1.+0.j]]
```

Finally, time-qarrays also support common utility functions, such as `.conj()`, or `.reshape()`. More details can be found in the [`TimeQArray`][dynamiqs.TimeQArray] API page.

## Defining a time-qarray

### Constant operators

A constant operator is defined by
$$
    O(t) = O_0
$$
for any time $t$, where $O_0$ is an arbitrary operator. The most practical way to define constant operators is using qarray-likes. They can also be instantiated as time-qarrays using the [`dq.constant()`][dynamiqs.constant] function. For instance, to define the Pauli operator $H = \sigma_z$, you can use any of the following syntaxes:

=== "Dynamiqs utilities"
    ```python
    import dynamiqs as dq
    H = dq.sigmaz()
    ```
=== "Dynamiqs qarray"
    ```python
    import dynamiqs as dq
    H = dq.asqarray([[1, 0], [0, -1]])
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

!!! Note
    Common operators are available as utility functions, see the list of available operators in the [Python API](../../python_api/index.md#operators).

### Piecewise constant operators

A piecewise constant (PWC) operator takes constant values over some time intervals. It is defined by
$$
    O(t) = \left(\sum_{k=0}^{N-1} c_k\; \Omega_{[t_k, t_{k+1}[}(t)\right) O_0
$$
where $c_k$ are constant values, $\Omega_{[t_k, t_{k+1}[}$ is the rectangular window function defined by $\Omega_{[t_a, t_b[}(t) = 1$ if $t \in [t_a, t_b[$ and $\Omega_{[t_a, t_b[}(t) = 0$ otherwise, and $O_0$ is a constant operator.

In Dynamiqs, PWC operators are defined by:

- `times`: the time points $(t_0, \ldots, t_N)$ defining the boundaries of the time intervals, of shape _(N+1,)_,
- `values`: the constant values $(c_0, \ldots, c_{N-1})$ for each time interval, of shape _(..., N)_,
- `qarray`: the qarray defining the constant operator $O_0$, of shape _(n, n)_.

To construct a PWC operator, these three arguments must be passed to the [`dq.pwc()`][dynamiqs.pwc] function, which returns a time-qarray. When called at some time $t$, this object then returns a qarray with shape _(..., n, n)_. For example, let us define a PWC operator $H(t)$ with constant value $3\sigma_z$ for $t\in[0, 1[$ and $-2\sigma_z$ for $t\in[1, 2[$:
```pycon
>>> times = [0.0, 1.0, 2.0]
>>> values = [3.0, -2.0]
>>> qarray = dq.sigmaz()
>>> H = dq.pwc(times, values, qarray)
>>> H
PWCTimeQArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
```

The returned object can be called at different times:
=== "$t = -1.0$"
    ```pycon
    >>> H(-1.0)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[  ⋅      ⋅   ]
     [  ⋅      ⋅   ]]
    ```
=== "$t=0.0$"
    ```pycon
    >>> H(0.0)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[ 3.+0.j    ⋅   ]
     [   ⋅    -3.+0.j]]
    ```
=== "$t=0.5$"
    ```pycon
    >>> H(0.5)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[ 3.+0.j    ⋅   ]
     [   ⋅    -3.+0.j]]
    ```
=== "$t=1.0$"
    ```pycon
    >>> H(1.0)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[-2.+0.j    ⋅   ]
     [   ⋅     2.+0.j]]
    ```
=== "$t=1.5$"
    ```pycon
    >>> H(1.5)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[-2.+0.j    ⋅   ]
     [   ⋅     2.+0.j]]
    ```
=== "$t=2.0$"
    ```pycon
    >>> H(2.0)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[  ⋅      ⋅   ]
     [  ⋅      ⋅   ]]
    ```

!!! Note
    The argument `times` must be sorted in ascending order, but does not need to be evenly spaced. When calling the resulting time-qarray at time $t$, the returned qarray is the operator $c_k\ O_0$ corresponding to the interval $[t_k, t_{k+1}[$ in which the time $t$ falls. If $t$ does not belong to any time intervals, the returned qarray is null.

??? Note "Batching PWC operators"
    The batching of the returned time-qarray is specified by `values`. For example, to define a PWC operator batched over a parameter $\theta$:
    ```pycon
    >>> thetas = jnp.linspace(0, 1.0, 11)  # (11,)
    >>> times = [0.0, 1.0, 2.0]
    >>> values = thetas[:, None] * jnp.array([3.0, -2.0])  # (11, 2)
    >>> qarray = dq.sigmaz()
    >>> H = dq.pwc(times, values, qarray)
    >>> H.shape
    (11, 2, 2)
    ```

### Modulated operators

A modulated operator is defined by
$$
    O(t) = f(t) O_0
$$
where $f(t)$ is a time-dependent scalar. In Dynamiqs, modulated operators are defined by:

- `f`: a Python function with signature `f(t: float) -> Scalar | Array` that returns the modulating factor $f(t)$ for any time $t$, as a scalar or an array of shape _(...)_,
- `qarray`: the qarray defining the constant operator $O_0$, of shape _(n, n)_.

To construct a modulated operator, these two arguments must be passed to the [`dq.modulated()`][dynamiqs.modulated] function, which returns a time-qarray. When called at some time $t$, this object then returns a qarray with shape _(..., n, n)_. For example, let us define the modulated operator $H(t)=\cos(2\pi t)\sigma_x$:
```pycon
>>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
>>> H = dq.modulated(f, dq.sigmax())
>>> H
ModulatedTimeQArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
```

The returned object can be called at different times:
=== "$t = 0.5$"
    ```pycon
    >>> H(0.5)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
    [[   ⋅    -1.+0.j]
     [-1.+0.j    ⋅   ]]
    ```
=== "$t=1.0$"
    ```pycon
    >>> H(1.0)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
    [[  ⋅    1.+0.j]
     [1.+0.j   ⋅   ]]
    ```

??? Note "Batching modulated operators"
    The batching of the returned time-qarray is specified by the array returned by `f`. For example, to define a modulated Hamiltonian $H(t)=\cos(\omega t)\sigma_x$ batched over the parameter $\omega$:
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
    argument `discontinuity_ts` to enforce adaptive step size methods to stop at these
    times (i.e., right before, and right after the jump).

### Arbitrary time-dependent operators

An arbitrary time-dependent operator is defined by
$$
    O(t) = f(t)
$$
where $f(t)$ is a time-dependent operator. In Dynamiqs, arbitrary time-dependent operators are defined by:

- `f`: a Python function with signature `f(t: float) -> QArray` that returns the operator $f(t)$ for any time $t$, as a qarray of shape _(..., n, n)_.

To construct an arbitrary time-dependent operator, pass this argument to the [`dq.timecallable()`][dynamiqs.timecallable] function, which returns a time-qarray. This object then returns a qarray with shape _(..., n, n)_ when called at any time $t$.

For example, let us define the arbitrary time-dependent operator $H(t)=\begin{pmatrix}t & 0\\0 & 1 - t\end{pmatrix}$:
```pycon
>>> f = lambda t: dq.asqarray([[t, 0], [0, 1 - t]])
>>> H = dq.timecallable(f)
>>> H
CallableTimeQArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
```

The returned object can be called at different times:
=== "$t = 0.5$"
    ```pycon
    >>> H(0.5)
    QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
    [[0.5 0. ]
     [0.  0.5]]
    ```
=== "$t=1.0$"
    ```pycon
    >>> H(1.0)
    QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
    [[1. 0.]
     [0. 0.]]
    ```

!!! Warning "The function `f` must return a qarray (not a qarray-like!)"
    An error is raised if the function `f` does not return a qarray. This error concerns any other qarray-likes. This is enforced to avoid costly conversions at every time step of the numerical integration.

??? Note "Batching arbitrary time-dependent operators"
    The batching of the returned time-qarray is specified by the qarray returned by `f`. For example, to define an arbitrary time-dependent operator batched over a parameter $\theta$:
    ```pycon
    >>> thetas = jnp.linspace(0, 1.0, 11)  # (11,)
    >>> f = lambda t: thetas[:, None, None] * dq.asqarray([[t, 0], [0, 1 - t]])
    >>> H = dq.timecallable(f)
    >>> H.shape
    (11, 2, 2)
    ```

??? Note "Function with additional arguments"
    To define an arbitrary time-dependent operator with a function that takes arguments other than time (extra `*args` and `**kwargs`), you can use [`functools.partial()`](https://docs.python.org/3/library/functools.html#functools.partial). For example:
    ```pycon
    >>> import functools
    >>> def func(t, a, amplitude=1.0):
    ...     return amplitude * dq.asqarray([[t, a], [a, 1 - t]])
    >>> # create function with correct signature (t: float) -> Array
    >>> f = functools.partial(func, a=1.0, amplitude=5.0)
    >>> H = dq.timecallable(f)
    ```

??? Note "Discontinuous function"
    If there is a discontinuous jump in the function values, you should use the optional
    argument `discontinuity_ts` to enforce adaptive step size methods to stop at these
    times (i.e., right before, and right after the jump).

## Clipping a time-qarray

Time-dependent operators can be clipped to a given time interval, outside which the
returned qarray is null. For example:
```pycon
>>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
>>> H = dq.modulated(f, dq.sigmax())
>>> H(2.0)
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
[[  ⋅    1.+0.j]
 [1.+0.j   ⋅   ]]
>>> H = H.clip(0, 1)  # clip to 0 <= t < 1
>>> H(2.0)
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
[[  ⋅      ⋅   ]
 [  ⋅      ⋅   ]]
```
