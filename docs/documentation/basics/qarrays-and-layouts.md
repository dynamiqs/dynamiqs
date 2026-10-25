# Quantum arrays and layouts

In Dynamiqs, states and operators are represented by the [`QArray`][dynamiqs.QArray] type. This tutorial explains what a qarray is, how to combine qarrays, and how they are stored in memory, either as dense arrays or in a sparse diagonal format.

```python
import dynamiqs as dq
import jax
import jax.numpy as jnp
import qutip as qt
```

## The [`QArray`][dynamiqs.QArray] type

A qarray wraps a JAX array, and adds to it the information needed to treat it as a quantum object: the Hilbert space dimension of each subsystem (`dims`), and the way the data is stored (`layout`). All Dynamiqs functions that create states or operators return a qarray:

```pycon
>>> psi = dq.fock(3, 1)
>>> psi
QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
[[0.+0.j]
 [1.+0.j]
 [0.+0.j]]
>>> psi.shape, psi.dims, psi.dtype, psi.layout
((3, 1), (3,), dtype('complex64'), dense)
```

### Shapes

Qarrays always have at least two dimensions, and the last two dimensions are those of the quantum object: a ket has shape `(n, 1)`, a bra `(1, n)`, and a density matrix or an operator `(n, n)`. Any leading dimension is a **batch dimension**, which holds several objects of the same kind. For example, a batch of three coherent states:

```pycon
>>> psis = dq.coherent(4, [0.0, 1.0, 2.0])
>>> psis.shape
(3, 4, 1)
```

Shapes are thus written `(..., n, 1)` for kets and `(..., n, n)` for operators throughout the documentation, where `...` stands for any number of batch dimensions. Solvers take advantage of them to run many simulations at once, see the [Batching simulations](batching-simulations.md) tutorial.

Indexing a qarray along its batch dimensions returns a qarray, but indexing into the last two dimensions returns a plain JAX array, since the result is no longer a quantum object:

```pycon
>>> psis[1].shape
(4, 1)
>>> type(psis[1, :, 0])
<class 'jaxlib._jax.ArrayImpl'>
```

### Creating a qarray from other types

Most functions accept a **qarray-like**: a qarray, a NumPy or JAX array, a QuTiP `Qobj`, or a nested Python list of these. You rarely need to convert them yourself, but [`dq.asqarray()`][dynamiqs.asqarray] does it explicitly:

```pycon
>>> dq.asqarray([[0, 1], [1, 0]])
QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dense
[[0 1]
 [1 0]]
```

Qarrays are JAX [PyTrees](https://docs.jax.dev/en/latest/pytrees.html), so they can be passed to and returned from functions transformed by `jax.jit`, `jax.grad` or `jax.vmap`, like any JAX array:

```pycon
>>> a = dq.destroy(3)
>>> jax.jit(lambda a: a.dag() @ a)(a)
QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dia, ndiags=1
[[  ⋅      ⋅      ⋅   ]
 [  ⋅    1.+0.j   ⋅   ]
 [  ⋅      ⋅    2.+0.j]]
```

## Composite systems

The `dims` attribute stores the Hilbert space dimension of each subsystem. It is set by [`dq.tensor()`][dynamiqs.tensor], or equivalently by the `&` operator, which compute the tensor product of their arguments:

```pycon
>>> psi = dq.tensor(dq.fock(3, 0), dq.fock(2, 1))  # |0⟩ ⊗ |1⟩
>>> psi.shape, psi.dims
((6, 1), (3, 2))
>>> (dq.destroy(3) & dq.eye(2)).dims
(3, 2)
```

Operator functions also build the operators of a composite system directly when they are passed several dimensions. For example, [`dq.destroy()`][dynamiqs.destroy] then returns one annihilation operator per subsystem, already embedded in the full Hilbert space:

```pycon
>>> a, b = dq.destroy(3, 2)
>>> a.shape, a.dims
((6, 6), (3, 2))
```

Thanks to `dims`, you do not need to repeat the system structure when taking a partial trace with [`dq.ptrace()`][dynamiqs.ptrace] or the `.ptrace()` method. Here, we keep subsystem `1` (the qubit) and trace out the oscillator:

```pycon
>>> psi.ptrace(1)
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
[[0.+0.j 0.+0.j]
 [0.+0.j 1.+0.j]]
```

Arrays and lists do not carry this information, so you must specify it with the `dims` argument of [`dq.asqarray()`][dynamiqs.asqarray] or [`dq.ptrace()`][dynamiqs.ptrace]:

```pycon
>>> psi_array = jnp.ones((6, 1)) / jnp.sqrt(6)
>>> dq.ptrace(psi_array, 0, dims=(3, 2))
QArray: shape=(3, 3), dims=(3,), dtype=float32, layout=dense
[[0.333 0.333 0.333]
 [0.333 0.333 0.333]
 [0.333 0.333 0.333]]
```

Qarrays with different `dims` cannot be combined, even if their shapes match:

```pycon
>>> dq.eye(2, 2) + dq.eye(4)
Traceback (most recent call last):
    ...
ValueError: Qarrays have incompatible Hilbert space dimensions. Got (2, 2) and (4,).
```

## Useful methods

Most quantum utilities can be called as methods of a qarray, for example `x.dag()` instead of `dq.dag(x)`. The most common ones are:

| Method | Description |
| --- | --- |
| `x.dag()` | Adjoint, see [`dq.dag()`][dynamiqs.dag]. |
| `x.trace()` | Trace, see [`dq.trace()`][dynamiqs.trace]. |
| `x.ptrace(*keep)` | Partial trace, see [`dq.ptrace()`][dynamiqs.ptrace]. |
| `x.norm()`, `x.unit()` | Norm and normalization, see [`dq.norm()`][dynamiqs.norm] and [`dq.unit()`][dynamiqs.unit]. |
| `x.expm()`, `x.powm(n)` | Matrix exponential and power, see [`dq.expm()`][dynamiqs.expm] and [`dq.powm()`][dynamiqs.powm]. |
| `x.toket()`, `x.tobra()`, `x.todm()` | Conversion between state representations, see [`dq.todm()`][dynamiqs.todm]. |
| `x.isket()`, `x.isdm()`, `x.isherm()` | Checks on the kind of object, see [`dq.isket()`][dynamiqs.isket] and [`dq.isherm()`][dynamiqs.isherm]. |
| `x.to_jax()`, `x.to_numpy()`, `x.to_qutip()` | Conversion to other array types, see [`dq.to_jax()`][dynamiqs.to_jax], [`dq.to_numpy()`][dynamiqs.to_numpy] and [`dq.to_qutip()`][dynamiqs.to_qutip]. |
| `x.asdense()`, `x.assparsedia()` | Conversion between [layouts](#dense-and-sparse-diagonal-layouts). |

For instance, to normalize a state and compute its density matrix:

```pycon
>>> (dq.fock(2, 0) + dq.fock(2, 1)).unit().todm()
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
[[0.5+0.j 0.5+0.j]
 [0.5+0.j 0.5+0.j]]
```

The full list of methods is given in the [`QArray`][dynamiqs.QArray] API page.

## Arithmetic operations

Qarrays follow the conventions of NumPy and JAX rather than those of QuTiP, and forbid the operations whose meaning would be ambiguous between the two:

| Operation | Supported operand `y` | Result |
| --- | --- | --- |
| `x + y`, `x - y` | qarray-like with the same `dims` and a broadcastable shape | element-wise sum |
| `x * y`, `x / y` | scalar, or array of shape `(..., 1, 1)` | element-wise product with a scalar |
| `x @ y` | qarray-like | matrix product |
| `x & y` | qarray | tensor product |
| `-x` | | negation |

In particular, the following operations raise an error, with a message pointing to the explicit alternative:

- `x + 2` and `x - 2`: write `x + 2 * dq.eye_like(x)` to add a multiple of the identity, or `x.addscalar(2)` to add a scalar to every element,
- `x * y` with a qarray-like `y`: write `x @ y` for the matrix product, or `x.elmul(y)` for the element-wise product,
- `x ** 2`: write `x.powm(2)` for the matrix power, or `x.elpow(2)` for the element-wise power,
- `x / y` with a qarray-like `y`, and `2 / x`.

These differences with QuTiP are explained in detail in [the sharp bits](../getting_started/sharp-bits.md#main-differences-with-qutip).

Since an array of shape `(..., 1, 1)` counts as a scalar, multiplication is the natural way to build a batch of operators from an array of parameters:

```pycon
>>> omegas = jnp.array([1.0, 2.0, 3.0])
>>> H = omegas[:, None, None] * dq.sigmaz()
>>> H.shape
(3, 2, 2)
```

Multiplying by `omegas` directly, of shape `(3,)`, raises an error instead.

The product of a bra and a ket returns a JAX array of shape `(..., 1, 1)` rather than a qarray, since the result is a number:

```pycon
>>> psi = dq.fock(2, 0)
>>> psi.dag() @ psi
Array([[1.+0.j]], dtype=complex64)
```

## Dense and sparse diagonal layouts

The data of a qarray is stored in one of two layouts:

- [`dq.dense`][dynamiqs.dense]: a dense JAX array of shape `(..., n, m)`, which stores every element,
- [`dq.dia`][dynamiqs.dia]: a sparse diagonal format, which only stores the non-zero diagonals of a square matrix, and their offsets from the main diagonal.

The layout is shown when printing a qarray, together with the number of stored diagonals for the `dia` layout. Zero elements outside the stored diagonals are displayed as `⋅`:

```pycon
>>> dq.destroy(4)
QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
[[    ⋅     1.   +0.j     ⋅         ⋅    ]
 [    ⋅         ⋅     1.414+0.j     ⋅    ]
 [    ⋅         ⋅         ⋅     1.732+0.j]
 [    ⋅         ⋅         ⋅         ⋅    ]]
```

Many operators of quantum physics are banded: the ladder and number operators, the Pauli matrices, and tensor products and polynomials of these only have a few non-zero diagonals. Storing them in the `dia` layout saves memory, and makes their products with a dense state cost $O(dn^2)$ instead of $O(n^3)$ for an operator with $d$ diagonals, which is why it is the default for operators.

### Which layout is used by default

- **Operators that accept a `layout` argument** return the `dia` layout by default. This includes [`dq.eye()`][dynamiqs.eye], [`dq.zeros()`][dynamiqs.zeros], [`dq.destroy()`][dynamiqs.destroy], [`dq.create()`][dynamiqs.create], [`dq.number()`][dynamiqs.number], [`dq.parity()`][dynamiqs.parity], [`dq.position()`][dynamiqs.position], [`dq.momentum()`][dynamiqs.momentum] and the Pauli matrices such as [`dq.sigmax()`][dynamiqs.sigmax].
- **Other operators** are dense, because they are not banded in general, for example [`dq.displace()`][dynamiqs.displace], [`dq.squeeze()`][dynamiqs.squeeze], [`dq.hadamard()`][dynamiqs.hadamard] or [`dq.cnot()`][dynamiqs.cnot].
- **States** such as [`dq.fock()`][dynamiqs.fock], [`dq.coherent()`][dynamiqs.coherent] or [`dq.coherent_dm()`][dynamiqs.coherent_dm] are dense, as are the states returned by solvers.
- **Conversions** with [`dq.asqarray()`][dynamiqs.asqarray] return a dense qarray, except when the input is already a qarray (returned unchanged), or a list of `dia` qarrays with identical shapes and `dims` (stacked in the `dia` layout).

The `layout` argument selects the layout of a single operator:

```pycon
>>> dq.destroy(4, layout=dq.dense)
QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
[[0.   +0.j 1.   +0.j 0.   +0.j 0.   +0.j]
 [0.   +0.j 0.   +0.j 1.414+0.j 0.   +0.j]
 [0.   +0.j 0.   +0.j 0.   +0.j 1.732+0.j]
 [0.   +0.j 0.   +0.j 0.   +0.j 0.   +0.j]]
```

and [`dq.set_layout()`][dynamiqs.set_layout] changes the default of all these operators for the rest of the program:

```pycon
>>> dq.set_layout('dense')
>>> dq.sigmaz()
QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
[[ 1.+0.j  0.+0.j]
 [ 0.+0.j -1.+0.j]]
>>> dq.set_layout('dia')  # back to the default layout
```

### Converting between layouts

The `.asdense()` and `.assparsedia()` methods convert an existing qarray, and the `layout` argument of [`dq.asqarray()`][dynamiqs.asqarray] converts any qarray-like:

```pycon
>>> dq.asqarray([[0, 1], [1, 0]], layout=dq.dia)
QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dia, ndiags=2
[[⋅ 1]
 [1 ⋅]]
```

To build a `dia` qarray directly from its diagonals, without going through a dense array, use [`dq.sparsedia_from_dict()`][dynamiqs.sparsedia_from_dict], which takes a dictionary mapping each offset to its diagonal (positive offsets are above the main diagonal):

```pycon
>>> dq.sparsedia_from_dict({0: [1.0, 2.0, 3.0], 1: [4.0, 5.0]})
QArray: shape=(3, 3), dims=(3,), dtype=float32, layout=dia, ndiags=2
[[1. 4. ⋅ ]
 [ ⋅ 2. 5.]
 [ ⋅  ⋅ 3.]]
```

!!! Warning "Converting to `dia` inside JAX transformations"
    Converting a dense array to the `dia` layout requires finding its non-zero diagonals, which depends on the values of the array. Inside a function transformed by `jax.jit` or `jax.grad`, these values are unknown, and you must pass the offsets of the diagonals explicitly, with the `offsets` argument of `.assparsedia()` or [`dq.asqarray()`][dynamiqs.asqarray]:

    ```pycon
    >>> f = jax.jit(lambda x: dq.asqarray(x, layout=dq.dia, offsets=(0,)))
    >>> f(jnp.eye(2))
    QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dia, ndiags=1
    [[1. ⋅ ]
     [ ⋅ 1.]]
    ```

### Combining layouts

Operations between two `dia` qarrays return a `dia` qarray. As soon as a dense qarray or array is involved, the result is generally dense:

| Operation | `dia` with `dia` | `dia` with dense |
| --- | --- | --- |
| `x + y`, `x - y` | `dia` | dense, with a warning |
| `x @ y` | `dia` | dense |
| `x & y` | `dia` | dense |
| `x.elmul(y)` | `dia` | `dia` |

Operations on a single `dia` qarray, such as `x.dag()`, `x.powm(n)` or the product with a scalar, keep the `dia` layout. A few operations are only implemented for dense matrices, and convert a `dia` qarray to dense with a warning: the matrix exponential `x.expm()` (unless `x` is diagonal), the eigendecompositions, `x.addscalar()`, and [`dq.where()`][dynamiqs.where].

A Hamiltonian built from `dia` operators thus stays sparse, but a single dense term makes the whole sum dense:

```pycon
>>> a = dq.destroy(4)
>>> (a.dag() @ a + 0.5 * (a + a.dag())).layout
dia
>>> (a.dag() @ a + dq.displace(4, 1.0)).layout
dense
```

### When to use each layout

Use the `dia` layout, the default, for the Hamiltonian and jump operators of a simulation. Solvers exploit it in their matrix products with the state, which stays dense, and the gain grows with the Hilbert space dimension. The `dense` layout is preferable for operators with many non-zero diagonals, for which the sparse format brings no saving: note that the number of diagonals grows with products and powers, for example $(a + a^\dagger)^3$ has four diagonals.

States are dense: in particular, [`dq.mesolve()`][dynamiqs.mesolve] requires a dense initial density matrix, so a `dia` qarray such as `dq.eye(n) / n` must be converted with `.asdense()` first. See the [Performance](performance.md#store-operators-in-the-dia-layout) tutorial for common patterns that silently make operators dense.

## Interoperability with QuTiP, NumPy and JAX

A QuTiP `Qobj` can be passed anywhere a qarray-like is accepted, including to solvers:

```pycon
>>> result = dq.sesolve(
...     qt.sigmax(), qt.basis(2, 0), [0.0, 1.0], exp_ops=[qt.sigmaz()], progress_meter=False
... )
>>> result.expects
Array([[ 1.   +0.j, -0.416+0.j]], dtype=complex64)
```

[`dq.asqarray()`][dynamiqs.asqarray] keeps the tensor product structure of a `Qobj` in `dims`:

```pycon
>>> dq.asqarray(qt.tensor(qt.destroy(3), qt.qeye(2))).dims
(3, 2)
```

Conversely, [`dq.to_qutip()`][dynamiqs.to_qutip] (or the `.to_qutip()` method) converts a qarray-like to a `Qobj`, and a batched qarray to a list of `Qobj`:

```pycon
>>> a, b = dq.destroy(3, 2)
>>> a.to_qutip()
Quantum object: dims=[[3, 2], [3, 2]], shape=(6, 6), type='oper', dtype=Dense, isherm=False
Qobj data =
[[0.    0.    1.    0.    0.    0.   ]
 [0.    0.    0.    1.    0.    0.   ]
 [0.    0.    0.    0.    1.414 0.   ]
 [0.    0.    0.    0.    0.    1.414]
 [0.    0.    0.    0.    0.    0.   ]
 [0.    0.    0.    0.    0.    0.   ]]
>>> len(dq.coherent(4, [0.0, 1.0, 2.0]).to_qutip())
3
```

Finally, `.to_jax()` and `.to_numpy()` return the data as a dense JAX or NumPy array, whatever the layout. Note that the result loses `dims`, and that building operators from plain arrays gives up the `dia` layout:

```pycon
>>> dq.sigmax().to_jax()
Array([[0.+0.j, 1.+0.j],
       [1.+0.j, 0.+0.j]], dtype=complex64)
>>> type(dq.sigmax().to_numpy())
<class 'numpy.ndarray'>
```
