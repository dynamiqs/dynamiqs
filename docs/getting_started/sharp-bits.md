# The sharp bits 🔪

This page highlight common pitfalls that users may encounter when learning to use dynamiqs.

```python
import dynamiqs as dq
```

## Main differences with QuTiP

<!-- If modifications are made in this section, ensure to also update the tutorials/defining-hamiltonians.md document to reflect these changes in the "Differences with QuTiP" warning admonition at the top of the file. -->

The syntax in dynamiqs is similar to [QuTiP](http://qutip.org/), a popular Python library for quantum simulation. However, there are some important differences that you should be aware of.

### Adding a scalar to an operator

In QuTiP, adding a scalar to a `QObj` performs an implicit multiplication of the scalar with the identity matrix. This convention differs from the one adopted by common scientific libraries such as NumPy, PyTorch, SciPy or JAX. In dynamiqs, adding a scalar to a tensor performs an element-wise addition. To achieve the same result as in QuTiP, you must **explicitly multiply the scalar with the identity matrix**:

```pycon
>>> I = dq.eye(2)
>>> sz = dq.sigmaz()
>>> sz - 2 * I  # correct
tensor([[-1.+0.j,  0.+0.j],
        [ 0.+0.j, -3.+0.j]])
>>> sz - 2  # incorrect
tensor([[-1.+0.j, -2.+0.j],
        [-2.+0.j, -3.+0.j]])
```

### Multiplying two operators

In QuTiP, the `*` symbol is used to multiply two operators. This convention also differs from common scientific libraries. In dynamiqs, **the `@` symbol is used for matrix multiplication**, and the `*` symbol is reserved for element-wise multiplication:

```pycon
>>> sx = dq.sigmax()
>>> sx @ sx  # correct
tensor([[1.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j]])
>>> sx * sx  # incorrect
tensor([[0.+0.j, 1.+0.j],
        [1.+0.j, 0.+0.j]])
```

Likewise, you should use `dq.mpow()` instead of `**` (element-wise power) to compute the power of a matrix:

```pycon
>>> dq.mpow(sx, 2)  # correct
tensor([[1.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j]])
>>> sx**2  # incorrect
tensor([[0.+0.j, 1.+0.j],
        [1.+0.j, 0.+0.j]])
```

### Computing the adjoint

Use `dq.dag(x)` or `x.T.conj()` instead of `x.dag()` to get the hermitian conjugate of `x`.

??? Note "Why is there no `.dag()` method in dynamiqs?"
    To guarantee optimum performances and straightforward compatibility with the JAX ecosystem, dynamiqs does not subclass JAX arrays. As a consequence, we can't define a custom `.dag()` method on tensors. Note that this will possibly change in the future, as we are working on an extension that will allow defining custom methods on arrays.


## Using a for loop

If you want to simulate multiple Hamiltonians or initial states, you should use batching instead of a `for` loop. We explain in detail how it works in the [Batching simulations](/tutorials/batching-simulations.html) tutorial, and the associated gain in performance.
