# The sharp bits 🔪

This page highlight common pitfalls that users may encounter when learning to use dynamiqs.

```python
import dynamiqs as dq
```

## Main differences with QuTiP

<!-- If modifications are made in this section, ensure to also update the tutorials/time-dependent-operators.md document to reflect these changes in the "Differences with QuTiP" warning admonition at the top of the file. -->

The syntax in dynamiqs is similar to [QuTiP](http://qutip.org/), a popular Python library for quantum simulation. However, there are some important differences that you should be aware of.

### Floating-point precision

In dynamiqs, all arrays are represented with **single-precision** floating-point numbers (complex64 or float32) by default, whereas QuTiP or NumPy arrays are represented with double-precision floating-point numbers (complex128 or float64). This choice is made for **performance**, as most problems do not require the higher precision provided by double-precision floating-point numbers. However, if needed, it is possible to switch back to double precision to accommodate more demanding computational tasks, using:
```python
dq.set_precision('double') # 'simple' by default
```

With single-precision, there are certain limitations to be aware of:

 - **Large numbers**: Single-precision floating-point numbers have a smaller range than double-precision floating-point numbers. This means that you may encounter overflow errors when working with very large numbers. Thus, it is advised to **set the scale of your problems such that the numbers involved are close to unity**.
 - **Tolerences**: If you require very precise ODE simulations (e.g. setting lower `rtol` and `atol` than defaults), you may encounter a divergence of the simulation due to the limited precision of single-precision floating-point numbers. In this case, you should switch to double-precision.

!!! Note
    Most GPUs do not have native support for double-precision floating-point numbers, such that single-precision is often much more efficient. Note that this is not the case of certain recent NVIDIA GPUs (e.g. A100, H100, H200) which do have efficient support for double-precision.


### Adding a scalar to an operator

In QuTiP, adding a scalar to a `QObj` performs an implicit multiplication of the scalar with the identity matrix. This convention differs from the one adopted by common scientific libraries such as NumPy, PyTorch or JAX. In dynamiqs, adding a scalar to an array performs an element-wise addition. To achieve the same result as in QuTiP, you must **explicitly multiply the scalar with the identity matrix**:

=== ":material-check: Correct"
    ```pycon
    >>> sz = dq.sigmaz()
    >>> sz - 2 * dq.eye(2)
    Array([[-1.+0.j,  0.+0.j],
           [ 0.+0.j, -3.+0.j]], dtype=complex64)
    ```
=== ":material-close: Incorrect"
    ```pycon
    >>> sz = dq.sigmaz()
    >>> sz - 2
    Array([[-1.+0.j, -2.+0.j],
           [-2.+0.j, -3.+0.j]], dtype=complex64)
    ```

### Multiplying two operators

In QuTiP, the `*` symbol is used to multiply two operators. This convention also differs from common scientific libraries. In dynamiqs, **the `@` symbol is used for matrix multiplication**, and the `*` symbol is reserved for element-wise multiplication:

=== ":material-check: Correct"
    ```pycon
    >>> sx = dq.sigmax()
    >>> sx @ sx
    Array([[1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]], dtype=complex64)
    ```
=== ":material-close: Incorrect"
    ```pycon
    >>> sx = dq.sigmax()
    >>> sx * sx
    Array([[0.+0.j, 1.+0.j],
           [1.+0.j, 0.+0.j]], dtype=complex64)
    ```

Likewise, you should use `dq.powm()` instead of `**` (element-wise power) to compute the power of a matrix:

=== ":material-check: Correct"
    ```pycon
    >>> dq.powm(sx, 2)
    Array([[1.+0.j, 0.+0.j],
           [0.+0.j, 1.+0.j]], dtype=complex64)
    ```
=== ":material-close: Incorrect"
    ```pycon
    >>> sx**2
    Array([[0.+0.j, 1.+0.j],
           [1.+0.j, 0.+0.j]], dtype=complex64)
    ```

### Computing matrix adjoint

Use `dq.dag(x)` or `x.mT.conj()` instead of `x.dag()` to get the hermitian conjugate of `x`.

??? Note "Why is there no `.dag()` method in dynamiqs?"
    To guarantee optimum performances and straightforward compatibility with the JAX ecosystem, dynamiqs does not subclass JAX arrays. As a consequence, we can't define a custom `.dag()` method on arrays. Note that this will possibly change in the future, as we are working on an extension that will allow defining custom methods on arrays.


## Using a for loop

If you want to simulate multiple Hamiltonians or initial states, you should use batching instead of a `for` loop. We explain in detail how it works in the [Batching simulations](../basics/batching-simulations.md) tutorial, and the associated gain in performance.
