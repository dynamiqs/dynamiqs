# The sharp bits 🔪

This page highlight common pitfalls that users may encounter when learning to use Dynamiqs.

```python
import dynamiqs as dq
```

## Main differences with QuTiP

<!-- If modifications are made in this section, ensure to also update the tutorials/time-dependent-operators.md document to reflect these changes in the "Differences with QuTiP" warning admonition at the top of the file. -->

The syntax in Dynamiqs is similar to [QuTiP](http://qutip.org/), a popular Python library for quantum simulation. However, there are some important differences that you should be aware of.

### Floating-point precision

In Dynamiqs, all arrays are represented by default with **single-precision** floating-point numbers (`float32` or `complex64`), whereas the default in QuTiP or NumPy is double-precision (`float64` or `complex128`). We made this choice to match JAX's default, and for **performance** reasons, as many problems do not require double-precision. If needed, it is possible to switch to double-precision using [`dq.set_precision()`][dynamiqs.set_precision]:
```python
dq.set_precision('double')  # 'simple' by default
```

When using single-precision, there are certain limitations to be aware of:

- **Large numbers**: Numerical errors in floating-point arithmetic become more significant when using large numbers. Therefore, you should try to choose units for your simulation such that all quantities involved are not too large.
- **Tolerances**: If you require very precise simulation results (e.g. if you set lower `rtol` and `atol` than the default values), the simulation time may increase significantly, and simulations may even get stuck. In such cases, it is recommended to switch to double-precision.

!!! Warning
    Most GPUs do not have native support for double-precision, and only perform well in single-precision. However, note that some recent NVIDIA GPUs (e.g. V100, A100, H100) do provide efficient support for double-precision.

<!-- set precision back to default
```python
dq.set_precision('simple')
```
-->

### Adding a scalar to an operator

In QuTiP, adding a scalar to a `Qobj` performs an implicit multiplication of the scalar with the identity matrix. This convention differs from the one adopted by common scientific libraries such as NumPy, PyTorch or JAX. In Dynamiqs, adding a scalar to an array performs an element-wise addition. To achieve the same result as in QuTiP, you must **explicitly multiply the scalar with the identity matrix**:

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

In QuTiP, the `*` symbol is used to multiply two operators. This convention also differs from common scientific libraries. In Dynamiqs, **the `@` symbol is used for matrix multiplication**, and the `*` symbol is reserved for element-wise multiplication:

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

??? Note "Why is there no `.dag()` method in Dynamiqs?"
    To guarantee optimum performances and straightforward compatibility with the JAX ecosystem, Dynamiqs does not subclass JAX arrays. As a consequence, we can't define a custom `.dag()` method on arrays. Note that this will possibly change in the future, as we are working on an extension that will allow defining custom methods on arrays.


## Using a for loop

If you want to simulate multiple Hamiltonians or initial states, you should use batching instead of a `for` loop. We explain in detail how it works in the [Batching simulations](../basics/batching-simulations.md) tutorial, and the associated gain in performance.
