# The sharp bits 🔪

This page highlight common pitfalls that users may encounter when learning to use Dynamiqs.

```python
import dynamiqs as dq
```

## Main differences with QuTiP

The syntax in Dynamiqs is similar to [QuTiP](http://qutip.org/), a popular Python library for quantum simulation. However, there are some important differences that you should be aware of.

### Floating-point precision

In Dynamiqs, all objects are represented by default with **single-precision** floating-point numbers (`float32` or `complex64`), whereas the default in QuTiP or NumPy is double-precision (`float64` or `complex128`). We made this choice to match JAX's default, and for **performance** reasons, as many problems do not require double-precision. If needed, it is possible to switch to double-precision using [`dq.set_precision()`][dynamiqs.set_precision]:

```python
dq.set_precision('double')  # 'single' by default
```

When using single-precision, there are certain limitations to be aware of:

- **Large numbers**: Numerical errors in floating-point arithmetic become more significant when using large numbers. Therefore, you should try to choose units for your simulation such that all quantities involved are not too large.
- **Tolerances**: If you require very precise simulation results (e.g. if you set lower `rtol` and `atol` than the default values), the simulation time may increase significantly, and simulations may even get stuck. In such cases, it is recommended to switch to double-precision.

!!! Warning
    Most GPUs do not have native support for double-precision, and only perform well in single-precision. However, note that some recent NVIDIA GPUs (e.g. V100, A100, H100) do provide efficient support for double-precision.

<!-- set precision back to default
```python
dq.set_precision('single')
```
-->

### Adding a scalar to an operator

In QuTiP, adding a scalar to a `Qobj` performs an implicit multiplication of the scalar with the identity matrix. This convention differs from the one adopted by common scientific libraries such as NumPy, PyTorch or JAX. In Dynamiqs, adding a scalar to an operator with `+` is forbidden. To achieve the same result as in QuTiP, you must **explicitly multiply the scalar with the identity matrix**:

=== ":material-check: Correct"
    ```pycon
    >>> sz = dq.sigmaz()
    >>> sz - 2 * dq.eye(2)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[-1.+0.j    ⋅   ]
     [   ⋅    -3.+0.j]]
    ```
=== ":material-close: Incorrect"
    ```pycon
    >>> sz = dq.sigmaz()
    >>> sz - 2
    Traceback (most recent call last):
        ...
    NotImplementedError: Adding a scalar to a qarray with the `+` operator is not supported. To add a scaled identity matrix, use `x + scalar * dq.eye_like(x)`. To add a scalar, use `x.addscalar(scalar)`.
    ```

If you *actually* want to add a scalar element-wise to an operator, you can use `x.addscalar(scalar)`.

### Multiplying two operators

In QuTiP, the `*` symbol is used to multiply two operators. This convention also differs from common scientific libraries. In Dynamiqs, **the `@` symbol is used for matrix multiplication**, and the `*` symbol is reserved for element-wise multiplication with a scalar:

=== ":material-check: Correct"
    ```pycon
    >>> sx = dq.sigmax()
    >>> sx @ sx
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[1.+0.j   ⋅   ]
     [  ⋅    1.+0.j]]
    ```
=== ":material-close: Incorrect"
    ```pycon
    >>> sx = dq.sigmax()
    >>> sx * sx
    Traceback (most recent call last):
        ...
    NotImplementedError: Element-wise multiplication of two qarrays with the `*` operator is not supported. For matrix multiplication, use `x @ y`. For element-wise multiplication, use `x.elmul(y)`.
    ```

If you *actually* want to multiply two operators element-wise,you can use `x.elmul(y)`.

Likewise, you should use `x.powm()` instead of `**` (element-wise power) to compute the power of a matrix:

=== ":material-check: Correct"
    ```pycon
    >>> sx.powm(2)
    QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
    [[1.+0.j   ⋅   ]
     [  ⋅    1.+0.j]]
    ```
=== ":material-close: Incorrect"
    ```pycon
    >>> sx**2
    Traceback (most recent call last):
        ...
    NotImplementedError: Computing the element-wise power of a qarray with the `**` operator is not supported. For the matrix power, use `x.pomw(power)`. For the element-wise power, use `x.elpow(power)`.
    ```

If you *actually* want to compute the element-wise power, you can use `x.elpow(power)`.

## Using a for loop

If you want to simulate multiple Hamiltonians or initial states, you should use batching instead of a `for` loop. This functionality is explained in detail in the [Batching simulations](../basics/batching-simulations.md) tutorial, together with the associated gain in performance.

## Computing the gradient with respect to complex parameters

To optimize a real-valued function of complex parameters $f:\mathbb{C}^p\to\mathbb{R}$, you should take a step in the direction of the **conjugate** of the gradient given by JAX. For example, if you use the SciPy optimizers with `scipy.minimize()`, you need to conjuguate the gradient before providing it to the optimiser:

<!-- skip: start -->
=== ":material-check: Correct"
    ```python
    grad = lambda x: jax.grad(f)(x).conj()
    scipy.minimize(f, ..., jac=grad)
    ```
=== ":material-close: Incorrect"
    ```python
    grad = lambda x: jax.grad(f)(x)
    scipy.minimize(f, ..., jac=grad)
    ```
<!-- skip: end -->

See the [JAX documentation](https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation) and the [PyTorch documentation](https://pytorch.org/docs/stable/notes/autograd.html#autograd-for-complex-numbers) for detailed discussions on complex numbers and differentiation.
