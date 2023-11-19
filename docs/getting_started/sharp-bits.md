# The sharp bits 🔪

This page highlight common pitfalls that users may encounter when learning to use dynamiqs.

In dynamiqs we use PyTorch tensors to represent quantum states and operators. A PyTorch tensor is very similar to a NumPy array, and most NumPy functions have a PyTorch equivalent. However, PyTorch tensors can be stored on GPU, and can be attached to a *computation graph* to compute gradients. This makes them very powerful, but also introduces some subtleties that you should be aware of.

```python
import dynamiqs as dq
import torch
```

## Main differences with QuTiP

<!-- If modifications are made in this section, ensure to also update the tutorials/defining-hamiltonians.md document to reflect these changes in the "Differences with QuTiP" warning admonition at the top of the file. -->

The syntax in dynamiqs is similar to [QuTiP](http://qutip.org/), a popular Python library for quantum simulation. However, there are some important differences that you should be aware of.

### Adding a scalar to an operator

In QuTiP, adding a scalar to a `QObj` performs an implicit multiplication of the scalar with the identity matrix. This convention differs from the one adopted by common scientific libraries such as NumPy and PyTorch. In dynamiqs, adding a scalar to a tensor performs an element-wise addition. To achieve the same result as in QuTiP, you must **explicitly multiply the scalar with the identity matrix**:

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
>>> dq.mpow(sx, 4)  # correct
tensor([[1.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j]])
>>> sx**4  # incorrect
tensor([[0.+0.j, 1.+0.j],
        [1.+0.j, 0.+0.j]])
```

### Computing the adjoint

Use `dq.dag(x)`, `x.mH` or `x.adjoint()` instead of `x.dag()` to get the hermitian conjugate of `x`.

??? Note "Why is there no `.dag()` method in dynamiqs?"
    To guarantee optimum performances and straightforward compatibility with the PyTorch ecosystem, dynamiqs does not subclass PyTorch tensors. As a consequence, we can't define a custom `.dag()` method on tensors.

## Use a NumPy function

A PyTorch tensor bears a strong resemblance to a NumPy array, and they support similar methods, but they are not interchangeable. If you need to apply a NumPy function to a tensor, you must first convert it to a NumPy array using `x.numpy()`.

!!! Warning
    There are two situations when you should not convert a tensor to a NumPy array:

    - **If you run the simulation on a GPU**: the conversion will move the tensor to the CPU, which might heavily slow down your computation.
    - **If you need to compute gradients**: the conversion will detach the tensor from the computation graph, which will prevent you from computing gradients.

    Remember that you should solely use PyTorch functions if you use a GPU or compute gradients.

    If in any case you're sure that you want to perform the conversion, use `x.numpy(force=True)` to detach the tensor from the computation graph, move it to the CPU and convert it to a NumPy array.

## RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

This error is raised when you try to compute gradients with respect to a tensor that is not attached to the computation graph. There are a few common situations when you can encounter this error:

- You forgot to specify the gradient algorithm, it must be explicitely specified in [`dq.sesolve()`](../python_api/solvers/sesolve.md), [`dq.mesolve()`](../python_api/solvers/mesolve.md) or [`dq.smesolve()`](../python_api/solvers/smesolve.md) using the `gradient` argument, for example with `gradient=dq.gradient.Autograd()` to use PyTorch autograd library.
- You forgot to set `requires_grad=True` on the parameters with respect to which you want to compute the gradient.
- You converted a tensor to a NumPy array at some point in the computation (see the previous section [Use a NumPy function](#use-a-numpy-function)).

See the [Computing gradients](/tutorials/computing-gradients.html) tutorial for more details on how to compute gradients with dynamiqs.

## Using a for loop

If you want to simulate multiple Hamiltonians or initial states, you should use batching instead of a `for` loop. We explain in detail how it works in the [Batching simulations](/tutorials/batching-simulations.html) tutorial, and the associated gain in performance.
