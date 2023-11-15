# The sharp bits 🔪

In dynamiqs a syntax similar to that of QuTiP is provided, yet with distinct differences. This similarity can lead to a false sense of familiarity, resulting in inadvertent errors. Below, we highlight several common pitfalls that users may encounter when learning to use the library.

## Operating on operators
### Adding scalars to operators

In QuTiP, adding a scalar to a `QObj` performs an implicit multiplication of the scalar with the identity matrix. Conversely, in dynamiqs, adding a scalar to a tensor performs an element-wise addition. To achieve the same result as in QuTiP, you must **explicitly multiply the scalar with the identity matrix**.

```pycon
>>> I = dq.eye(2)
>>> sz = dq.sigmaz()
>>> sz - 2 * I # correct
tensor([[-1.+0.j,  0.+0.j],
        [ 0.+0.j, -3.+0.j]])
>>> sz - 2 # incorrect in dynamiqs
tensor([[-1.+0.j, -2.+0.j],
        [-2.+0.j, -3.+0.j]])
```

### Multiplying operators

In QuTiP, the multiplication of operators is executed using the `*` symbol. This convention diverges from the one adopted by common scientific libraries such as NumPy, PyTorch, and Jax. In dynamiqs, which is built on PyTorch, **the `@` symbol is used for matrix multiplication**, and the `*` symbol is reserved for element-wise multiplication.

```pycon
>>> dq.sigmax() @ dq.sigmax()  # correct
tensor([[1.+0.j, 0.+0.j],
        [0.+0.j, 1.+0.j]])
>>> dq.sigmax() * dq.sigmax()  # incorrect
tensor([[0.+0.j, 1.+0.j],
        [1.+0.j, 0.+0.j]])
```

Likewise, use `torch.linalg.matrix_power` instead of `**` for matrix power.

### Computing daggers

Use `dq.dag(x)`, `x.mH` or `x.adjoint()` instead of `x.dag()` to get the hermitian conjugate of `x`.

??? Note "Why is there no .dag() method ?"
    For optimal performance, dynamiqs does not subclass PyTorch tensors. Because of that, it is not possible to define a custom `x.dag()` method on tensors.

## NumPy interoperability

A PyTorch tensor bears a strong resemblance to a NumPy array, given that both support a similar range of methods. However, it is crucial to remember that they are not entirely interchangeable. Occasionally, you might encounter situations where you need to apply functions to your tensors that are exclusively compatible with NumPy arrays. To convert a tensor into a NumPy array, simply invoke `x.numpy()`. Should the tensor reside on the GPU, you can perform the conversion by using `x.numpy(force=True)`.

## RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

To calculate gradients using a dynamiqs solver, it is necessary to designate a gradient algorithm explicitly. Failing to do so will prompt PyTorch to raise a `RuntimeError`, specifically stating that `element 0 of tensors does not require grad and does not have a grad_fn.`

```python
import dynamiqs as dq
import torch

# define simulation parameters
param = torch.tensor(0.1, requires_grad=True)
H = dq.zero(20)
jump_ops = [param * dq.destroy(20)]
exp_ops = [dq.number(20)]
rho0 = dq.coherent(20, 2.0)
tsave = torch.linspace(0, 1, 100)

# run simulation
result = dq.mesolve(
    H, jump_ops, rho0, tsave,
    exp_ops=exp_ops,
    gradient=dq.gradient.Autograd() # required to calculate gradients !
)

# compute loss and gradient
loss = result.expects[0, -1].real
loss.backward()
```

Should the same error persist, ensure that your simulation computation includes at least one parameter set to `requires_grad=True`.

## Using `for` loops

If you want to compute the evolution of multiple initial states or with multiple Hamiltonian, you can use the [batching](/tutorials/batching.html) feature. Avoid using `for` loops to compute the evolution of multiple initial states or with multiple Hamiltonian, as it will be much slower than using batching.
