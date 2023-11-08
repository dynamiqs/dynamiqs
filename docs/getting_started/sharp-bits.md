# The sharp bits 🔪

Dynamiqs provides a syntax similar to that of Qutip, yet with distinct differences. This similarity can lead to a false sense of familiarity, resulting in inadvertent errors. Below, we highlight several common pitfalls that users may encounter when initially working with the library.

## Scalar additions

In qutip, adding a tensor with a numerical value equates to integrating the number scaled by the identity matrix into the tensor. Conversely, in Dynamiqs, appending a numerical value to a tensor results in the addition of the number to each element of the tensor on an element-wise basis.

```pycon
>>> import torch
>>> I = torch.eye(2)
>>> a = torch.tensor([[2., 0.], [0., 2.]])
>>> a - 2 * I # good 👍
tensor([[0., 0.],
        [0., 0.]])
>>> a - 2 # way to write it in Qutip, but does not work in Dynamiqs.
tensor([[0., -2.],
        [-2., 0.]])
```

## Matrix multiplication

In qutip, the multiplication of operators is executed using the `*` symbol. This convention diverges from the one commonly adopted by numerous scientific libraries such as NumPy, PyTorch, and Jax. Dynamiqs, which is built on PyTorch, employs the `@` symbol to carry out matrix multiplication, whereas the `*` symbol is reserved for element-wise multiplication.

```pycon
>>> import torch
>>> I = torch.eye(2)
>>> a = torch.tensor([[1., 2.], [3., 4.]])
>>> I @ a # good 👍
tensor([[1., 2.],
        [3., 4.]])
>>> I * a # right in qutip, not in dynamiqs 🙅
tensor([[1., 0.],
        [0., 4.]])
```

Likewise, use `torch.linalg.matrix_power` instead of `**` for matrix power.

## Computing the adjoint

Use `x.mH` or `dq.dag(x)` instead of `x.dag()` to get the adjoint.

## If your simulation is very slow

You may have forgotten to move the tensors on the GPU. An easy way to ensure to be on gpu is to move your operators to the GPU or to use the `device` option of your solver.

```python
import dynamiqs as dq
import torch
a, b = dq.destroy(20, 20, device="cuda") # method 1
rho = dq.fock((20, 20), (0, 0))
time = torch.linspace(0, 1, 100)
dq.sesolve(a @ b, rho, time, device="cuda") # method 2
```

## `RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn`

To calculate gradients using a Dynamiqs solver, it is necessary to designate a gradient algorithm explicitly. Failing to do so will prompt PyTorch to raise a RuntimeError, specifically stating that "element 0 of tensors does not require grad and does not have a grad_fn."

```python
import dynamiqs as dq
import torch
a = dq.destroy(20)
param = torch.tensor(0.01, requires_grad=True)
rho = dq.coherent(20, 3)
time = torch.linspace(0, 1, 100)
result = dq.mesolve(
    0 * a, [param * a], 
    rho, 
    time, 
    exp_ops=[dq.dag(a) @ a],
    gradient=dq.gradient.Autograd() # 🚨 don't forget this
)
loss = result.expects[0].real.sum()
loss.backward()
```

Should the same error persist, ensure that your simulation computation includes at least one parameter set to `requires_grad=True`.

## Using `for` loops

Don't. If you want to compute the evolution of multiple initial states or with multiple Hamiltonian, use the batching feature. You'll find a tutorial for it in [the batching tutorial](/tutorials/batching.html)