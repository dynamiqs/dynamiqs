# Gradients

Beyond GPU-accelerated simulations, dynamiqs allows auto-differentiation with of the numerical simulations. This tutorial explains the different mechanisms dynamiqs leverages to compute gradients.

## Gradient computation overview
### Backward gradient computation

Backward gradient computation, commonly known as backpropagation, is an essential mechanism in PyTorch, usually used for training neural networks. The process involves computing the gradient of a loss function with respect to some physical parameters. These gradients are then used to update the parameters with the aim of minimizing the loss function.

In PyTorch, every tensor has a flag: requires_grad that allows for fine control over which parts of the network require gradient computation. If `requires_grad=True`, PyTorch keeps track of all operations performed on the tensor. This tracking happens in the "forward pass".

When the forward pass is completed, and the loss is computed, the "backward pass" begins by calling `.backward()` on the loss tensor. The `.backward()` function triggers the computation of gradients.

Backpropagation relies on the chain rule of calculus to compute the derivatives of the loss with respect to the parameters. Given a composite function, the chain rule allows the derivative with respect to its inputs to be expressed in terms of derivatives of its constituent functions.

For a neural network, which is essentially a composite function of its layers and activation functions, the chain rule is applied iteratively from the output layer backward through the hidden layers, hence the name "backpropagation". 

PyTorch creates a computational graph on the fly during the forward pass. This graph represents all the operations that have been performed on tensors with `requires_grad=True`. Nodes in the graph represent tensors, while edges represent functions that produce output tensors from input tensors.

During the backward pass, PyTorch traverses this graph in the reverse direction to compute gradients, starting from the loss tensor. For each tensor `t` with `t.requires_grad=True`, PyTorch accumulates `t.grad`, which holds the gradient of the loss with respect to t`.


The beauty of PyTorch's autograd system is that it abstracts the complexity of the derivative computation. The user doesn’t need to define the derivatives explicitly. This is particularly beneficial for simulations, where manual computation of derivatives would be cumbersome.

While the backward gradient computation is powerful, it can lead to memory overhead due to the storage of intermediate values in the computational graph. For large networks, this can be mitigated using the adjoint method.

In summary, backward gradient computation in PyTorch is a sophisticated mechanism that leverages dynamic computational graphs, automatic differentiation, and the chain rule to efficiently compute gradients, which are then used to update the parameters of neural networks in the training process. This method is integral to the optimization and successful training of deep learning models.

For more information, see [https://pytorch.org/blog/overview-of-pytorch-autograd-engine/](https://pytorch.org/blog/overview-of-pytorch-autograd-engine/)

### Adjoint method 

!!! Warning "Work in Progress."

## Using gradients
### Computing gradients

Let's see a very simple case of how to compute gradients in PyTorch. In this example we compute the gradient of the square root of a gradient. 
Let $x = [1, 2, 3]$. $||x|| = \sum_i x_i^2 = 1^2 + 2^2 + 3^2 = 14.0$. Let's compute it in torch:

```python
>>> import torch
>>> x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
>>> sum = x @ x
>>> sum
tensor(14., grad_fn=<DotBackward0>)
```

Now we can compute analytically the gradient with respect to each coordinate: $\frac{\partial ||x||}{\partial x_i} = 2 * x_i $.
Thus, we have $\frac{\partial ||x||}{\partial x} = [2, 4, 6]$. We can now compute it with torch:
```pycon
>>> sum.backward()
>>> x.grad
tensor([2., 4., 6.])
```

Et voilà! You now know how to compute gradients in PyTorch 

!!! Warning "TODO"
    Point to tutorials that use gradient computation


### Gradient descent

PyTorch provides a suite of gradient descent algorithms, including Stochastic Gradient Descent (SGD), Adam, and LBFGS. Both SGD and Adam are highly effective for machine learning applications, capable of scaling to handle an extensive number of parameters. On the other hand, LBFGS is better suited for optimizations involving a smaller scale of parameters, albeit with a somewhat more complex usage. For detailed instructions on how to use LBFGS, please consult the following tutorials [insert tutorial link].


```python 
import torch

# Initialize x and y as tensors with requires_grad=True to compute gradients
x = torch.randn(1, requires_grad=True)
y = torch.randn(1, requires_grad=True)

# Define the SGD optimizer with the parameters to optimize ([x, y]) and a learning rate
optimizer = torch.optim.Adam([x, y])

for _ in range(10_000):
    # Define the function to be minimized
    function = x**2 + 2 * (y - 2)**2

    # Zero the gradients before the backward pass
    optimizer.zero_grad()

    # Perform the backward pass to compute gradients
    function.backward()

    # Update the parameters
    optimizer.step()
```

Observe the following streamlined example which demonstrates the optimization of the function $f(x,y) = x^2 + (y - 2)^2$ using PyTorch's optimization tools. 

```pycon
>>> print(f'Optimized x: {x.item()}')
'Optimized x: -2.382207389352189e-44'
>>> print(f'Optimized y: {y.item()}')
'Optimized y: 1.999999761581421'
```
