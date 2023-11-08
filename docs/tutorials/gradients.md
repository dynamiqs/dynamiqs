# Gradients

Beyond GPU-accelerated simulations, dynamiqs allows auto-differentiation with of the numerical simulations. This tutorial explains the different mechanisms dynamiqs leverages to compute gradients.

## Gradient computation overview
### Backward gradient computation

Backward gradient computation, commonly known as backpropagation, is an essential mechanism in PyTorch, usually used for training neural networks. The process involves computing the gradient of the loss function with respect to each parameter (weight and bias) in the network. These gradients are then used to update the parameters with the aim of minimizing the loss function. Here's a detailed explanation tailored for physicists familiar with mathematical concepts:

Gradient Computation

In PyTorch, every tensor has a flag: requires_grad that allows for fine control over which parts of the network require gradient computation. If requires_grad=True, PyTorch keeps track of all operations performed on the tensor. This tracking happens in the forward pass.

When the forward pass is completed, and the loss is computed, the backward pass begins by calling .backward() on the loss tensor. The .backward() function triggers the computation of gradients.

The Chain Rule

Backpropagation relies on the chain rule of calculus to compute the derivatives of the loss with respect to the parameters. Given a composite function, the chain rule allows the derivative with respect to its inputs to be expressed in terms of derivatives of its constituent functions.

For a neural network, which is essentially a composite function of its layers and activation functions, the chain rule is applied iteratively from the output layer backward through the hidden layers, hence the name "backpropagation".

Computational Graph

PyTorch creates a computational graph on the fly during the forward pass. This graph represents all the operations that have been performed on tensors with requires_grad=True. Nodes in the graph represent tensors, while edges represent functions that produce output tensors from input tensors.

Backward Pass

During the backward pass, PyTorch traverses this graph in the reverse direction to compute gradients, starting from the loss tensor. For each tensor t with t.requires_grad=True, PyTorch accumulates t.grad, which holds the gradient of the loss with respect to t.

This traversal is depth-first, and PyTorch uses the chain rule to calculate the gradients by multiplying the gradient of the loss with respect to the output of an operation by the gradients of the output with respect to its inputs.

Gradient Accumulation

It's important to note that gradients are accumulated in the grad attributes of tensors. This means that if .backward() is called multiple times on the loss without resetting the gradients, the gradients from each pass will be summed. This behavior is often utilized in RNN training where gradients from different time steps are accumulated.

Detaching Tensors

Sometimes, it's necessary to prevent gradients from being tracked, for example, when we're evaluating the model and not training it. PyTorch allows for tensors to be detached from the computation graph using .detach() method, which creates a new tensor with the same content but with requires_grad=False.

Automatic Differentiation

The beauty of PyTorch's autograd system is that it abstracts the complexity of the derivative computation. The user doesn’t need to define the derivatives explicitly. This is particularly beneficial for physicists working with complex models where manual computation of derivatives would be cumbersome.

Limitations and Considerations

While the backward gradient computation is powerful, it can lead to memory overhead due to the storage of intermediate values in the computational graph. For large networks, this can be mitigated using techniques like gradient checkpointing.

In summary, backward gradient computation in PyTorch is a sophisticated mechanism that leverages dynamic computational graphs, automatic differentiation, and the chain rule to efficiently compute gradients, which are then used to update the parameters of neural networks in the training process. This method is integral to the optimization and successful training of deep learning models.

### Adjoint method 

## Using gradients
### Computing gradients

### Gradient descent

## The sharp bits 🔪

