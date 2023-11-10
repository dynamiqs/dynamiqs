# GPU simulations

This tutorial demonstrates how to leverage GPUs to accelerate the simulation of quantum systems with dynamiqs. By directing our simulations to run on the GPU, we can significantly enhance the computational efficiency, especially for large-dimensional quantum systems.

```python
import torch
import dynamiqs as dq
```

## The options dictionnary

### Selecting a device

Running simulations on a GPU is achieved in a single-line through the `options` argument of dynamiqs solvers.

% skip: start

```python
# option 1: use a string
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# option 2: use a torch.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# pass the device to the options dictionnary and to your solver
options = dict(device=device)
result = dq.sesolve(..., options=options)
```

% skip: end

Conversion of your tensors to the GPU is then handled automatically by dynamiqs. Note that the `device` argument is also available for the `mesolve` and `smesolve` solvers.

!!! Warning "Time-dependent Hamiltonians"
    For time-dependent Hamiltonians defined using a callable function (see [Defining Hamiltonians](/tutorials/defining-hamiltonians.html#time-dependent-hamiltonians)), the returned Tensor should already be on the same device (and dtype) as the one passed in `options`. If this is not the case, dynamiqs will raise an error. This avoids unnecessary data transfers between the CPU and the GPU.

### Selecting a data type

Two complex data types are available in dynamiqs: `torch.complex64` and `torch.complex128`, corresponding to single and double precision respectively. By default, dynamiqs uses `torch.complex64` for all tensors. However, you can change this by passing the `dtype` argument to the options dictionnary.

% skip: start

```python
dtype = torch.complex128
options = dict(device=device, dtype=dtype)
result = dq.sesolve(..., options=options)
```

% skip: end

!!! Note
    Many modern GPUs are only optimized for single precision operations, i.e. `torch.complex64`. In this case, using `torch.complex128` will result in a significant performance penalty.

    Unless you specifically require double precision for your problem, we recommend to stick to the default `torch.complex64` data type. In doubt, you can check the TFLOPs of your GPU on [techpowerup.com](https://www.techpowerup.com/gpu-specs/).

## Conversion between data types

On PyTorch 2.1 and above, it is possible to convert between complex and real data types using:

% skip: start

```pycon
>>> dtype = torch.complex128
>>> dtype.to_real()
torch.float64
```

```pycon
>>> dtype = torch.float32
>>> dtype.to_complex()
torch.complex64
```

% skip: end

## Speeding up matrix multiplications on NVIDIA GPUs

On certain NVIDIA GPUs (e.g. V100, A100 or some RTX series), PyTorch can leverage Tensor Cores to accelerate matrix multiplications. This is achieved by writing the following line at the top of your programs:

```python
torch.set_float32_matmul_precision("high")
```

More details on [PyTorch documentation](https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html).
