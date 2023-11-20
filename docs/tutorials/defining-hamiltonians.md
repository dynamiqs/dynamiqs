# Defining Hamiltonians

In this short tutorial, we explain how to define Hamiltonians in dynamiqs. There are currently three ways: using array-like objects for constant Hamiltonians, defining a function for time-dependent Hamiltonians, and using a custom list format for piecewise constant Hamiltonians.

!!! Warning "Differences with QuTiP"
    dynamiqs manipulates PyTorch tensors, which are different from QuTiP quantum objects. See in [The sharp bits 🔪](/getting_started/sharp-bits.html) page the main differences, briefly:

    - use `A + 2 * dq.eye(n)` instead of `A + 2`
    - use `A @ B` instead of `A * B`, and `dq.mpow(A, 4)` instead of `A**4`
    - use `dq.dag(x)`, `x.mH` or `x.adjoint()` instead of `x.dag()`

## Constant Hamiltonians

A constant Hamiltonian can be defined using **array-like objects**, i.e. Python lists, NumPy arrays, QuTiP quantum objects or PyTorch tensors. In all cases, the Hamiltonian is then converted internally into a PyTorch tensor for differentiability and GPU support. It is also possible to directly use dynamiqs [utility functions](../python_api/index.md) for common Hamiltonians.

For instance, to define the Pauli Z operator $H = \sigma_z$, you can use any of the following syntaxes:

```python
# using Python lists
H = [[1, 0], [0, -1]]

# using NumPy arrays
import numpy as np
H = np.array([[1, 0], [0, -1]])

# using QuTiP quantum objects
import qutip as qt
H = qt.Qobj([[1, 0], [0, -1]])
H = qt.sigmaz()

# using PyTorch tensors
import torch
H = torch.tensor([[1, 0], [0, -1]])

# using dynamiqs
import dynamiqs as dq
H = dq.sigmaz()
```

## Time-dependent Hamiltonians

A time-dependent Hamiltonian can be defined using a Python function with signature `H(t: float) -> Tensor` that returns the Hamiltonian as a PyTorch tensor for any time `t`.

For instance, to define a time-dependent Hamiltonian $H = \sigma_z + \cos(t)\sigma_x$, you can use the following syntax:

```python
def H(t):
    return dq.sigmaz() + torch.cos(t) * dq.sigmax()
```

!!! Warning "Function returning non-tensor object"
    An error is raised if `H(t)` return a non-tensor object, or a tensor with a different `dtype` or `device` than the ones specified to the solver. This is enforced to avoid costly type, dtype or device conversions at every time step of the numerical integration.

??? Note "Function with optional arguments"
    To define a time-dependent Hamiltonian with additional arguments, you can use Python's lambda:
    ```python
    def H_args(t, omega):
        return dq.sigmaz() + torch.cos(omega * t) * dq.sigmax()
    H = lambda t: H_args(t, 1.0)
    ```

## Piecewise constant Hamiltonians

!!! Warning "Work in Progress."
