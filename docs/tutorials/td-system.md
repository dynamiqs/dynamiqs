# Defining Hamiltonians

In this short tutorial, we will show how to define Hamiltonians in dynamiqs. There are currently three main ways: using array-like objects for constant Hamiltonians, using `callable` objects for time-dependent Hamiltonians, and using a custom list format for piecewise constant Hamiltonians.

***

## Constant Hamiltonians

A constant Hamiltonian can be defined using **array-like objects**, i.e. Python lists, NumPy arrays, QuTiP quantum objects or PyTorch tensors. In all cases, the Hamiltonian is then converted internally into a PyTorch tensor for differentiability and GPU support. It is also possible to use the [utility functions](../python_api/utils.md) provided by dynamiqs to define Hamiltonians.

For instance, to define the Pauli Z operator $H = \sigma_z$, one can use the following syntax:
```python
>>> # using Python lists
>>> H = [[1, 0], [0, -1]]

>>> # using NumPy arrays
>>> import numpy as np
>>> H = np.array([[1, 0], [0, -1]])

>>> # using QuTiP quantum objects
>>> import qutip as qt
>>> H = qt.Qobj([[1, 0], [0, -1]])

>>> # using PyTorch tensors
>>> import torch
>>> H = torch.tensor([[1, 0], [0, -1]])

>>> # using dynamiqs
>>> H = dq.sigmaz()
```

!!! Warning "No Tensor subclassing in dynamiqs"
    Contrary to QuTiP in which a `qutip.QObj` is internally a SciPy or NumPy array, dynamiqs **does not** subclass PyTorch tensors. This means that you cannot use non-PyTorch methods directly on Tensor objects. For example, you should use `H.mH` or `H.adjoint()` (PyTorch methods) instead of `H.dag()`.

    Also, to compute the **product of quantum operators**, one should use the matrix multiplication operator `@` instead of the `*` operator as in QuTiP. For instance:
    ```python
    >>> print(dq.sigmax() @ dq.sigmax() == dq.eye(2))
    True
    >>> print(dq.sigmax() * dq.sigmax() == dq.eye(2))
    False
    ```

## Time-dependent Hamiltonians

A time-dependent Hamiltonian can be defined using a **callable**, i.e. a Python function or a Python class with a `__call__` method. The function should have signature `H(t: float) -> Tensor` where `t` is the time, and should return a Hamiltonian as a PyTorch tensor.

For instance, to define a time-dependent Hamiltonian $H = \sigma_z + \cos(t)\sigma_x$, one can use the following syntax:
```python
>>> def H(t):
>>>     return dq.sigmaz() + torch.cos(t) * dq.sigmax()
```

!!! Warning "Returning non-Tensor arrays"
    An error will be raised if `H(t)` return a non-Tensor array, or a Tensor with a different `dtype` or `device` than specified to the solver. This is enforced to avoid costly type, dtype or device conversions at every time step of the numerical integration.

??? Note "Optional arguments"
    To define a time-dependent Hamiltonian with additional arguments, one can use the following syntax:
    ```python
    >>> def H_args(t, omega):
    >>>    return dq.sigmaz() + torch.cos(omega * t) * dq.sigmax()
    >>> H = lambda t: H_args(t, 1.0)
    ```

## Piecewise constant

!!! Warning "Work in Progress."
