# Experimental data fits

Dynamiqs enables rapid simulations and facilitates automatic gradient computation, making it highly suitable for data fitting tasks. This tutorial will guide you through the process of simulating and fitting real-world experimental data, step by step.

## Context of the experiment

!!! note Info
    This part introduces the kind of data and process that we will fit in the notebook. If you are only interested in the code you can skip it.

!!! example "Diving deep 🤿"
    You can discover further details about this experiment in the original research paper from which this data was sourced [^1]

The data we will work with are deflation curves. Our system is composed of a high lifetime cavity - the memory -  coupled to a lossy one - the buffer - in vacuum state. We denote $a$ (respectively $b$) the memory (respectively buffer) annihilation operator. The memory and the buffer are coupled through a two to one photon exchange mechanism $g_2 a^2 b^\dagger + h.c.$. The buffer is a highly-dissipative mode, that we model with the dissipator $D[\sqrt{\kappa_b} b]$. Both cavities have a finite temperature, expressed in thermal population.

At the beginning of the experiment the memory is populated with a coherent state and the buffer is in thermal vacuum. We enable for a period of two photon exchange mechanism. During this period, pairs of photons leave the memory and are converted into a single buffer photon, that is then dissipated into the environment. The memory photon number $a^\dagger a$ is measured at different times and saved into the `data` variable. Our goal is to infer $g_2$ from the data.

![Pulse sequence](/figs-docs/g2-pulse-sequence.png)
_Pulse sequence of the experiment, taken from [^1]_

[^1]:  [Quantum control of a cat-qubit with bit-flip times exceeding ten seconds, arxiv:2307.06617](https://arxiv.org/pdf/2307.06617.pdf)

## Step by step solution

We start by importing all necessary libraries:

```python
import dynamiqs as dq
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True) # crash instead of creating `nan` values 

device = "cuda" if torch.cuda.is_available() else "cpu"
```

Declare useful helpers to handle units:
```python
MHz = 2 * np.pi
kHz = 2 * np.pi * 1e-3
us = 1.0
ns = 1.0e-3
```

Define physical values that describe our system (taken from [^1]):
```python
thermal_a, thermal_b = 0.1, 0.011
kappa_a = 9.3e-3 * kHz
kappa_b = 2.6 * MHz
```

Instanciate our bosonic annihilation operators:
```python
Na, Nb = 15, 7
a, b = dq.destroy(Na, Nb, device=device)
```

### Loading the data

We provide real experimental data but also leave the possibility to simply simulate them, you can choose the one that suits you best

#### Option 1: Real world data

Before running the code, you can download the data at this link (TODO: add link)

Load the data
```python
def load(file):
    x = np.loadtxt(file)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return x
    
alpha2 = load("alpha2.npy")
data = load("experimental-fits-data.npy")
time = load("experimental-fits-time.npy")
```

and plot them
```python
plt.figure(figsize=(10, 5))

plt.title("Short timescale")
plt.plot(time.numpy(force=True), data.numpy(force=True), "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
render('real_data')
```

#### Option 2: Synthetic data

Create the synthetic data 
```python
torch.manual_seed(42) # set the random seed for reproducibility

g2_true = 1.2 * MHz
b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))

alpha2 = np.linspace(1, 3, 3)
rho0 = torch.stack([dq.tensprod(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

H = g2_true * a @ a @ dq.dag(b)
H += dq.dag(H)

La_down = np.sqrt(kappa_a * (1 + thermal_a)) * a
La_up   = np.sqrt(kappa_a * thermal_a) * dq.dag(a)
Lb_down = np.sqrt(kappa_b * (1 + thermal_b)) * b
Lb_up   = np.sqrt(kappa_b * thermal_b) * dq.dag(b)

time = torch.linspace(0, 0.6, 501, device=device)
result = dq.mesolve(
    H, [Lb_down, Lb_up, La_down, La_up],
    rho0, time, 
    exp_ops=[dq.dag(a) @ a],
    options=dict(device=device)
)

data = result.expects[:, 0].real.T
data += (torch.rand_like(data) - 0.5) * 0.15 # add noise to the data
```
and plot them
```python
plt.plot(time.numpy(force=True), data.numpy(force=True), "+")
plt.grid()
render('synthetic_data')
```

### Perform the fit

We define the simulation code
```python
def simulate(g2, verbose=False):
    b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))
    rho0 = torch.stack([dq.tensprod(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

    H = g2 * a @ a @ dq.dag(b)
    H += dq.dag(H)

    La_down = np.sqrt(kappa_a * (1 + thermal_a)) * a
    La_up   = np.sqrt(kappa_a * thermal_a) * dq.dag(a)
    Lb_down = np.sqrt(kappa_b * (1 + thermal_b)) * b
    Lb_up   = np.sqrt(kappa_b * thermal_b) * dq.dag(b)
    
    result = dq.mesolve(
        H, [Lb_down, Lb_up, La_down, La_up],
        rho0, time, 
        exp_ops=[dq.dag(a) @ a],
        gradient=dq.gradient.Adjoint(params=params), 
        options=dict(device=device, save_states=True, verbose=verbose)
    )

    return result
```

We choose an initial guess that is reasonable with respect to the data we have. We converge to a steady state in roughly $1.0\mu$s so $g_2$ should be of the order of $1$MHz. Let's take $0.1$MHz to have some margin.
```python
g2 = torch.tensor(0.1 * MHz, requires_grad=True, device=device)
result = simulate(g2, verbose=True)

plt.figure(figsize=(10, 5))

time_np = time.numpy(force=True)
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].numpy(force=True).real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].numpy(force=True).real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
render('initial_guess')
```

% skip: start

The optimization loop is defined as follow
```python
losses = []
criterion = torch.nn.MSELoss() # Mean square error loss

def closure():
    lbfgs.zero_grad()
    result = simulate(g2)
    loss = criterion(result.expects[:, 0].real.T, data)
    loss.backward()
    losses.append(loss.item())
    return loss
    
lbfgs = torch.optim.LBFGS([g2], max_iter=4, line_search_fn="strong_wolfe")
for i in tqdm(range(5)):
    lbfgs.step(closure)
```
we use the `L-BFGS` Pytorch solver as its convergence is faster than a simple gradient descent.

!!! warning Optimisation time
    The full optimisation runs in 5 to 10 minutes on a ``

To retreive the values, we show the fit plot
```python 
result = simulate(g2, verbose=True)

plt.figure(figsize=(10, 5))

time_np = time.numpy(force=True) # convert to numpy to use in matplotlib
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].numpy(force=True).real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].numpy(force=True).real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
```
[Fit result](/figs-docs/g2-fit-result.png)

and we can plot the loss to ensure we converged correclty
```python
plt.plot(torch.tensor(losses))
plt.yscale("log")
plt.grid()
```

[Loss plot](/figs-docs/g2-loss-plot.png)

We finally find $g_2$:
```pycon
>>> g2 / MHz
tensor(0.8646, device='cuda:0', grad_fn=<DivBackward0>)
```

## Full code


```python
import dynamiqs as dq
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
torch.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Useful values

MHz = 2 * np.pi
kHz = 2 * np.pi * 1e-3
us = 1.0
ns = 1.0e-3

# Declare bosonic operators

Na, Nb = 15, 7
a, b = dq.destroy(Na, Nb, device=device)

# Known system values

thermal_a, thermal_b = 0.1, 0.011
kappa_a = 9.3e-3 * kHz
kappa_b = 2.6 * MHz

# Load and plot the data

def load(file):
    x = np.loadtxt(file)
    x = torch.tensor(x, dtype=torch.float32, device=device)
    return x
    
data = load("experimental-fits-data.npy")
time = load("experimental-fits-time.npy")
alpha2 = data[0]

plt.figure(figsize=(10, 5))

plt.plot(time.numpy(force=True), data.numpy(force=True), "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")

# Define simulation code

def simulate(g2, verbose=False):
    b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))
    rho0 = torch.stack([dq.tensprod(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

    H = g2 * a @ a @ dq.dag(b)
    H += dq.dag(H)

    La_down = np.sqrt(kappa_a * (1 + thermal_a)) * a
    La_up   = np.sqrt(kappa_a * thermal_a) * dq.dag(a)
    Lb_down = np.sqrt(kappa_b * (1 + thermal_b)) * b
    Lb_up   = np.sqrt(kappa_b * thermal_b) * dq.dag(b)
    
    result = dq.mesolve(
        H, [Lb_down, Lb_up, La_down, La_up],
        rho0, time, 
        exp_ops=[dq.dag(a) @ a],
        gradient=dq.gradient.Adjoint(params=[g2]), 
        options=dict(device=device, save_states=True, verbose=verbose)
    )

    return result

# Plot initial guess

g2 = torch.tensor(0.1 * MHz, requires_grad=True, device=device)
result = simulate(g2, verbose=True)

plt.figure(figsize=(10, 5))

time_np = time.numpy(force=True)
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].numpy(force=True).real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].numpy(force=True).real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
plt.show()

# Fit the data

losses = []

def closure():
    lbfgs.zero_grad()
    
    result = simulate(g2)
    criterion = torch.nn.MSELoss()
    
    loss = criterion(result.expects[:, 0].real.T, data)
    loss.backward()
    losses.append(loss.item())
    
    return loss
    
lbfgs = torch.optim.LBFGS([g2], max_iter=4, line_search_fn="strong_wolfe")
for i in tqdm(range(5)):
    lbfgs.step(closure)

# Plot the fit

result = simulate(g2, verbose=True)

plt.figure(figsize=(10, 5))

time_np = time.numpy(force=True)
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].numpy(force=True).real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].numpy(force=True).real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
plt.show()

# Plot the loss

plt.show(losses)
plt.yscale("log")
plt.grid()
plt.show()

# Print the final value

print(g2 / MHz)
```

% skip: end
