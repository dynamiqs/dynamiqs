# Experimental data fits

Dynamiqs enables rapid simulations and facilitates automatic gradient computation, making it highly suitable for data fitting tasks. This tutorial will guide you through the process of simulating and fitting real-world experimental data, step by step.

## Context of the experiment

!!! note Info
    This part introduces the kind of data and process that we will fit in the notebook. If you are only interested in the code you can skip it.


The data we will work with are deflation curves. Our system is composed of a high lifetime cavity - the memory -  coupled to a lossy one - the buffer - in vacuum state. We denote $a$ (respectively $b$) the memory (respectively buffer) annihilation operator. The memory and the buffer are coupled through a two to one photon exchange mechanism $g_2 a^2 b^\dagger + h.c.$. The buffer is a highly-dissipative mode, that we model with the dissipator $D[\sqrt{\kappa_b} b]$. Both cavities have a finite temperature, expressed in thermal population.

At the beginning of the experiment the memory is populated with a coherent state and the buffer is in thermal vacuum. We enable for a period of two photon exchange mechanism. During this period, pairs of photons leave the memory and are converted into a single buffer photon, that is then dissipated into the environment. The memory photon number $a^\dagger a$ is measured at different times and saved into the `data` variable. Our goal is to infer $g_2$ from the data.

![Pulse sequence](/figs-docs/g2-pulse-sequence.png)
_Pulse sequence of the experiment, taken from [^1]_

[^1]:  [Quantum control of a cat-qubit with bit-flip times exceeding ten seconds, arxiv:2307.06617](https://arxiv.org/pdf/2307.06617.pdf)

## Step by step solution

We start by importing all necessary libraries:

```python
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
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
a, b = dq.destroy(Na, Nb))
```

### Loading the data

We provide real experimental data but also leave the possibility to simply simulate them, you can choose the one that suits you best

#### Option 1: Real world data

Before running the code, you can download the data at this link (TODO: add link)

Load the data

```python
def load(file):
    x = np.loadtxt(file)
    x = jnp.array(x)
    return x

data = load("short.npy")
time = load("time_short.npy")
alpha2 = data[0]
```

and plot them

```python
plt.figure(figsize=(10, 5))

plt.plot(time, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
renderfig('g2-real-data')
```

% invisible-code-block: python
% exp_time, exp_data, exp_alpha2 = time, data, alpha2

![Loaded data](/figs-docs/g2-real-data.png)

#### Option 2: Synthetic data

Create the synthetic data 

```python
g2_true = 1.2 * MHz
b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))

alpha2 = np.linspace(1, 3, 3)
rho0 = jnp.stack([dq.tensor(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

H = g2_true * a @ a @ dq.dag(b)
H += dq.dag(H)

La_down = np.sqrt(kappa_a * (1 + thermal_a)) * a
La_up   = np.sqrt(kappa_a * thermal_a) * dq.dag(a)
Lb_down = np.sqrt(kappa_b * (1 + thermal_b)) * b
Lb_up   = np.sqrt(kappa_b * thermal_b) * dq.dag(b)

time = jnp.linspace(0, 0.6, 501)
result = dq.mesolve(
    H, [Lb_down, Lb_up, La_down, La_up],
    rho0, time,
    exp_ops=[dq.dag(a) @ a],
    solver=dq.solver.Dopri5()
)

data = result.expects[:, 0].real.T
key = jax.random.key(42)
data += (jax.random.uniform(key, data.shape) - 0.5) * 0.15
```
and plot them
```python
plt.figure(figsize=(10, 5))

plt.plot(time, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
renderfig('g2-synthetic-data')
```
![Synthetic data](/figs-docs/g2-synthetic-data.png)


### Perform the fit

% invisible-code-block: python
% time, data, alpha2 = exp_time, exp_data, exp_alpha2

We define the simulation code

```python
def simulate(g2):
    b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))
    rho0 = jnp.stack([dq.tensor(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

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
    )

    return result
```

We choose an initial guess that is reasonable with respect to the data we have. We converge to a steady state in roughly $1.0\mu$s so $g_2$ should be of the order of $1$MHz. Let's take $0.1$MHz to have some margin.
```python
g2 = jnp.array(0.1 * MHz)
result = simulate(g2)

plt.figure(figsize=(10, 5))

time_np = time
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
renderfig('g2-initial-guess')
```
![g2 initial guess](/figs-docs/g2-initial-guess.png)


% skip: start

The optimization loop is defined as follow
```python
def cost(g2, data):
    res = simulate(g2)
    return jnp.sum((data - res.expects[:, 0].real.T) ** 2)

g2 = jnp.array(0.1 * MHz)
costs = []
g2s = [g2]
for _ in tqdm(range(40)):
    val, grad = jax.value_and_grad(cost)(g2, data)
    grad = jnp.clip(grad, -100, 100)
    g2 = g2 - 0.01 * grad
    costs.append(val)
    g2s.append(g2)
costs.append(cost(g2, data))

costs = np.array(costs)
g2s = np.array(g2s)
```
we use a plain gradient descent algorithm for simplicity but one could imagine using any other algorithm, such as the ones provided in [Optax](https://github.com/google-deepmind/optax).

!!! note Optimisation time
    The full optimisation runs in less than one minute on a NVIDIA GeForce RTX 4090 and around 5 minutes on a M1 Mac.


We can plot the loss to ensure we converged correctly
```python
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(costs)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("loss")
plt.yscale("log")

plt.subplot(122)
plt.plot(g2s / MHz)
plt.grid()
plt.xlabel("iteration")
plt.xlabel("g2 [MHz]")
```
![Fit result](/figs-docs/g2-fit-result.png)

Finally we plot the fit and compare with the data
```python
result = simulate(g2)
plt.figure(figsize=(10, 5))

time_np = time
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")

print(g2 / MHz)
```

!!! example "Final note"
    $g_2$ value differs from the one reported in
    [arxiv:2307.06617](https://arxiv.org/pdf/2307.06617.pdf)
    by approximately 15%. This discrepancy is attributed to the unavailability 
    of Dynamiqs for performing fits at the time of writing the article. 
    Consequently, the authors were unable to employ gradient descent for their
    fit and had to resort to alternative methods.


## Full code

```python
import dynamiqs as dq
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# === useful helpers ===

MHz = 2 * np.pi
kHz = 2 * np.pi * 1e-3
us = 1.0
ns = 1.0e-3

# === Define operators ===

Na, Nb = 15, 7
a, b = dq.destroy(Na, Nb)

# === Various physical parameters ===

thermal_a, thermal_b = 0.1, 0.011
kappa_a = 9.3e-3 * kHz
kappa_b = 2.6 * MHz

# === OPTION 1 === 
# === Generate synthetic data ===

g2_true = 1.2 * MHz
b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))

alpha2 = np.linspace(1, 3, 3)
rho0 = jnp.stack([dq.tensor(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

H = g2_true * a @ a @ dq.dag(b)
H += dq.dag(H)

La_down = np.sqrt(kappa_a * (1 + thermal_a)) * a
La_up   = np.sqrt(kappa_a * thermal_a) * dq.dag(a)
Lb_down = np.sqrt(kappa_b * (1 + thermal_b)) * b
Lb_up   = np.sqrt(kappa_b * thermal_b) * dq.dag(b)

time = jnp.linspace(0, 0.6, 501)
result = dq.mesolve(
    H, [Lb_down, Lb_up, La_down, La_up],
    rho0, time, 
    exp_ops=[dq.dag(a) @ a],
    solver=dq.solver.Dopri5()
)

data = result.expects[:, 0].real.T
key = jax.random.key(42)
data += (jax.random.uniform(key, data.shape) - 0.5) * 0.15

# === Show simulated data ===

plt.figure(figsize=(10, 5))

plt.plot(time, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")


# === OPTION 2 === 
# === Load real experimental data === 

def load(file):
    x = np.loadtxt(file)
    x = jnp.array(x)
    return x
    
data = load("short.npy")
time = load("time_short.npy")
alpha2 = data[0]

# === Show the data ===

plt.figure(figsize=(10, 5))

plt.plot(time, data, "+")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")

# === Define the simulation code ===

def simulate(g2):
    b0 = dq.unit(dq.fock(Nb, 0) + thermal_b ** 0.5 * dq.fock(Nb, 1))
    rho0 = jnp.stack([dq.tensor(dq.coherent(Na, alpha), b0) for alpha in alpha2 ** 0.5])

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
    )

    return result

# === Plot initial guess and data ===

g2 = jnp.array(0.1 * MHz)
result = simulate(g2)

plt.figure(figsize=(10, 5))

time_np = time
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")

# === Implement gradient descent ===

def cost(g2, data):
    res = simulate(g2)
    return jnp.sum((data - res.expects[:, 0].real.T) ** 2)

g2 = jnp.array(0.1 * MHz)
costs = []
g2s = [g2]
for _ in tqdm(range(40)):
    val, grad = jax.value_and_grad(cost)(g2, data)
    grad = jnp.clip(grad, -100, 100)
    g2 = g2 - 0.01 * grad
    costs.append(val)
    g2s.append(g2)
costs.append(cost(g2, data))

costs = np.array(costs)
g2s = np.array(g2s)

# === Show gradient descent progress ===

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.plot(costs)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("loss")
plt.yscale("log")

plt.subplot(122)
plt.plot(g2s / MHz)
plt.grid()
plt.xlabel("iteration")
plt.xlabel("g2 [MHz]")

# === Visualize the fit ===

result = simulate(g2)
plt.figure(figsize=(10, 5))

time_np = time
for i in range(len(alpha2)):
    plt.plot(time_np, data[:, i].real, f"+C{i}")
    plt.plot(time_np, result.expects[i, 0].real, f"C{i}")
plt.grid()
plt.xlabel("Time [us]")
plt.ylabel("Photon number")
print(g2 / MHz)
```

% skip: end
