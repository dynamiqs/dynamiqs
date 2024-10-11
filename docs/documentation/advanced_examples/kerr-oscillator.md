# Driven-dissipative Kerr oscillator

In this example, we show how to simulate a **driven-dissipative Kerr oscillator** in Dynamiqs. It is a simple example of a non-linear quantum harmonic oscillator with dissipative coupling to its environment. In the appropriate rotating frame, it is described by the master equation
$$
    \frac{\dd\rho}{\dt} = -i [H(t), \rho] + \kappa \mathcal{D}[a] (\rho),
$$
with Hamiltonian
$$
    H(t) = -K a^{\dagger 2} a^2 + \epsilon(t) a^\dagger + \epsilon^*(t) a,
$$
with $\kappa$ the rate of single-photon dissipation, $K$ the Kerr non-linearity, and $\epsilon(t)$ the driving field.

```python
import dynamiqs as dq
import jax.numpy as jnp
import matplotlib.pyplot as plt
```

## Simulating time evolution

Let us begin with a simple simulation of the **time evolution of this system**, assuming that the driving field is constant and the system is initially in a coherent state.

```python
# define parameters
K = 1.0        # Kerr non-linearity
epsilon = 3.0  # driving field
kappa = 1.5    # dissipation rate
alpha = 2.0    # coherent state amplitude
sim_time = 15.0 # simulation time
ntsave = 100   # number of saved states
N = 32         # truncation of Fock space

# save times
tsave = jnp.linspace(0.0, sim_time, ntsave)

# operators
a, adag = dq.destroy(N), dq.create(N)
H = -K * adag @ adag @ a @ a + epsilon * (a + adag)
jump_ops = [jnp.sqrt(kappa) * a]

# initial state
psi0 = dq.coherent(N, alpha)

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave)
```

From this simulation, we can extract any property of the evolved state at any saved time. We can for instance access the number of photons in the final state using

```pycon
>>> dq.expect(adag @ a, result.states[-1])
Array(1.435+0.j, dtype=complex64)
```

Alternatively, we can also plot the Wigner function of the evolved state.

```python
gif = dq.plot.wigner_gif(result.states, ymax=3.0)
rendergif(gif, 'wigner-kerr-oscillator')
```

![plot_wigner_gif_kerr](/figs_docs/wigner-kerr-oscillator.gif){.fig}

## Periodic revival of a coherent state

### Time evolution of the cavity field

One of the most striking features of the Kerr oscillator is the periodic revival of the initial coherent state. This phenomenon is a direct consequence of the non-linear interaction between the photons in the cavity. We can observe this effect by plotting the absolute value of the **cavity field as a function of time**, for a simulation time over several units of Kerr, and as long as photon loss is not too important.

```python
# redefine parameters
K = 1.0
kappa = 0.02
alpha = 2.0
ntsave = 200
N = 32

# save times
sim_time = 5 * jnp.pi / K
tsave = jnp.linspace(0.0, sim_time, ntsave)

# operators
a, adag = dq.destroy(N), dq.create(N)
H = -K * adag @ adag @ a @ a
jump_ops = [jnp.sqrt(kappa) * a]

# initial state
psi0 = dq.coherent(N, alpha)

# expectation operator
exp_ops = [a]

# run simulation
result = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops)

# plot cavity field
plt.plot(tsave * K / jnp.pi, jnp.abs(result.expects[0]))
plt.xlabel(r'Time, $tK / \pi$')
plt.ylabel(r'$|\langle a(t) \rangle|$')
renderfig('photon-number-kerr-oscillator')
```

![plot_photon_number_kerr](/figs_docs/photon-number-kerr-oscillator.png){.fig}

We indeed observe a periodic revival of the coherent state, with a period of $\pi / K$. These revivals have a reduced amplitude due to the presence of photon loss.

### Study of the revival amplitudes

We can further investigate these periodic revivals by plotting the amplitude of the first revival as a function of the photon loss $\kappa$, and as a function of the initial coherent state amplitude. To do so, we make use of a powerful feature of Dynamiqs: the ability to **batch simulations concurrently**. Here, we can batch over both a jump operator and the initial state.

```python
# parameters to sweep
kappas = jnp.linspace(0.0, 0.1, 11)
nbars = jnp.linspace(0.4, 4.0, 10)
alphas = jnp.sqrt(nbars)

# save times
sim_time = jnp.pi / K # a single revival
tsave = jnp.linspace(0.0, sim_time, 100)

# redefine jump operators and initial states
jump_ops = [jnp.sqrt(kappas[:, None, None]) * a] # using numpy broadcasting
psi0 = dq.coherent(N, alphas) # dq.coherent accepts a batched input

# run batched simulation
result = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops)
amp_revivals = jnp.abs(result.expects[:, :, 0, -1] / result.expects[:, :, 0, 0])

# plot a 2D map of the normalized amplitude revivals
fig, ax = plt.subplots()
contour = ax.pcolormesh(nbars, kappas / K, amp_revivals)
cbar = plt.colorbar(contour, label=r'$|\langle a(T) \rangle / \langle a(0) \rangle |$')
ax.set_xlabel(r'Initial coherent state amplitude, $\bar{n} = |\alpha_0|^2$')
ax.set_ylabel(r'Loss rate, $\kappa / K$')
renderfig('amplitude-revivals-kerr-oscillator')
```

![plot_amplitude_revivals_kerr](/figs_docs/amplitude-revivals-kerr-oscillator.png){.fig}

We observe that the amplitude of the first revival decreases monotically with the photon loss rate $\kappa$, and with the initial coherent state amplitude $\bar{n}$. This behavior is consistent with the expected behavior of the Kerr oscillator. Remarkably, thanks to batching, such a set of hundreds of simulations can be run in a few seconds.
