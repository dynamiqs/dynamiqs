# Computing gradients

Every Dynamiqs solver is differentiable. You can compute the derivative of any quantity obtained from a simulation (an expectation value, a fidelity, a cost function) with respect to any parameter of the simulation (a Hamiltonian coefficient, a decay rate, the amplitudes of a pulse, the initial state), using the automatic differentiation of [JAX](https://docs.jax.dev/en/latest/automatic-differentiation.html). This tutorial explains how to compute gradients, how to choose the algorithm used to compute them, and how to use them to optimize parameters.

```python
import jax
import jax.numpy as jnp
import optax
import dynamiqs as dq
```

## Differentiating a simulation

To compute a gradient, write a function that maps the parameters to a real scalar by running a simulation, then transform it with [`jax.grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html). For example, consider a qubit starting in $\ket{e}$ and driven by the Hamiltonian $H = \frac{\Omega}{2}\sigma_x$. At time $T$, its polarization is $\braket{\sigma_z}(T) = \cos(\Omega T)$, so that
$$
    \frac{\partial\braket{\sigma_z}(T)}{\partial\Omega} = -T\sin(\Omega T).
$$

```pycon
>>> T = 1.0
>>> def polarization(omega):
...     H = 0.5 * omega * dq.sigmax()
...     tsave = jnp.linspace(0.0, T, 11)
...     exp_ops = [dq.sigmaz()]
...     result = dq.sesolve(H, dq.excited(), tsave, exp_ops=exp_ops, progress_meter=False)
...     return result.expects[0, -1].real  # <σz>(T)
>>> omega = 2.0
>>> grad = jax.grad(polarization)(omega)
>>> print(f'{grad:.4f}')  # numerical gradient
-0.9093
>>> print(f'{-T * jnp.sin(omega * T):.4f}')  # analytical gradient
-0.9093
```

The function can contain any JAX code before and after the solver call, and it can depend on several parameters at once. Use the `argnums` argument of [`jax.grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html) to choose the parameters to differentiate with respect to, or pass all of them in a single array or [PyTree](https://docs.jax.dev/en/latest/pytrees.html). To evaluate both the function and its gradient in a single call, use [`jax.value_and_grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.value_and_grad.html). The same applies to every solver, see for instance [this example](../getting_started/examples.md#compute-gradients-with-respect-to-some-parameters) which differentiates the photon number obtained with [`dq.mesolve()`][dynamiqs.mesolve] with respect to the frequency, the decay rate and the initial amplitude of an oscillator.

!!! Note "Return a real scalar"
    [`jax.grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html) only differentiates functions that return a real scalar. Expectation values are stored as complex numbers, so take their real part with `.real`, and reduce batched results to a scalar (e.g. with `.sum()` or `.mean()`) before differentiating.

!!! Note "Progress meter"
    The progress meter only displays the progress of the forward pass, the backward pass that computes the gradient is not tracked. Pass `progress_meter=False` to turn it off.

## Jacobians and Hessians

To differentiate a function that returns an array, for instance the expectation value at all times in `tsave`, compute its Jacobian with [`jax.jacrev()`](https://docs.jax.dev/en/latest/_autosummary/jax.jacrev.html) (reverse mode) or [`jax.jacfwd()`](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html) (forward mode). Reverse mode computes one row of the Jacobian per output, and forward mode computes one column per input: forward mode is more efficient when there are more outputs than inputs, as is the case here.

By default, the solvers only support reverse-mode differentiation. To use forward mode, pass `gradient=dq.gradient.Forward()` to the solver (see [Choosing a gradient algorithm](#choosing-a-gradient-algorithm) below):

```pycon
>>> tsave = jnp.linspace(0.0, T, 6)
>>> def polarizations(omega):
...     H = 0.5 * omega * dq.sigmax()
...     result = dq.sesolve(
...         H,
...         dq.excited(),
...         tsave,
...         exp_ops=[dq.sigmaz()],
...         gradient=dq.gradient.Forward(),
...         progress_meter=False,
...     )
...     return result.expects[0].real  # <σz>(t) for all t in tsave
>>> jax.jacfwd(polarizations)(omega)  # numerical Jacobian
Array([ 0.   , -0.078, -0.287, -0.559, -0.8  , -0.909], dtype=float32)
>>> -tsave * jnp.sin(omega * tsave)  # analytical Jacobian
Array([-0.   , -0.078, -0.287, -0.559, -0.8  , -0.909], dtype=float32)
```

Higher-order derivatives such as the Hessian computed by [`jax.hessian()`](https://docs.jax.dev/en/latest/_autosummary/jax.hessian.html) combine forward and reverse mode, and require `gradient=dq.gradient.HigherOrder()`. For the previous example, $\partial^2\braket{\sigma_z}(T)/\partial\Omega^2 = -T^2\cos(\Omega T)$:

```pycon
>>> def polarization(omega):
...     H = 0.5 * omega * dq.sigmax()
...     result = dq.sesolve(
...         H,
...         dq.excited(),
...         jnp.linspace(0.0, T, 11),
...         exp_ops=[dq.sigmaz()],
...         gradient=dq.gradient.HigherOrder(),
...         progress_meter=False,
...     )
...     return result.expects[0, -1].real
>>> print(f'{jax.hessian(polarization)(omega):.4f}')  # numerical Hessian
0.4161
>>> print(f'{-(T**2) * jnp.cos(omega * T):.4f}')  # analytical Hessian
0.4161
```

## Choosing a gradient algorithm

All solvers take a `gradient` argument that selects the algorithm used to differentiate through the integration. There are four options for the time integration solvers, defined in `dq.gradient`:

| Gradient | Differentiation mode | When to use it |
| --- | --- | --- |
| [`BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed] | Reverse | The default for ODE methods, and the preferred choice for most problems. |
| [`Forward`][dynamiqs.gradient.Forward] | Forward | Functions with more outputs than inputs, e.g. Jacobians computed with [`jax.jacfwd()`](https://docs.jax.dev/en/latest/_autosummary/jax.jacfwd.html). |
| [`HigherOrder`][dynamiqs.gradient.HigherOrder] | Forward and reverse | Hessians and other higher-order derivatives. |
| [`Direct`][dynamiqs.gradient.Direct] | Forward and reverse | Methods that support no other option, or when you need both modes. |

- [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed] backpropagates through the solver internals, and stores the state only at a few checkpoints during the forward pass, recomputing the intermediate steps during the backward pass. This bounds the memory usage, at the cost of some extra computation. The number of checkpoints is set by its `ncheckpoints` argument: allocate as many as your memory allows to speed up the backward pass. It cannot be used for forward-mode differentiation.
- [`dq.gradient.Forward`][dynamiqs.gradient.Forward] propagates the derivatives alongside the state during the integration. Its cost grows with the number of inputs and does not depend on the number of outputs. It cannot be used for reverse-mode differentiation, so in particular not with [`jax.grad()`](https://docs.jax.dev/en/latest/_autosummary/jax.grad.html).
- [`dq.gradient.Direct`][dynamiqs.gradient.Direct] differentiates through the solver internals with standard JAX automatic differentiation, in both modes. For adaptive step size methods, reverse-mode differentiation with this option is usually slower and uses more memory than with [`dq.gradient.BackwardCheckpointed`][dynamiqs.gradient.BackwardCheckpointed].
- [`dq.gradient.HigherOrder`][dynamiqs.gradient.HigherOrder] uses the same algorithm as [`dq.gradient.Direct`][dynamiqs.gradient.Direct], but makes the requirement for higher-order differentiation explicit and is only accepted by the methods on which it is supported.

When `gradient` is not specified, a method-dependent default is used. The following table lists the supported options for each method, with the default marked in bold:

| Method | `Direct` | `BackwardCheckpointed` | `Forward` | `HigherOrder` |
| --- | :---: | :---: | :---: | :---: |
| [`Tsit5`][dynamiqs.method.Tsit5], [`Dopri5`][dynamiqs.method.Dopri5], [`Dopri8`][dynamiqs.method.Dopri8], [`Euler`][dynamiqs.method.Euler] | :material-check: | **default** | :material-check: | :material-check: |
| [`Kvaerno3`][dynamiqs.method.Kvaerno3], [`Kvaerno5`][dynamiqs.method.Kvaerno5] | :material-check: | **default** | :material-check: | |
| [`Rouchon1`][dynamiqs.method.Rouchon1], [`Rouchon2`][dynamiqs.method.Rouchon2], [`Rouchon3`][dynamiqs.method.Rouchon3] | :material-check: | **default** | :material-check: | |
| [`LowRank`][dynamiqs.method.LowRank] | :material-check: | **default** | :material-check: | |
| [`Expm`][dynamiqs.method.Expm] | **default** | | | |

[`dq.mesteadystate()`][dynamiqs.mesteadystate] uses a fifth option by default, [`dq.gradient.Implicit`][dynamiqs.gradient.Implicit], which differentiates the equation satisfied by the steady state instead of the iterations that reached it, see the [Steady states](steady-states.md#computing-gradients) tutorial.

Passing an unsupported option raises an error that lists the supported ones. The Monte Carlo methods [`JumpMonteCarlo`][dynamiqs.method.JumpMonteCarlo] and [`DiffusiveMonteCarlo`][dynamiqs.method.DiffusiveMonteCarlo] support the options of the method they use to sample trajectories. For the methods of the stochastic solvers, refer to the "Supported gradients" section of each method's documentation.

## Differentiating with respect to a pulse

Time-dependent operators are differentiable with respect to their parameters too, see the [Time-dependent operators](time-dependent-operators.md) tutorial to define them. A common case is a control pulse discretized as a piecewise constant function with [`dq.pwc()`][dynamiqs.pwc], whose amplitudes are differentiated with respect to.

Let us drive the same qubit with $H(t) = \frac{u(t)}{2}\sigma_x$, where $u(t)=u_k$ on the time interval $[t_k, t_{k+1}[$ of duration $\Delta t$. The Hamiltonian commutes with itself at all times, so the qubit rotates by the total pulse area $\theta=\sum_k u_k\Delta t$, and $\braket{\sigma_z}(T) = \cos\theta$. The gradient with respect to each amplitude is thus $\partial\braket{\sigma_z}(T)/\partial u_k = -\Delta t\sin\theta$:

```pycon
>>> times = jnp.linspace(0.0, T, 6)  # 5 time intervals of duration 0.2
>>> def polarization(amps):
...     H = dq.pwc(times, amps, 0.5 * dq.sigmax())
...     exp_ops = [dq.sigmaz()]
...     result = dq.sesolve(
...         H, dq.excited(), [0.0, T], exp_ops=exp_ops, progress_meter=False
...     )
...     return result.expects[0, -1].real
>>> amps = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0])  # pulse area θ = 1.0
>>> jax.grad(polarization)(amps)  # numerical gradient
Array([-0.168, -0.168, -0.168, -0.168, -0.168], dtype=float32)
>>> print(f'{-0.2 * jnp.sin(1.0):.3f}')  # analytical gradient
-0.168
```

The gradient has the same shape as `amps`, with one entry per time interval. This is the starting point of gradient-based optimal control: the [Driven-dissipative Kerr oscillator](../advanced_examples/kerr-oscillator.md#optimization-with-grape) example uses it to optimize a $\pi$-pulse with the GRAPE algorithm.

## Optimizing parameters

Gradients can be fed to any JAX optimizer. We use here the [Optax](https://optax.readthedocs.io/en/latest/) library to fit the parameters of a model to some data. Let us consider a qubit starting in $\ket{g}$, driven with Rabi frequency $\Omega$ and decaying with rate $\gamma$:
$$
    H = \frac{\Omega}{2}\sigma_x, \qquad L = \sqrt{\gamma}\,\sigma_-.
$$
We generate a reference signal $\braket{\sigma_z}(t)$ with $\Omega = 3$ and $\gamma = 0.5$, and then recover these values from an initial guess by minimizing the mean squared error between the simulated signal and the reference one with the Adam optimizer:

```pycon
>>> tsave = jnp.linspace(0.0, 5.0, 51)
>>> def signal(params):
...     omega, gamma = params
...     H = 0.5 * omega * dq.sigmax()
...     jump_ops = [jnp.sqrt(gamma) * dq.sigmam()]
...     exp_ops = [dq.sigmaz()]
...     result = dq.mesolve(
...         H, jump_ops, dq.ground(), tsave, exp_ops=exp_ops, progress_meter=False
...     )
...     return result.expects[0].real
>>> data = signal(jnp.array([3.0, 0.5]))  # reference signal
>>> def loss(params):
...     return ((signal(params) - data) ** 2).mean()
>>> optimizer = optax.adam(learning_rate=0.05)
>>> @jax.jit
... def step(params, opt_state):
...     grads = jax.grad(loss)(params)
...     updates, opt_state = optimizer.update(grads, opt_state)
...     return optax.apply_updates(params, updates), opt_state
>>> params = jnp.array([2.5, 0.2])  # initial guess
>>> opt_state = optimizer.init(params)
>>> for _ in range(150):
...     params, opt_state = step(params, opt_state)
>>> print(f'Omega = {params[0]:.3f}, gamma = {params[1]:.3f}')
Omega = 3.000, gamma = 0.500
```

The whole optimization step is JIT-compiled with [`jax.jit()`](https://docs.jax.dev/en/latest/_autosummary/jax.jit.html), so the simulation and its gradient are compiled once and then reused at every iteration.

!!! Tip
    To run several optimizations from different initial guesses at once, batch the simulation (see the [Batching simulations](batching-simulations.md) tutorial), or vectorize the optimization step with [`jax.vmap()`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html).

## Complex parameters

Parameters can be complex, for example the amplitude $\alpha$ of an initial coherent state or the complex amplitudes of a drive. For a real-valued function $f$ of a complex parameter $z=x+iy$, JAX returns
$$
    \texttt{jax.grad(f)(z)} = \frac{\partial f}{\partial x} - i\frac{\partial f}{\partial y},
$$
which is the **conjugate** of the direction of steepest ascent. For example, a coherent state $\ket{\alpha}$ decaying with rate $\kappa$ has $\bar n(T) = |\alpha|^2 e^{-\kappa T}$ photons at time $T$, whose gradient as returned by JAX is $2\alpha^* e^{-\kappa T}$:

```pycon
>>> n, kappa = 16, 0.5
>>> a = dq.destroy(n)
>>> def photon_number(alpha):
...     H = dq.zeros(n)
...     jump_ops = [jnp.sqrt(kappa) * a]
...     psi0 = dq.coherent(n, alpha)
...     exp_ops = [dq.number(n)]
...     result = dq.mesolve(
...         H, jump_ops, psi0, [0.0, T], exp_ops=exp_ops, progress_meter=False
...     )
...     return result.expects[0, -1].real
>>> alpha = 1.0 + 0.5j
>>> print(f'{jax.grad(photon_number)(alpha):.3f}')  # numerical gradient
1.213-0.607j
>>> print(f'{2 * jnp.conj(alpha) * jnp.exp(-kappa * T):.3f}')  # analytical gradient
1.213-0.607j
```

To minimize $f$, you should therefore take a step along the conjugate of the gradient, e.g. `z = z - learning_rate * jax.grad(f)(z).conj()`. The same holds when passing gradients to an optimizer such as Optax, as done in the [Driven-dissipative Kerr oscillator](../advanced_examples/kerr-oscillator.md#optimization-with-grape) example. See [The sharp bits](../getting_started/sharp-bits.md#computing-the-gradient-with-respect-to-complex-parameters) for more details.
