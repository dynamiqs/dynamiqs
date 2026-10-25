# Floquet integration

This tutorial introduces Floquet integration, why it is useful and how it is implemented numerically.

## Floquet's theorem

Consider a Hamiltonian that is periodic with period $T$, $H(t+T)=H(t)$. Floquet's theorem states that
solutions to the corresponding Schrödinger equation can be written as
$$
    |\Psi_{m}(t)\rangle = e^{-i\epsilon_{m}t}|\phi_{m}(t)\rangle
$$
where $m$ indexes the states, $\epsilon_m$ is the so-called quasienergy with Floquet modes $|\phi_{m}(t)\rangle$ and Floquet states $|\Psi_{m}(t)\rangle$. The Floquet modes $|\phi_{m}(t)\rangle$ are also time periodic, $|\phi_{m}(t+T)\rangle=|\phi_{m}(t)\rangle$. Floquet's theorem is in complete analogy to Bloch's theorem for a particle in a spatially periodic potential. The quasienergies are only defined modulo $\omega_d = 2\pi/T$, and Dynamiqs returns them in the interval $[-\omega_d/2, \omega_d/2[$.

A numerical recipe for computing Floquet modes and quasienergies is obtained by using the definition of the Floquet states. On the one hand, we have
$$
\begin{aligned}
    |\Psi_{m}(t+T)\rangle &= e^{-i\epsilon_{m}(t+T)}|\phi_{m}(t+T)\rangle \\\\
                          &= e^{-i\epsilon_{m}(t+T)}|\phi_{m}(t)\rangle \\\\
                          &= e^{-i\epsilon_{m}T}|\Psi_{m}(t)\rangle.
\end{aligned}
$$
On the other hand, the time-propagated Floquet state $|\Psi_{m}(t+T)\rangle$ can be obtained from the propagator $U(t, t+T)$ from time $t$ to time $t+T$,
$$
    |\Psi_{m}(t+T)\rangle=U(t,t+T)|\Psi_{m}(t)\rangle.
$$
Setting $t=0$ and using $|\Psi_{m}(0)\rangle=|\phi_{m}(0)\rangle$, we find that the Floquet modes at $t=0$ satisfy an eigenvalue equation
$$
    U(0,T)|\phi_{m}(0)\rangle=e^{-i\epsilon_{m}T}|\phi_{m}(0)\rangle,
$$
where the eigenvalues are exponentials of the quasienergies. Therefore, one means of obtaining the $t=0$ Floquet modes and associated quasienergies is to diagonalize the propagator $U(0,T)$. The Floquet modes at other times can then be immediately obtained from the Floquet modes at $t=0$ by noting that
$$
\begin{aligned}
    |\Psi_{m}(t)\rangle &= e^{-i\epsilon_{m}t}|\phi_{m}(t)\rangle \\\\
        &= U(0,t)|\Psi_{m}(0)\rangle \\\\
        &= U(0,t)|\phi_{m}(0)\rangle.
\end{aligned}
$$
From the first and the last equality we then obtain
$$
    |\phi_{m}(t)\rangle=e^{i\epsilon_{m}t}U(0,t)|\phi_{m}(0)\rangle.
$$
Oftentimes, this procedure must be done numerically if the Hamiltonian is time dependent. However, there are simple cases that can be treated analytically, one of which we explore below to demonstrate the technique and build intuition.

## Driven qubit

Let us consider a qubit of frequency $\omega$ driven at frequency $\omega_d=2\pi/T$ with amplitude $A$, within the rotating-wave approximation (RWA),
$$
    H(t) = \frac{\omega}{2}\sigma_{z}+\frac{A}{2}(\sigma_{+}e^{-i\omega_{d}t} + \mathrm{h.c.}),
$$
where $\sigma_{+}=|0\rangle\langle1|$ is [`dq.sigmap()`][dynamiqs.sigmap] (in Dynamiqs, $|0\rangle$ is the excited state [`dq.excited()`][dynamiqs.excited], with $\sigma_z|0\rangle=|0\rangle$). We can straightforwardly solve for the propagator associated with this Hamiltonian by first moving into a rotating frame (also called interaction frame) defined by the unitary transformation $U_{r}(t)=\exp(-i\omega_{d}t\sigma_{z}/2)$, with $|\Psi(t)\rangle = U_r(t)|\Psi'(t)\rangle$. Using $U_r^\dagger\sigma_+U_r = \sigma_+ e^{i\omega_d t}$, the Hamiltonian in the rotating frame is
$$
    H'=U_{r}^{\dagger} H(t)U_{r}-iU_{r}^{\dagger}\dot{U_{r}} =\frac{\Delta}{2}\sigma_{z}+\frac{A}{2}\sigma_{x},
$$
where $\Delta=\omega-\omega_{d}$. This is a time-independent Hamiltonian, for which we can obtain the propagator immediately by exponentiation. The propagator in this frame after one period of the drive is
$$
    U'(0,T) = e^{-iH'T} = \cos\left(\frac{\Omega T}{2}\right)I-i\frac{\sin(\frac{\Omega T}{2})}{\Omega}\left(\Delta\sigma_{z} + A\sigma_{x}\right),
$$
where $\Omega=\sqrt{\Delta^2+A^2}$. To obtain the Floquet modes and quasienergies, we need to diagonalize this propagator. This 2×2 matrix can be diagonalized by hand, or with symbolic algebra software, yielding the eigenvalues
$$
    \eta_{\pm} = e^{\mp i\Omega T/2},
$$
and associated (unnormalized) eigenvectors
$$
    |\phi_{\pm}(0)\rangle = \left(
    \begin{matrix}
        1 \\\\ \frac{A}{\Delta\pm\Omega}
    \end{matrix}
    \right).
$$
In the rotating frame, the quasienergies are thus $\epsilon'_{\pm} = \pm \Omega/2$. We can simplify the form of the eigenvectors by using some trigonometric identities. We first define $\Delta/\Omega = \cos\theta$ and $A/\Omega = \sin\theta$. We then multiply the eigenvectors by successive constants to simplify them. For instance, for $|\phi_{+}(0)\rangle$ we obtain
$$
    |\phi_{+}(0)\rangle = \left(
    \begin{matrix}
        1 \\\\ \frac{A}{\Delta+\Omega}
    \end{matrix}
    \right) \rightarrow
    \left(
    \begin{matrix}
        \Delta + \Omega \\\\ A
    \end{matrix}
    \right)
    \rightarrow
    \left(
    \begin{matrix}
        1 + \Delta/\Omega \\\\ A/\Omega
    \end{matrix}
    \right)
    =\left(
    \begin{matrix}
        1 + \cos\theta \\\\ \sin\theta
    \end{matrix}
    \right).
$$
Normalizing and noting that $1 + \cos\theta=2\cos^2\tfrac{\theta}{2}$ and $\sin\theta=2\sin\tfrac{\theta}{2}\cos\tfrac{\theta}{2}$, we obtain
$$
    |\phi_{+}(0)\rangle = \left(
    \begin{matrix}
        \cos\tfrac{\theta}{2} \\\\ \sin\tfrac{\theta}{2}
    \end{matrix}
    \right).
$$
By a similar analysis, using $\cos\theta - 1 = -2\sin^2\tfrac{\theta}{2}$, we find for the other eigenvector
$$
    |\phi_{-}(0)\rangle = \left(
    \begin{matrix}
        -\sin\tfrac{\theta}{2} \\\\ \cos\tfrac{\theta}{2}
    \end{matrix}
    \right).
$$

Back in the lab frame, the propagator is $U(0,T)=U_r(T)U'(0,T)U_r^\dagger(0)$. Since $U_r(0)=I$ and $U_r(T)=e^{-i\pi\sigma_z}=-I$, we have $U(0,T)=-U'(0,T)$: the Floquet modes at $t=0$ are the same, and the eigenvalues are multiplied by $-1=e^{-i\omega_d T/2}$. The lab-frame quasienergies are thus shifted to $\epsilon_\pm = \pm\Omega/2 + \omega_d/2$ (modulo $\omega_d$).

This example, written directly in the rotating frame, is used as an analytical test for the Floquet functionality of Dynamiqs.

## Finding Floquet modes and quasienergies using Dynamiqs

Let us now consider a truly time-dependent example for which there is no analytical solution: the Rabi model of a driven two-level system, without the RWA. The Floquet modes and quasienergies are computed with [`dq.floquet()`][dynamiqs.floquet]:

```pycon
>>> import jax.numpy as jnp
>>> import dynamiqs as dq
>>> jnp.set_printoptions(precision=3, suppress=True)  # set custom array print style
>>> T = 1.0  # drive period
>>> omega_d = 2 * jnp.pi / T  # drive frequency
>>> H0 = 0.1 * dq.sigmaz()  # drift Hamiltonian
>>> f = lambda t: jnp.cos(omega_d * t)  # time-dependent drive
>>> H1 = dq.modulated(f, dq.sigmax())  # drive Hamiltonian
>>> tsave = jnp.linspace(0, T, 11)  # saving times
>>> res = dq.floquet(H0 + H1, T, tsave, progress_meter=False)  # run the simulation
>>> res.quasienergies
Array([ 0.097, -0.097], dtype=float32)
>>> res.modes.shape
(11, 2, 2, 1)
```

The quasienergies `res.quasienergies` have shape _(n,)_, and the Floquet modes `res.modes` have shape _(ntsave, n, n, 1)_: for each time in `tsave`, the $n$ Floquet modes, each a ket of shape _(n, 1)_. The mode `res.modes[k, m]` at time `tsave[k]` is associated to the quasienergy `res.quasienergies[m]`. Because the Floquet modes are periodic, they only need to be computed over a single period: `tsave` must be sorted in strictly ascending order and satisfy `tsave[-1] - tsave[0] <= T`.

The Hamiltonian can be batched as for the other solvers, see the [Batching simulations](batching-simulations.md) tutorial. The period `T` must however be common to all elements of the batch. To compute the Floquet modes for different drive periods, wrap the call to [`dq.floquet()`][dynamiqs.floquet] in a [`jax.vmap()`](https://docs.jax.dev/en/latest/_autosummary/jax.vmap.html):

```python
import jax


def single_floquet(omega_d):
    H = 0.1 * dq.sigmaz() + dq.modulated(lambda t: jnp.cos(omega_d * t), dq.sigmax())
    T = 2.0 * jnp.pi / omega_d
    tsave = jnp.linspace(0.0, T, 11)
    return dq.floquet(H, T, tsave, progress_meter=False)


omegas_d = jnp.array([0.9, 1.0, 1.1]) * 2.0 * jnp.pi
result = jax.vmap(single_floquet)(omegas_d)  # result.quasienergies has shape (3, 2)
```
