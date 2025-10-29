# Floquet Integration

This tutorial introduces Floquet integration, why it is useful and how it is implemented numerically.

## Floquet's Theorem

Consider a Hamiltonian that is periodic with period $T$: $H(t+T)=H(t)$. Floquet's theorem teaches us that the
solutions to the Schrödinger equation can be written as
$$
    |\Psi_{m}(t)\rangle = e^{-i\epsilon_{m}t}|\phi_{m}(t)\rangle
$$
where $m$ indexes the states, $\epsilon_m$ is the so-called quasienergy, the $|\phi_{m}(t)\rangle$ are
called Floquet modes and $|\Psi_{m}(t)\rangle$ are called Floquet states. The Floquet modes $|\phi_{m}(t)\rangle$
are time periodic: $|\phi_{m}(t+T)\rangle=|\phi_{m}(t)\rangle$. Floquet's theorem is in
complete analogy to Bloch's theorem in 3 dimensions for a particle in a periodic potential. 

A numerical recipe for computing the Floquet modes and quasienergies can be obtained by using the definition
of the Floquet states. On the one hand, we have
$$
    \begin{aligned}
    |\Psi_{m}(t+T)\rangle &= e^{-i\epsilon_{m}(t+T)}|\phi_{m}(t+T)\rangle
                          &= e^{-i\epsilon_{m}(t+T)}|\phi_{m}(t)\rangle.
    \end{aligned}
$$
On the other hand, the time-propagated Floquet state $|\Psi_{m}(t+T)\rangle$ can be obtained from the 
propagator
$$
    |\Psi_{m}(t+T)\rangle=U(0,T)|\Psi_{m}(t)\rangle.
$$
Thus we find that the Floquet modes at $t=0$ satisfy an eigenvalue equation
$$
    U(0,T)|\phi_{m}(0)\rangle=e^{-i\epsilon_{m}T}|\phi_{m}(0)\rangle,
$$
where the eigenvalues are exponentials of the quasienergies. Therefore, one means of obtaining the $t=0$ Floquet modes and associated
quasienergies is to diagonalize the propagator. The Floquet modes at other times can then be immediately obtained from the Floquet modes at $t=0$ by noting that
$$
    |\Psi_{m}(t)\rangle = e^{-i\epsilon_{m}t}|\phi_{m}(t)\rangle=U(0,t)|\Psi_{m}(0)\rangle=U(0,t)|\phi_{m}(0)\rangle.
$$
From the second and the last equality we then obtain
$$
    |\phi_{m}(t)\rangle=e^{i\epsilon_{m}t}U(0,t)|\phi_{m}(0)\rangle.
$$
Oftentimes, this procedure must be done numerically if the Hamiltonian is time dependent. However, there are simple cases that can be treated analytically, one of which we explore below to demonstrate the technique and build intuition.

## Driven qubit

Let us consider a driven qubit within the rotating-wave approximation (RWA)
$$
    H(t) = -\frac{\omega}{2}\sigma_{z}+\frac{A}{2}(\sigma_{+}e^{-i\omega_{d}t} + {\rm H.~c.~}),
$$
where $\sigma_{+}=|1\rangle\langle0|$. We can straightforwardly solve for the propagator associated with this
Hamiltonian by first moving into a rotating frame (also called the interaction frame) defined by the unitary 
transformation $U_{r}(t)=\exp(i\omega_{d}t\sigma_{z}/2)$. The Hamiltonian in the rotating frame is
$$
    H'(t)=U_{r}^{\dagger} H(t)U_{r}-iU_{r}^{\dagger}\dot{U_{r}} =-\frac{\delta\omega}{2}\sigma_{z}+\frac{A}{2}\sigma_{x},
$$
where $\delta\omega=\omega-\omega_{d}$. This is a time-independent Hamiltonian, for which we can obtain the propagator immediately by exponentiation. The
propagator in this frame after one period of the drive is
$$
    U'(0,T) = \cos\left(\frac{\Omega t}{2}\right)I-i\frac{\sin(\frac{\Omega t}{2})}{\Omega}\left(-\delta\omega\sigma_{z} + A\sigma_{x}\right),
$$
where $\Omega=\sqrt{\delta\omega^2+A^2}.$ To obtain the Floquet modes and quasienergies, we need to diagonalize this propagator. This 2x2 matrix can be diagonalized by hand, or with symbolic algebra software, yielding the eigenvalues
$$
\eta_{\pm} = e^{-i(\pm\Omega T/2)},
$$
and associated (unnormalized) eigenvectors
$$
|\epsilon_{\pm}\rangle = \left( 
\begin{matrix}
1 \\ \frac{\epsilon}{\delta\omega\pm\Omega}
\end{matrix}
\right).
$$
The quasienergies are thus $\epsilon_{\pm} = \pm \Omega/2$.
We can simplify the form of the eigenvectors by using some trig identities: we first define $\delta\omega/\Omega = \cos(\theta), \epsilon/\Omega = \sin(\theta)$. We then multiply the eigenvectors by successive constants to simplify them: for instance, for $|\epsilon_{+}\rangle$ we obtain
$$
|\epsilon_{+}\rangle = \left( 
\begin{matrix}
1 \\ \frac{\epsilon}{\delta\omega+\Omega}
\end{matrix}
\right) \rightarrow 
\left( 
\begin{matrix}
\delta\omega + \Omega \\ \epsilon
\end{matrix}
\right)
\rightarrow
\left( 
\begin{matrix}
\delta\omega/\Omega + 1 \\ \epsilon/\Omega
\end{matrix}
\right)
=\left( 
\begin{matrix}
\cos\theta + 1 \\ \sin\theta
\end{matrix}
\right).
$$
Normalizing and noting that $\cos\theta+1=2\cos^2\tfrac{\theta}{2}$ and $\sin\theta=2\sin\tfrac{\theta}{2}\cos\tfrac{\theta}{2}$, we obtain
$$
|\epsilon_{+}\rangle = \left( 
\begin{matrix}
\cos\tfrac{\theta}{2} \\ \sin\tfrac{\theta}{2}
\end{matrix}
\right).
$$
By a similar analysis, we find for the other eigenvector
$$
|\epsilon_{-}\rangle = \left( 
\begin{matrix}
\sin\tfrac{\theta}{2} \\ -\cos\tfrac{\theta}{2}
\end{matrix}
\right).
$$
This example is used as an analytical test for the Floquet functionality of Dynamiqs.

## Finding Floquet modes and quasienergies using Dynamiqs

Let us now consider a truly time-dependent example for which there is no analytical solution: the Rabi model of a driven two-level system.

```python
import jax.numpy as jnp
import dynamiqs as dq

jnp.set_printoptions(precision=3, suppress=True)  # set custom array print style

T = 1.0                             # drive period
omega_d = 2 * jnp.pi / T            # drive frequency
H0 = 0.1 * dq.sigmaz()              # drift Hamiltonian
f = lambda t: jnp.cos(omega_d * t)  # time-dependent drive
H1 = dq.modulated(f, dq.sigmax())   # drive Hamiltonian
tsave = jnp.linspace(0, T, 11)      # saving times
res = dq.floquet(H0 + H1, T, tsave)  # run the simulation
print(res.quasienergies)            # print the quasienergies
print(res.modes[-1])                # print the final floquet modes
```

```text title="Output"
|██████████| 100.0% ◆ elapsed 1.99ms ◆ remaining 0.00ms
[ 0.972 -0.972]
QArray: shape=(2, 2, 1), dims=(2,), dtype=complex64, layout=dense
[[[ 0.998+0.j]
  [-0.055-0.j]]

 [[ 0.055-0.j]
  [ 0.998+0.j]]]
```