# Summary of benchmark systems

## Transmon qubit $\pi$-pulse

The Hamiltonian that describes a driven transmon is

$$
    \hat H = 4 E_C (\hat n - n_g)^2 - E_J \cos(\hat \varphi) + \varepsilon(t) \cos(\omega_d t + \phi_d) \hat n
$$

where $\omega_d$ is the drive frequency, on resonance with the transmon frequency $\omega_{eg}$, $\varepsilon(t)$ is the drive envelope, and $\phi_d$ is the drive phase. For a gaussian drive pulse, the drive amplitude is given by

$$
    \varepsilon(t) = \varepsilon_0 \left[\exp\left(-\frac{(t - T / 2)^2}{2 T_0^2}\right) - \exp\left(-\frac{T^2}{8 T_0^2}\right)\right]
$$

where $T$ is the gate time, $T_0$ is the gaussian width, and $\varepsilon_0$ is the maximum drive amplitude. We take $\varepsilon_0$ such that a $\pi$-pulse is achieved in time $T$, i.e.

$$
    \pi = \int_0^T \varepsilon(t) dt
$$
