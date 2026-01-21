import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

__all__ = ['clicktimes_sse_to_sme', 'measurements_sse_to_sme']


def clicktimes_sse_to_sme(
    clicktimes: ArrayLike,
    tsave: ArrayLike,
    thetas: ArrayLike,
    etas: ArrayLike,
    key: PRNGKeyArray,
) -> Array:
    r"""Post-process click times from a jump SSE simulation to obtain click times for a
    jump SME simulation.

    The SME click times are obtained from the SSE click times by adding false clicks
    with rate $\theta$ and removing actual clicks with probability $1-\eta$.

    Args:
        clicktimes: See result of [`dq.jssesolve()`][dynamiqs.jssesolve].
        tsave: See [`dq.jssesolve()`][dynamiqs.jssesolve].
        thetas: See [`dq.jsmesolve()`][dynamiqs.jsmesolve].
        etas: See [`dq.jsmesolve()`][dynamiqs.jsmesolve].
        key: PRNG key used to sample the added noise for the post-processing.

    Returns:
        (array of shape (...)): SME click times. The shape is the same as
            `clicktimes`, except that the dimension corresponding to the number of
            jump operators measured may be smaller, if the corresponding efficiency is
            null.
    """
    clicktimes = jnp.asarray(clicktimes)
    tsave = jnp.asarray(tsave)
    thetas = jnp.asarray(thetas)
    etas = jnp.asarray(etas)

    # select loss operators with non-zero efficiency
    Ik_sse = clicktimes[..., etas != 0, :]
    thetas = thetas[etas != 0]
    etas = etas[etas != 0]

    # for broadcasting later
    thetas = thetas[:, None]
    etas = etas[:, None]

    nmaxclick = clicktimes.shape[-1]
    shape = Ik_sse.shape[:-1]

    k1, k2, k3 = jax.random.split(key, 3)

    # === keep true clicks with probability eta
    keep_mask = jax.random.bernoulli(k1, p=etas, shape=Ik_sse.shape)
    clicktimes_sme = jnp.where(keep_mask, Ik_sse, jnp.nan)

    # === sample false clicks times with rate theta
    t0, t1 = tsave[0], tsave[-1]

    # the number of false clicks is Poisson distributed with mean theta * (t1 - t0)
    nfalseclicks = jax.random.poisson(k2, thetas * (t1 - t0), shape=shape)
    # nfalseclicks: (..., ntrajs, nLms)

    # the false clicks times are uniformly distributed in [t0, t1)
    # for JIT-compatibility, we sample a predefined number of times, and then use
    # masking
    times = jax.random.uniform(k3, (*shape, nmaxclick), minval=t0, maxval=t1)
    mask = jnp.arange(nmaxclick) < nfalseclicks[..., None]
    false_clicks_times = jnp.where(mask, times, jnp.nan)

    # add the false clicks
    clicktimes_sme = jnp.concatenate([clicktimes_sme, false_clicks_times], -1)
    return clicktimes_sme.sort(-1)[..., :nmaxclick]


def measurements_sse_to_sme(
    measurements: ArrayLike, tsave: ArrayLike, etas: ArrayLike, key: PRNGKeyArray
) -> Array:
    r"""Post-process measurements from a diffusive SSE simulation to obtain measurements
    for a diffusive SME simulation.

    The SME measurements are obtained from the SSE measurements by adding additional
    Gaussian noise.

    Note:
        More precisely, the SME measurement record $\dd Y$ for a specific jump operator
        is defined as follows from the SSE measurement record $\dd \tilde Y$:
        $$
            \dd Y = \sqrt{\eta}\,\dd \tilde Y + \sqrt{1-\eta}\,\dd W
        $$
        where $\dd W$ is another independent Wiener process sampled for the
        post-processing.

        The SME time-averaged measurement $I(t_n, t_{n+1})$ is then defined as follows
        from the SSE time-averaged measurements $\tilde I(t_n, t_{n+1})$:
        $$
            I(t_n, t_{n+1})
            = \frac{1}{\Delta t_n}\int_{t_n}^{t_{n+1}} \dd Y(t)
            = \sqrt{\eta}\, \tilde I(t_n, t_{n+1})
            + \sqrt{1-\eta}\,\frac{\Delta W}{\sqrt{\Delta t_n}},
        $$
        where $\Delta t_n=t_{n+1}-t_n$ and $\Delta W\sim \mathcal{N}(0, 1)$ is
        a standard Gaussian random variable.

    Args:
        measurements: See result of [`dq.dssesolve()`][dynamiqs.dssesolve].
        tsave: See [`dq.dssesolve()`][dynamiqs.dssesolve].
        etas: See [`dq.dsmesolve()`][dynamiqs.dsmesolve].
        key: PRNG key used to sample the added noise for the post-processing.

    Returns:
        (array of shape (...)): SME measurements. The shape is the same as
            `measurements`, except that the dimension corresponding to the number of
            jump operators measured may be smaller, if the corresponding efficiency is
            null.
    """
    measurements = jnp.asarray(measurements)
    tsave = jnp.asarray(tsave)
    etas = jnp.asarray(etas)
    key = jnp.asarray(key)

    # select loss operators with non-zero efficiency
    Ik_sse = measurements[..., etas != 0, :]
    etas = etas[etas != 0]

    # for broadcasting later
    etas = etas[:, None]

    # \sqrt{\Delta}
    sDt = jnp.sqrt(jnp.diff(tsave))
    normal = jax.random.normal(key, shape=Ik_sse.shape)
    return jnp.sqrt(etas) * Ik_sse + jnp.sqrt(1 - etas) * normal / sDt
