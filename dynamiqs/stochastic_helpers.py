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
        (array of shape (..., ntrajs, nLm, nmaxclick)): SME click times, where _nLm_
            is the number of jump operators with non-zero efficiency. Jump operators
            whose efficiency is zero are removed.

    Examples:
        For two jump operators, keep all clicks of the first one and drop the second
        one:
        >>> clicktimes = jnp.array([[[0.1, 0.5, jnp.nan], [0.3, jnp.nan, jnp.nan]]])
        >>> tsave = jnp.linspace(0.0, 1.0, 11)
        >>> key = jax.random.PRNGKey(42)
        >>> dq.clicktimes_sse_to_sme(clicktimes, tsave, [0.0, 0.0], [1.0, 0.0], key)
        Array([[[0.1, 0.5, nan]]], dtype=float32)

        With false clicks and imperfect efficiency for the first jump operator:
        >>> dq.clicktimes_sse_to_sme(clicktimes, tsave, [2.0, 0.0], [0.5, 1.0], key)
        Array([[[0.5  , 0.667,   nan],
                [0.3  ,   nan,   nan]]], dtype=float32)
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

    Args:
        measurements: See result of [`dq.dssesolve()`][dynamiqs.dssesolve].
        tsave: See [`dq.dssesolve()`][dynamiqs.dssesolve].
        etas: See [`dq.dsmesolve()`][dynamiqs.dsmesolve].
        key: PRNG key used to sample the added noise for the post-processing.

    Returns:
        (array of shape (..., ntrajs, nLm, ntsave-1)): SME measurements, where _nLm_ is
            the number of jump operators with non-zero efficiency. Jump operators whose
            efficiency is zero are removed.

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
            + \sqrt{1-\eta}\,\frac{\xi}{\sqrt{\Delta t_n}},
        $$
        where $\Delta t_n=t_{n+1}-t_n$ and $\xi\sim \mathcal{N}(0, 1)$ is
        a standard Gaussian random variable.

    Examples:
        For two jump operators, keep the measurements of the first one and drop the
        second one:
        >>> measurements = jnp.array([[[0.2, -0.1, 0.4], [1.0, 0.5, 0.0]]])
        >>> tsave = jnp.linspace(0.0, 1.0, 4)
        >>> key = jax.random.PRNGKey(42)
        >>> dq.measurements_sse_to_sme(measurements, tsave, [1.0, 0.0], key)
        Array([[[ 0.2, -0.1,  0.4]]], dtype=float32)

        With imperfect efficiency for the first jump operator:
        >>> dq.measurements_sse_to_sme(measurements, tsave, [0.5, 1.0], key)
        Array([[[0.107, 0.501, 0.645],
                [1.   , 0.5  , 0.   ]]], dtype=float32)
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
