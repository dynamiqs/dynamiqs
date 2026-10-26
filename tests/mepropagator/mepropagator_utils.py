import jax

import dynamiqs as dq


def rand_mepropagator_args(n, nH, nLs):
    nkeys = len(nLs) + 1
    kH, *kLs = jax.random.split(jax.random.PRNGKey(42), nkeys)
    H = dq.random.operator(kH, n, batch=nH)
    Ls = [dq.random.operator(kL, n, batch=nL) for kL, nL in zip(kLs, nLs, strict=True)]
    return H, Ls
