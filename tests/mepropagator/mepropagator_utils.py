import jax

import dynamiqs as dq


def rand_mepropagator_args(n, nH, nLs):
    nkeys = len(nLs) + 1
    kH, *kLs = jax.random.split(jax.random.PRNGKey(42), nkeys)
    H = dq.random.herm(kH, (*nH, n, n))
    Ls = [dq.random.herm(kL, (*nL, n, n)) for kL, nL in zip(kLs, nLs)]
    return H, Ls
