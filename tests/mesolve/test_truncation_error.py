import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq
from dynamiqs.truncation_error import inner_outer_indices, truncation_error_rate


# the estimator compares a residual against zero, so single precision would drown the
# reference comparisons in round-off
@pytest.fixture(scope='module', autouse=True)
def _double_precision():
    # keep precision changes local to this module to avoid cross-test leakage.
    prev_x64 = jax.config.read('jax_enable_x64')
    dq.set_precision('double')
    yield
    dq.set_precision('double' if prev_x64 else 'single')


def pad(rho, dims, extended_dims):
    rho = rho.reshape(*dims, *dims)
    width = [(0, e - d) for d, e in zip(dims, extended_dims, strict=True)] * 2
    n_extended = int(np.prod(extended_dims))
    return jnp.pad(rho, width).reshape(n_extended, n_extended)


def lindbladian(H, Ls, rho):
    out = -1j * (H @ rho - rho @ H)
    for L in Ls:
        Ldag = L.conj().T
        out += L @ rho @ Ldag - 0.5 * (Ldag @ L @ rho + rho @ Ldag @ L)
    return out


def dense_rate(rho, H, Ls, H_extended, Ls_extended, dims, extended_dims):
    """Brute-force reference: full residual on the extended space, dense trace norm."""
    residual = lindbladian(
        H_extended, Ls_extended, pad(rho, dims, extended_dims)
    ) - pad(lindbladian(H, Ls, rho), dims, extended_dims)
    return jnp.abs(jnp.linalg.eigvalsh(residual)).sum(-1)


def random_density_matrix(n, seed):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    rho = matrix @ matrix.conj().T
    return jnp.asarray(rho / np.trace(rho))


def single_mode_model(jump_kind, dim):
    a = dq.destroy(dim)
    H = 0.3 * a.dag() @ a + 0.4 * (a @ a + a.dag() @ a.dag())
    jumps = {
        'lowering': [a],
        'raising': [a.dag()],
        'two_photon': [a @ a],
        'mixed': [a + a.dag()],
        'several': [a, a.dag(), a @ a + a.dag()],
    }[jump_kind]
    return H, jumps


def two_mode_model(na, nb):
    a, b = dq.destroy(na, nb)
    H = 0.3 * a.dag() @ a + 0.2 * b.dag() @ b + 0.1 * (a.dag() @ b + b.dag() @ a)
    return H, [a @ a - 4.0 * dq.eye_like(a), 0.5 * b, b.dag()]


@pytest.mark.parametrize(
    'jump_kind', ['lowering', 'raising', 'two_photon', 'mixed', 'several']
)
# a small space takes the dense path, a large one the compressed path
@pytest.mark.parametrize(('dim', 'extended_dim'), [(6, 10), (40, 44)])
def test_rate_matches_dense_residual(jump_kind, dim, extended_dim):
    rho = random_density_matrix(dim, seed=7)
    H, Ls = single_mode_model(jump_kind, dim)
    H_extended, Ls_extended = single_mode_model(jump_kind, extended_dim)
    inner, outer = inner_outer_indices((dim,), (extended_dim,))

    rate = truncation_error_rate(
        0.0,
        rho,
        dq.constant(H_extended),
        [dq.constant(L) for L in Ls_extended],
        inner,
        outer,
    )
    expected = dense_rate(
        rho,
        H.to_jax(),
        [L.to_jax() for L in Ls],
        H_extended.to_jax(),
        [L.to_jax() for L in Ls_extended],
        (dim,),
        (extended_dim,),
    )
    assert np.allclose(rate, expected, rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize(
    ('dims', 'extended_dims'), [((5, 4), (8, 6)), ((20, 14), (22, 15))]
)
def test_rate_matches_dense_residual_multimode(dims, extended_dims):
    rho = random_density_matrix(int(np.prod(dims)), seed=11)
    H, Ls = two_mode_model(*dims)
    H_extended, Ls_extended = two_mode_model(*extended_dims)
    inner, outer = inner_outer_indices(dims, extended_dims)

    rate = truncation_error_rate(
        0.0,
        rho,
        dq.constant(H_extended),
        [dq.constant(L) for L in Ls_extended],
        inner,
        outer,
    )
    expected = dense_rate(
        rho,
        H.to_jax(),
        [L.to_jax() for L in Ls],
        H_extended.to_jax(),
        [L.to_jax() for L in Ls_extended],
        dims,
        extended_dims,
    )
    assert np.allclose(rate, expected, rtol=1e-10, atol=1e-12)


def driven_oscillator(dim, drive=2.0):
    a = dq.destroy(dim)
    return drive * (a + a.dag()), [a]


def solve_with_estimate(dim, buffer, tsave, **kwargs):
    H, Ls = driven_oscillator(dim)
    H_extended, Ls_extended = driven_oscillator(dim + buffer)
    return dq.mesolve(
        H,
        Ls,
        dq.fock(dim, 0),
        tsave,
        truncation_error=dq.TruncationError(H_extended, Ls_extended),
        **kwargs,
    )


def test_estimate_is_zero_for_a_number_conserving_model():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    a, a_extended = dq.destroy(dim), dq.destroy(dim + 2)
    result = dq.mesolve(
        a.dag() @ a,
        [a],
        dq.fock(dim, 3),
        tsave,
        truncation_error=dq.TruncationError(
            a_extended.dag() @ a_extended, [a_extended]
        ),
    )
    assert np.allclose(result.truncation_error, 0.0, atol=1e-12)


def test_estimate_is_monotone_and_bounds_the_true_error():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    result = solve_with_estimate(dim, 1, tsave)
    xi = result.truncation_error

    assert xi.shape == tsave.shape
    assert xi[0] == 0.0
    assert bool(jnp.all(jnp.diff(xi) >= -1e-12))

    reference_dim = 40
    H, Ls = driven_oscillator(reference_dim)
    reference = dq.mesolve(H, Ls, dq.fock(reference_dim, 0), tsave)
    difference = (
        pad(result.states.to_jax()[-1], (dim,), (reference_dim,))
        - reference.states.to_jax()[-1]
    )
    true_error = jnp.abs(jnp.linalg.eigvalsh(difference)).sum()
    assert true_error <= xi[-1] + 1e-8


def test_estimate_decreases_with_the_truncature():
    tsave = jnp.linspace(0.0, 1.0, 11)
    finals = [
        solve_with_estimate(dim, 1, tsave).truncation_error[-1].item()
        for dim in (6, 10, 14)
    ]
    assert finals[0] > finals[1] > finals[2]


def test_estimate_is_unchanged_by_a_larger_buffer():
    # the residual is exactly representable with a buffer of 1 for `H = u (a + a^dag)`
    # and `L = a`, so enlarging the extended space must not change the estimate
    tsave = jnp.linspace(0.0, 1.0, 11)
    estimates = [
        solve_with_estimate(8, buffer, tsave).truncation_error for buffer in (1, 6)
    ]
    assert np.allclose(estimates[0], estimates[1], rtol=1e-6, atol=1e-12)


def test_rate_matches_the_driven_oscillator_closed_form():
    # for `H = u (a + a^dag)` with no dissipation the paper gives (Eqs. 29-30)
    # `xi_dot = 2 |u| sqrt(N + 1) sqrt(<N| rho^2 |N>)` with `N` the top retained level
    dim, drive = 6, 0.7
    rho = random_density_matrix(dim, seed=3)
    a_extended = dq.destroy(dim + 1)

    rate = truncation_error_rate(
        0.0,
        rho,
        dq.constant(drive * (a_extended + a_extended.dag())),
        [],
        *inner_outer_indices((dim,), (dim + 1,)),
    )
    expected = (
        2 * abs(drive) * jnp.sqrt(dim) * jnp.sqrt((rho @ rho)[dim - 1, dim - 1].real)
    )
    assert np.allclose(rate, expected, rtol=1e-10, atol=1e-12)


def test_estimate_is_batched_with_the_operators():
    dim, buffer, tsave = 8, 1, jnp.linspace(0.0, 1.0, 11)
    drives = jnp.array([0.5, 1.0, 2.0])[:, None, None]
    a, a_extended = dq.destroy(dim), dq.destroy(dim + buffer)
    result = dq.mesolve(
        drives * (a + a.dag()).to_jax(),
        [a],
        dq.fock(dim, 0),
        tsave,
        truncation_error=dq.TruncationError(
            drives * (a_extended + a_extended.dag()).to_jax(), [a_extended]
        ),
    )
    xi = result.truncation_error
    assert xi.shape == (3, 11)
    # a stronger drive populates higher Fock levels, so it truncates worse
    assert bool(jnp.all(jnp.diff(xi[:, -1]) > 0))
    # and each batch element matches the same solve run on its own
    for index, drive in enumerate([0.5, 1.0, 2.0]):
        single = dq.mesolve(
            drive * (a + a.dag()),
            [a],
            dq.fock(dim, 0),
            tsave,
            truncation_error=dq.TruncationError(
                drive * (a_extended + a_extended.dag()), [a_extended]
            ),
        )
        assert np.allclose(xi[index], single.truncation_error, rtol=1e-6, atol=1e-12)


def test_estimate_is_rejected_for_unsupported_methods():
    dim, tsave = 4, jnp.linspace(0.0, 1.0, 3)
    H, Ls = driven_oscillator(dim)
    H_extended, Ls_extended = driven_oscillator(dim + 1)
    spec = dq.TruncationError(H_extended, Ls_extended)
    with pytest.raises(TypeError, match='not supported for the method `Expm`'):
        dq.mesolve(
            H,
            Ls,
            dq.fock(dim, 0),
            tsave,
            method=dq.method.Expm(),
            truncation_error=spec,
        )
    with pytest.raises(
        ValueError, match='not supported together with `vectorized=True`'
    ):
        dq.mesolve(
            H, Ls, dq.fock(dim, 0), tsave, vectorized=True, truncation_error=spec
        )
    with pytest.raises(
        ValueError, match='one extended jump operator per jump operator'
    ):
        dq.mesolve(
            H,
            [],
            dq.fock(dim, 0),
            tsave,
            truncation_error=dq.TruncationError(H_extended, Ls_extended),
        )


def test_estimate_works_with_rouchon():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    reference = solve_with_estimate(
        dim, 1, tsave, method=dq.method.Tsit5(rtol=1e-10, atol=1e-12)
    ).truncation_error
    xi = solve_with_estimate(
        dim, 1, tsave, method=dq.method.Rouchon1(dt=1e-4)
    ).truncation_error

    assert bool(jnp.all(jnp.diff(xi) >= -1e-12))
    # Rouchon1 is first order, so the two agree only up to its own state error, which
    # dominates at short times where the estimate is still vanishingly small
    assert np.allclose(xi[-1], reference[-1], rtol=1e-2)
