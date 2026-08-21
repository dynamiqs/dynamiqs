import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dynamiqs as dq
from dynamiqs.integrators._utils import astimeqarray
from dynamiqs.truncation_error import (
    assumed_degree,
    extend_qarray,
    extension_buffer,
    inner_outer_indices,
    offset_digits,
    truncation_error_rate,
)


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


def restrict(qarray, dims, restricted_dims):
    """Exact truncation of an operator built on a much larger Fock space."""
    flat = [
        np.ravel_multi_index(occupation, dims)
        for occupation in itertools.product(*[range(d) for d in restricted_dims])
    ]
    matrix = np.asarray(qarray.to_jax())[np.ix_(flat, flat)]
    return dq.asqarray(matrix, dims=restricted_dims, layout=dq.dia)


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


# === deriving the extended-space operators


def polynomial_models():
    """Normal-ordered operators built on a large space, with a per-mode degree."""
    a, alpha = dq.destroy(80), 1.0
    adag, eye = a.dag(), dq.eye(80)
    cosh, sinh = np.cosh(1.25), np.sinh(1.25)
    A, B = dq.destroy(60, 30)
    coupling = (A @ A - alpha**2 * dq.eye(60, 30)) @ B.dag()
    sz, osc = dq.tensor(dq.sigmaz(), dq.eye(40)), dq.tensor(dq.eye(2), dq.destroy(40))
    return {
        'two_photon': (a @ a - alpha**2 * eye, (80,), (16,), (20,), 2),
        'squeezed_cat': (
            cosh**2 * (a @ a)
            + 2 * cosh * sinh * (adag @ a)
            + sinh**2 * (adag @ adag)
            + (cosh * sinh - alpha**2) * eye,
            (80,),
            (16,),
            (20,),
            2,
        ),
        'kerr_drive': (adag @ adag @ a @ a + a + adag, (80,), (16,), (20,), 4),
        'cat_with_buffer': (coupling + coupling.dag(), (60, 30), (16, 8), (20, 10), 4),
        'qubit_oscillator': (sz @ (osc + osc.dag()), (2, 40), (2, 16), (2, 20), 4),
    }


@pytest.mark.parametrize('model', polynomial_models())
def test_extension_reproduces_the_operator_at_the_larger_dimension(model):
    operator, dims, truncated_dims, extended_dims, degree = polynomial_models()[model]
    truncated = restrict(operator, dims, truncated_dims)
    expected = restrict(operator, dims, extended_dims)

    extended = extend_qarray(truncated, extended_dims, degree, 'H')

    assert extended.dims == extended_dims
    assert extended.layout == dq.dia
    assert np.allclose(extended.to_jax(), expected.to_jax(), rtol=1e-8, atol=1e-10)


def test_extension_carries_the_operator_batch_dimensions():
    dim, extended_dim = (12,), (14,)
    drives = jnp.array([0.5, 1.0, 2.0])[:, None, None]
    a = dq.destroy(*dim)
    batched = drives * (a + a.dag())

    extended = extend_qarray(batched, extended_dim, 2, 'H')

    assert extended.shape == (3, 14, 14)
    expected = (
        jnp.array([0.5, 1.0, 2.0])[:, None, None]
        * (dq.destroy(*extended_dim) + dq.destroy(*extended_dim).dag()).to_jax()
    )
    assert np.allclose(extended.to_jax(), expected, rtol=1e-8, atol=1e-10)


def test_extension_of_an_operator_with_no_stored_diagonal():
    # a zero placeholder term can be stored with no diagonal at all
    zero = dq.asqarray(jnp.zeros((4, 4)), layout=dq.dia)

    extended = extend_qarray(zero, (6,), 4, 'H')

    assert extended.dims == (6,)
    assert np.allclose(extended.to_jax(), 0.0)


def test_offset_digits_splits_a_flat_offset_per_mode():
    single = [offset_digits(o, (8,)) for o in (-7, -1, 0, 3)]
    assert single == [(-7,), (-1,), (0,), (3,)]
    # with dims=(16, 8) the second mode's stride is 8, so -15 = -2 * 8 + 1
    assert offset_digits(-15, (16, 8)) == (-2, 1)
    assert offset_digits(15, (16, 8)) == (2, -1)
    assert offset_digits(1, (16, 8)) == (0, 1)


def test_offset_digits_rejects_an_ambiguous_split():
    # on a 4-level trailing mode the flat offset 3 is shared by `b @ b @ b`, of shifts
    # (0, 3), and `a @ b.dag()`, of shifts (1, -1)
    assert offset_digits(3, (8, 4)) == (1, -1)  # what the balanced digits guess
    with pytest.raises(ValueError, match='unambiguously'):
        offset_digits(3, (8, 4), 4)
    # a smaller declared degree rules the wrap-around out
    assert offset_digits(3, (8, 4), 2) == (1, -1)
    # the paper's two-mode model is unambiguous at the default degree
    assert offset_digits(15, (16, 8), 4) == (2, -1)
    assert offset_digits(1, (16, 8), 4) == (0, 1)


@pytest.mark.parametrize(
    ('name', 'buffer'),
    [
        # every case is the buffer stated by the paper for that model
        ('driven', (1,)),
        ('lowering', (1,)),
        ('two_photon', (2,)),
        ('squeezed_cat', (4,)),
        ('cat_with_buffer', (2, 1)),
        ('number_conserving', (0,)),
    ],
)
def test_buffer_matches_the_paper(name, buffer):
    alpha = 1.0
    a = dq.destroy(16)
    cosh, sinh = np.cosh(1.25), np.sinh(1.25)
    A, B = dq.destroy(16, 8)
    coupling = (A @ A - alpha**2 * dq.eye(16, 8)) @ B.dag()
    zero = 0.0 * dq.eye(16)
    models = {
        'driven': (a + a.dag(), []),
        'lowering': (zero, [a]),
        'two_photon': (zero, [a @ a - alpha**2 * dq.eye(16)]),
        'squeezed_cat': (
            zero,
            [
                cosh**2 * (a @ a)
                + 2 * cosh * sinh * (a.dag() @ a)
                + sinh**2 * (a.dag() @ a.dag())
                + (cosh * sinh - alpha**2) * dq.eye(16)
            ],
        ),
        'cat_with_buffer': (coupling + coupling.dag(), [B]),
        'number_conserving': (a.dag() @ a, [a.dag() @ a]),
    }
    H, Ls = models[name]
    H, Ls = astimeqarray(H), [astimeqarray(L) for L in Ls]
    assert extension_buffer(H, Ls) == buffer


def test_two_level_modes_are_exact():
    # a genuine qubit cannot leak, so it gets no buffer and no estimate of its own
    nq, no, tsave = 2, 8, jnp.linspace(0.0, 1.0, 6)
    sigmam = dq.tensor(dq.sigmam(), dq.eye(no))
    a = dq.tensor(dq.eye(nq), dq.destroy(no))
    # Jaynes-Cummings, with a drive to actually populate the oscillator
    H = 0.5 * (sigmam.dag() @ a + sigmam @ a.dag()) + 1.0 * (a + a.dag())
    assert extension_buffer(astimeqarray(H), [astimeqarray(a)], 4) == (0, 1)

    rho0 = dq.tensor(dq.fock(nq, 0), dq.fock(no, 0))
    xi = dq.mesolve(H, [a], rho0, tsave, truncation_error=True).truncation_error
    assert bool(jnp.all(jnp.diff(xi) >= -1e-12))

    # and it bounds the error against a much larger oscillator
    big = 30
    sigmam_big = dq.tensor(dq.sigmam(), dq.eye(big))
    a_big = dq.tensor(dq.eye(nq), dq.destroy(big))
    H_big = 0.5 * (sigmam_big.dag() @ a_big + sigmam_big @ a_big.dag()) + 1.0 * (
        a_big + a_big.dag()
    )
    reference = dq.mesolve(
        H_big, [a_big], dq.tensor(dq.fock(nq, 0), dq.fock(big, 0)), tsave
    )
    difference = (
        pad(
            dq.mesolve(H, [a], rho0, tsave).states.to_jax()[-1],
            (nq, no),
            (nq, big),
        )
        - reference.states.to_jax()[-1]
    )
    assert dq.norm(difference) <= xi[-1] + 1e-8

    # a qubit on its own has nothing to truncate
    qubit = dq.mesolve(
        dq.sigmax(), [dq.sigmam()], dq.fock(2, 0), tsave, truncation_error=True
    )
    assert np.allclose(qubit.truncation_error, 0.0)


def test_derived_buffer_is_large_enough():
    # the derived buffer must saturate the rate: enlarging the extended space past it
    # cannot change the residual
    dim, alpha = 10, 1.0
    a = dq.destroy(dim)
    rho = random_density_matrix(dim, seed=5)
    H, Ls = 0.4 * (a + a.dag()), [a @ a - alpha**2 * dq.eye(dim), a.dag()]
    buffer = extension_buffer(astimeqarray(H), [astimeqarray(L) for L in Ls])
    assert buffer == (2,)

    def rate_at(extra):
        extended_dim = dim + buffer[0] + extra
        big = dq.destroy(extended_dim)
        return truncation_error_rate(
            0.0,
            rho,
            dq.constant(0.4 * (big + big.dag())),
            [
                dq.constant(big @ big - alpha**2 * dq.eye(extended_dim)),
                dq.constant(big.dag()),
            ],
            *inner_outer_indices((dim,), (extended_dim,)),
        )

    assert np.allclose(rate_at(0), rate_at(4), rtol=1e-10, atol=1e-12)


def test_assumed_degree_covers_the_offsets_present():
    a = dq.destroy(16)
    # nothing above degree 4 in the offsets, so the default is kept
    assert assumed_degree(astimeqarray(a + a.dag()), []) == 4
    # a bare power reaches further than the default, and is picked up on its own
    sixth = astimeqarray(a @ a @ a @ a @ a @ a)
    assert assumed_degree(sixth, []) == 6


# === end-to-end through `dq.mesolve()`


def driven_oscillator(dim, drive=2.0):
    a = dq.destroy(dim)
    return drive * (a + a.dag()), [a]


def solve_with_estimate(dim, tsave, truncation_error=True, **kwargs):
    H, Ls = driven_oscillator(dim)
    return dq.mesolve(
        H, Ls, dq.fock(dim, 0), tsave, truncation_error=truncation_error, **kwargs
    )


def test_estimate_is_zero_for_a_number_conserving_model():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    a = dq.destroy(dim)
    result = dq.mesolve(a.dag() @ a, [a], dq.fock(dim, 3), tsave, truncation_error=True)
    assert np.allclose(result.truncation_error, 0.0, atol=1e-12)


def test_estimate_is_monotone_and_bounds_the_true_error():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    result = solve_with_estimate(dim, tsave)
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
        solve_with_estimate(dim, tsave).truncation_error[-1].item()
        for dim in (6, 10, 14)
    ]
    assert finals[0] > finals[1] > finals[2]


def test_estimate_is_unchanged_by_a_larger_declared_degree():
    # over-declaring the degree only adds monomials that are fitted to zero
    tsave = jnp.linspace(0.0, 1.0, 11)
    estimates = [
        solve_with_estimate(8, tsave, truncation_error=degree) for degree in (2, 4, 8)
    ]
    for other in estimates[1:]:
        assert np.allclose(
            estimates[0].truncation_error, other.truncation_error, rtol=1e-6, atol=1e-12
        )


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
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    drives = jnp.array([0.5, 1.0, 2.0])[:, None, None]
    a = dq.destroy(dim)
    result = dq.mesolve(
        drives * (a + a.dag()), [a], dq.fock(dim, 0), tsave, truncation_error=True
    )
    xi = result.truncation_error
    assert xi.shape == (3, 11)
    # a stronger drive populates higher Fock levels, so it truncates worse
    assert bool(jnp.all(jnp.diff(xi[:, -1]) > 0))
    # and each batch element matches the same solve run on its own
    for index, drive in enumerate([0.5, 1.0, 2.0]):
        single = dq.mesolve(
            drive * (a + a.dag()), [a], dq.fock(dim, 0), tsave, truncation_error=True
        )
        assert np.allclose(xi[index], single.truncation_error, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize('kind', ['constant', 'pwc', 'modulated', 'summed'])
def test_estimate_supports_time_dependent_operators(kind):
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    a = dq.destroy(dim)
    drive = a + a.dag()
    H = {
        'constant': 2.0 * drive,
        'pwc': dq.pwc([0.0, 0.5, 1.0], [1.0, 3.0], drive),
        'modulated': dq.modulated(lambda t: 2.0 * jnp.cos(t), drive),
        'summed': dq.modulated(lambda t: 2.0 * jnp.cos(t), drive) + a.dag() @ a,
    }[kind]
    xi = dq.mesolve(
        H, [a], dq.fock(dim, 0), tsave, truncation_error=True
    ).truncation_error
    assert bool(jnp.all(jnp.diff(xi) >= -1e-12))
    assert xi[-1] > 0.0


@pytest.mark.parametrize('dim', [3, 6])  # the dense path, then the compressed one
def test_estimate_is_differentiable(dim):
    # the residual is low rank by construction, so the range generators of its
    # compressed form are rank deficient and must not be differentiated through
    tsave = jnp.linspace(0.0, 1.0, 5)

    def final_estimate(drive):
        a = dq.destroy(dim)
        return dq.mesolve(
            drive * (a + a.dag()),
            [a],
            dq.fock(dim, 0),
            tsave,
            method=dq.method.Rouchon1(dt=1e-2),  # fixed steps, for a clean comparison
            truncation_error=True,
        ).truncation_error[-1]

    grad = jax.grad(final_estimate)(0.8)
    finite_difference = (final_estimate(0.8 + 1e-5) - final_estimate(0.8 - 1e-5)) / 2e-5
    assert np.allclose(grad, finite_difference, rtol=1e-5)


def test_estimate_respects_the_clipping_of_a_summed_operator():
    # the clipping of a sum lives on the sum itself, and must survive the extension
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    a = dq.destroy(dim)
    H = (dq.modulated(lambda t: 2.0 * jnp.cos(t), a + a.dag()) + a.dag() @ a).clip(
        0.0, 0.2
    )
    xi = dq.mesolve(
        H, [a], dq.fock(dim, 0), tsave, truncation_error=True
    ).truncation_error

    # `H` is null past `tend` and `L = a` cannot leak out of the truncature, so the
    # estimate grows while the drive is on and is constant afterwards
    assert xi[3] > 0.0
    assert np.allclose(xi[3:], xi[3], rtol=1e-6)


def test_estimate_is_rejected_for_unsupported_methods():
    dim, tsave = 4, jnp.linspace(0.0, 1.0, 3)
    H, Ls = driven_oscillator(dim)
    with pytest.raises(TypeError, match='not supported for the method `Expm`'):
        dq.mesolve(
            H,
            Ls,
            dq.fock(dim, 0),
            tsave,
            method=dq.method.Expm(),
            truncation_error=True,
        )
    with pytest.raises(
        ValueError, match='not supported together with `vectorized=True`'
    ):
        dq.mesolve(
            H, Ls, dq.fock(dim, 0), tsave, vectorized=True, truncation_error=True
        )
    with pytest.raises(TypeError, match='must be a bool or an int'):
        dq.mesolve(H, Ls, dq.fock(dim, 0), tsave, truncation_error='2')


def test_estimate_is_rejected_for_operators_it_cannot_derive():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 3)
    a = dq.destroy(dim)
    rho0 = dq.fock(dim, 0)
    drive = a + a.dag()

    with pytest.raises(ValueError, match='must have layout `dia`'):
        dq.mesolve(drive.to_jax(), [a], rho0, tsave, truncation_error=True)

    with pytest.raises(ValueError, match='does not support'):
        dq.mesolve(
            dq.timecallable(lambda t: 2.0 * jnp.cos(t) * drive),
            [a],
            rho0,
            tsave,
            truncation_error=True,
        )

    # degree 8 needs more entries on a diagonal than a 3-level mode provides
    small = dq.destroy(3)
    with pytest.raises(ValueError, match='usable entries on the diagonal'):
        dq.mesolve(
            small + small.dag(), [small], dq.fock(3, 0), tsave, truncation_error=8
        )

    # `H` is a polynomial of degree 6, and the default degree of 4 cannot fit it
    kerr = a.dag() @ a.dag() @ a.dag() @ a @ a @ a + drive
    with pytest.raises(Exception, match='not a normal-ordered polynomial'):
        dq.mesolve(kerr, [a], rho0, tsave, truncation_error=True)
    # declaring the right degree fixes it
    xi = dq.mesolve(kerr, [a], rho0, tsave, truncation_error=6).truncation_error
    assert xi[-1] > 0.0


def test_estimate_works_with_rouchon():
    dim, tsave = 8, jnp.linspace(0.0, 1.0, 11)
    reference = solve_with_estimate(
        dim, tsave, method=dq.method.Tsit5(rtol=1e-10, atol=1e-12)
    ).truncation_error
    xi = solve_with_estimate(
        dim, tsave, method=dq.method.Rouchon1(dt=1e-4)
    ).truncation_error

    assert bool(jnp.all(jnp.diff(xi) >= -1e-12))
    # Rouchon1 is first order, so the two agree only up to its own state error, which
    # dominates at short times where the estimate is still vanishingly small
    assert np.allclose(xi[-1], reference[-1], rtol=1e-2)
