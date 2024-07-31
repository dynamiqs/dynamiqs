import dynamiqs as dq


def test_global_dispatch():
    dim = 4

    # default: sparse DIA
    assert isinstance(dq.eye(dim), dq.SparseDIAQArray)
    assert isinstance(dq.eye(dim, matrix_format=dq.dense), dq.DenseQArray)

    dq.set_matrix_format(dq.dense)
    assert isinstance(dq.eye(dim), dq.DenseQArray)
    assert isinstance(dq.eye(dim, matrix_format=dq.dia), dq.SparseDIAQArray)

    dq.set_matrix_format(dq.dia)
    assert isinstance(dq.eye(dim), dq.SparseDIAQArray)
    assert isinstance(dq.eye(dim, matrix_format=dq.dense), dq.DenseQArray)


def test_operators_dispatch():
    dims = (3, 4)
    dim = 20
    alpha = 2
    n = 4

    # assert jnp.allclose(
    #     dq.eye(*dims, matrix_format=dq.dense).to_jax(),
    #     dq.eye(*dims, matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.zero(*dims, matrix_format=dq.dense).to_jax(),
    #     dq.zero(*dims, matrix_format=dq.dia).to_jax(),
    # )
    #
    # # === dq.destroy ===
    #
    # assert jnp.allclose(
    #     dq.destroy(*dims, matrix_format=dq.dense)[0].to_jax(),
    #     dq.destroy(*dims, matrix_format=dq.dia)[0].to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.destroy(*dims, matrix_format=dq.dense)[1].to_jax(),
    #     dq.destroy(*dims, matrix_format=dq.dia)[1].to_jax(),
    # )
    #
    # # === dq.create ===
    #
    # assert jnp.allclose(
    #     dq.create(*dims, matrix_format=dq.dense)[0].to_jax(),
    #     dq.create(*dims, matrix_format=dq.dia)[0].to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.create(*dims, matrix_format=dq.dense)[1].to_jax(),
    #     dq.create(*dims, matrix_format=dq.dia)[1].to_jax(),
    # )
    #
    # # === end dq.create ===
    #
    # assert jnp.allclose(
    #     dq.number(dim, matrix_format=dq.dense).to_jax(),
    #     dq.number(dim, matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.parity(dim, matrix_format=dq.dense).to_jax(),
    #     dq.parity(dim, matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.quadrature(dim, 0.0, matrix_format=dq.dense).to_jax(),
    #     dq.quadrature(dim, 0.0, matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.position(dim, matrix_format=dq.dense).to_jax(),
    #     dq.position(dim, matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.momentum(dim, matrix_format=dq.dense).to_jax(),
    #     dq.momentum(dim, matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.sigmax(matrix_format=dq.dense).to_jax(),
    #     dq.sigmax(matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.sigmay(matrix_format=dq.dense).to_jax(),
    #     dq.sigmay(matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.sigmaz(matrix_format=dq.dense).to_jax(),
    #     dq.sigmaz(matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.sigmap(matrix_format=dq.dense).to_jax(),
    #     dq.sigmap(matrix_format=dq.dia).to_jax(),
    # )
    #
    # assert jnp.allclose(
    #     dq.sigmam(matrix_format=dq.dense).to_jax(),
    #     dq.sigmam(matrix_format=dq.dia).to_jax(),
    # )
    #

    print(dq.hadamard(2, matrix_format=dq.dia))

    # assert jnp.allclose(
    #     dq.hadamard(n, matrix_format=dq.dense).to_jax(),
    #     dq.hadamard(n, matrix_format=dq.dia).to_jax(),
    # )
