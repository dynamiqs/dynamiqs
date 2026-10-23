import jax
import pytest

import dynamiqs as dq

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
@pytest.mark.parametrize(
    ('matmul_precision', 'jax_matmul_precision'),
    [('low', 'bfloat16'), ('high', 'high'), ('highest', 'highest')],
)
def test_set_matmul_precision(matmul_precision, jax_matmul_precision):
    try:
        dq.set_matmul_precision(matmul_precision)
        assert jax.config.jax_default_matmul_precision == jax_matmul_precision
    finally:
        dq.set_matmul_precision('highest')
