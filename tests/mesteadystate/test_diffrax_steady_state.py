import pytest

from dynamiqs.method import DiffraxSteadyState, Expm

from ..order import TEST_INSTANT


@pytest.mark.run(order=TEST_INSTANT)
def test_unsupported_me_method():
    with pytest.raises(TypeError, match='Argument `me_method` must be one of'):
        DiffraxSteadyState(Expm())
