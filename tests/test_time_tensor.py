import pytest
import torch
from torch import Tensor

from dynamiqs.time_array import (
    CallableTimeArray,
    ConstantTimeArray,
    ModulatedTimeArray,
    PWCTimeArray,
    _ModulatedFactor,
    _PWCFactor,
)


def assert_equal(xt: Tensor, y: list):
    assert torch.equal(xt, torch.tensor(y))


class TestConstantTimeArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = ConstantTimeArray(torch.tensor([1, 2]))

    def test_call(self):
        assert_equal(self.x(0.0), [1, 2])
        assert_equal(self.x(1.0), [1, 2])

    def test_call_caching(self):
        assert hash(self.x(0.0)) == hash(self.x(0.0))
        assert hash(self.x(1.0)) == hash(self.x(1.0))

    def test_view(self):
        x = self.x.reshape(1, 2)
        assert_equal(x(0.0), [[1, 2]])

    def test_adjoint(self):
        x = ConstantTimeArray(torch.tensor([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]))
        x = x.adjoint()
        res = torch.tensor([[1 - 1j, 3 - 3j], [2 - 2j, 4 - 4j]])
        assert torch.equal(x(0.0), res)

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [-1, -2])

    def test_mul(self):
        # test type `Number`
        x = self.x * 2
        assert_equal(x(0.0), [2, 4])

        # test type `Tensor`
        x = self.x * torch.tensor([0, 1])
        assert_equal(x(0.0), [0, 2])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [2, 4])

        # test type `Tensor`
        x = torch.tensor([0, 1]) * self.x
        assert_equal(x(0.0), [0, 2])

    def test_add(self):
        # test type `Tensor`
        x = self.x + torch.tensor([0, 1])
        assert isinstance(x, ConstantTimeArray)
        assert_equal(x(0.0), [1, 3])

        # test type `ConstantTimeArray`
        x = self.x + self.x
        assert isinstance(x, ConstantTimeArray)
        assert_equal(x(0.0), [2, 4])

    def test_radd(self):
        # test type `Tensor`
        x = torch.tensor([0, 1]) + self.x
        assert isinstance(x, ConstantTimeArray)
        assert_equal(x(0.0), [1, 3])


class TestCallableTimeArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        f = lambda t: t * torch.tensor([1, 2])
        self.x = CallableTimeArray(f, f(0.0))

    def test_call(self):
        assert_equal(self.x(0.0), [0, 0])
        assert_equal(self.x(1.0), [1, 2])

    def test_call_caching(self):
        assert hash(self.x(0.0)) == hash(self.x(0.0))
        assert hash(self.x(1.0)) == hash(self.x(1.0))

    def test_view(self):
        x = self.x.reshape(1, 2)
        assert_equal(x(0.0), [[0, 0]])
        assert_equal(x(1.0), [[1, 2]])

    def test_adjoint(self):
        f = lambda t: t * torch.tensor([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        x = CallableTimeArray(f, f(0.0))
        x = x.adjoint()
        res = torch.tensor([[1 - 1j, 3 - 3j], [2 - 2j, 4 - 4j]])
        assert torch.equal(x(1.0), res)

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [-1, -2])

    def test_mul(self):
        # test type `Number`
        x = self.x * 2
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

        # test type `Tensor`
        x = self.x * torch.tensor([0, 1])
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [0, 2])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

        # test type `Tensor`
        x = torch.tensor([0, 1]) * self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [0, 2])

    def test_add(self):
        # test type `Tensor`
        x = self.x + torch.tensor([0, 1])
        assert isinstance(x, CallableTimeArray)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])

        # test type `CallableTimeArray`
        x = self.x + self.x
        assert isinstance(x, CallableTimeArray)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

    def test_radd(self):
        # test type `Tensor`
        x = torch.tensor([0, 1]) + self.x
        assert isinstance(x, CallableTimeArray)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])


class TestPWCTimeArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        # PWC factor 1
        t1 = torch.tensor([0, 1, 2, 3])
        v1 = torch.tensor([1, 10, 100])
        f1 = _PWCFactor(t1, v1)
        tensor1 = torch.tensor([[1, 2], [3, 4]])

        # PWC factor 2
        t2 = torch.tensor([1, 3, 5])
        v2 = torch.tensor([1, 1])
        f2 = _PWCFactor(t2, v2)
        tensor2 = torch.tensor([[1j, 1j], [1j, 1j]])

        factors = [f1, f2]
        tensors = torch.stack([tensor1, tensor2])
        self.x = PWCTimeArray(factors, tensors)  # shape at t: (2, 2)

    def test_call(self):
        assert_equal(self.x(-0.1), [[0, 0], [0, 0]])
        assert_equal(self.x(0.0), [[1, 2], [3, 4]])
        assert_equal(self.x(0.5), [[1, 2], [3, 4]])
        assert_equal(self.x(0.999), [[1, 2], [3, 4]])
        assert_equal(self.x(1.0), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])
        assert_equal(self.x(1.999), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])
        assert_equal(self.x(3.0), [[1j, 1j], [1j, 1j]])
        assert_equal(self.x(5.0), [[0, 0], [0, 0]])

    def test_call_caching(self):
        assert hash(self.x(-0.1)) == hash(self.x(-0.1)) == hash(self.x(-0.2))
        assert hash(self.x(0.0)) == hash(self.x(0.0)) == hash(self.x(0.5))
        assert hash(self.x(1.0)) == hash(self.x(1.0)) == hash(self.x(1.999))
        assert hash(self.x(5.0)) == hash(self.x(5.0)) == hash(self.x(6.0))

    def test_view(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(-0.1), [[[0, 0], [0, 0]]])
        assert_equal(x(0.0), [[[1, 2], [3, 4]]])

    def test_adjoint(self):
        x = self.x.adjoint()
        assert_equal(x(1.0), [[10 - 1j, 30 - 1j], [20 - 1j, 40 - 1j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1, -2], [-3, -4]])

    def test_mul(self):
        # test type `Number`
        x = self.x * 2
        assert_equal(x(0.0), [[2, 4], [6, 8]])

        # test type `Tensor`
        x = self.x * torch.tensor([2])
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [[2, 4], [6, 8]])

        # test type `Tensor`
        x = torch.tensor([2]) * self.x
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_add(self):
        tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)

        # test type `Tensor`
        x = self.x + tensor
        assert isinstance(x, PWCTimeArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `PWCTimeArray`
        x = self.x + self.x
        assert isinstance(x, PWCTimeArray)
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_radd(self):
        tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)

        # test type `Tensor`
        x = tensor + self.x
        assert isinstance(x, PWCTimeArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])


class TestModulatedTimeArray:
    @pytest.fixture(autouse=True)
    def setup(self):
        one = torch.tensor(1.0)

        # modulated factor 1
        eps1 = lambda t: (0.5 * t + 1.0j) * one
        eps1_0 = eps1(0.0)
        f1 = _ModulatedFactor(eps1, eps1_0)
        tensor1 = torch.tensor([[1, 2], [3, 4]])

        # modulated factor 2
        eps2 = lambda t: t**2 * one
        eps2_0 = eps2(0.0)
        f2 = _ModulatedFactor(eps2, eps2_0)
        tensor2 = torch.tensor([[1j, 1j], [1j, 1j]])

        factors = [f1, f2]
        tensors = torch.stack([tensor1, tensor2])
        self.x = ModulatedTimeArray(factors, tensors)

    def test_call(self):
        assert_equal(self.x(0.0), [[1.0j, 2.0j], [3.0j, 4.0j]])
        assert_equal(self.x(2.0), [[1.0 + 5.0j, 2.0 + 6.0j], [3.0 + 7.0j, 4.0 + 8.0j]])

    def test_view(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[1.0j, 2.0j], [3.0j, 4.0j]]])
        assert_equal(x(2.0), [[[1.0 + 5.0j, 2.0 + 6.0j], [3.0 + 7.0j, 4.0 + 8.0j]]])

    def test_adjoint(self):
        x = self.x.adjoint()
        assert_equal(x(0.0), [[-1.0j, -3.0j], [-2.0j, -4.0j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_mul(self):
        # test type `Number`
        x = self.x * 2
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

        # test type `Tensor`
        x = self.x * torch.tensor([2])
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

        # test type `Tensor`
        x = torch.tensor([2]) * self.x
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_add(self):
        tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)

        # test type `Tensor`
        x = self.x + tensor
        assert isinstance(x, ModulatedTimeArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ModulatedTimeArray`
        x = self.x + self.x
        assert isinstance(x, ModulatedTimeArray)
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_radd(self):
        tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)

        # test type `Tensor`
        x = tensor + self.x
        assert isinstance(x, ModulatedTimeArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])
