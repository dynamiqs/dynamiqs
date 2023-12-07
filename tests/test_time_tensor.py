import pytest
import torch
from torch import Tensor

from dynamiqs.time_tensor import (
    CallableTimeTensor,
    ConstantTimeTensor,
    PWCTimeTensor,
    _PWCTimeTensor,
)


def assert_equal(xt: Tensor, y: list):
    assert torch.equal(xt, torch.tensor(y))


class TestConstantTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = ConstantTimeTensor(torch.tensor([1, 2]))

    def test_call(self):
        assert_equal(self.x(0.0), [1, 2])
        assert_equal(self.x(1.0), [1, 2])

    def test_call_caching(self):
        assert hash(self.x(0.0)) == hash(self.x(0.0))
        assert hash(self.x(1.0)) == hash(self.x(1.0))

    def test_view(self):
        x = self.x.view(1, 2)
        assert_equal(x(0.0), [[1, 2]])

    def test_adjoint(self):
        x = ConstantTimeTensor(torch.tensor([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]))
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
        assert isinstance(x, ConstantTimeTensor)
        assert_equal(x(0.0), [1, 3])

        # test type `ConstantTimeTensor`
        x = self.x + self.x
        assert isinstance(x, ConstantTimeTensor)
        assert_equal(x(0.0), [2, 4])

    def test_radd(self):
        # test type `Tensor`
        x = torch.tensor([0, 1]) + self.x
        assert isinstance(x, ConstantTimeTensor)
        assert_equal(x(0.0), [1, 3])


class TestCallableTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        f = lambda t: t * torch.tensor([1, 2])
        self.x = CallableTimeTensor(f, f(0.0))

    def test_call(self):
        assert_equal(self.x(0.0), [0, 0])
        assert_equal(self.x(1.0), [1, 2])

    def test_call_caching(self):
        assert hash(self.x(0.0)) == hash(self.x(0.0))
        assert hash(self.x(1.0)) == hash(self.x(1.0))

    def test_view(self):
        x = self.x.view(1, 2)
        assert_equal(x(0.0), [[0, 0]])
        assert_equal(x(1.0), [[1, 2]])

    def test_adjoint(self):
        f = lambda t: t * torch.tensor([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        x = CallableTimeTensor(f, f(0.0))
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
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])

        # test type `CallableTimeTensor`
        x = self.x + self.x
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

    def test_radd(self):
        # test type `Tensor`
        x = torch.tensor([0, 1]) + self.x
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])


class TestPrivatePWCTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        times = torch.tensor([0.0, 1.0, 2.0, 3.0])
        values = torch.tensor([10, 100, 1000])  # (3,)
        tensor = torch.tensor([[1, 2], [3, 4]])  # (2, 2)
        self.x = _PWCTimeTensor(times, values, tensor)  # shape (2, 2)

    def test_call(self):
        assert_equal(self.x(-0.1), [[0, 0], [0, 0]])
        assert_equal(self.x(0.0), [[10, 20], [30, 40]])
        assert_equal(self.x(0.5), [[10, 20], [30, 40]])
        assert_equal(self.x(0.999), [[10, 20], [30, 40]])
        assert_equal(self.x(1.0), [[100, 200], [300, 400]])
        assert_equal(self.x(1.999), [[100, 200], [300, 400]])
        assert_equal(self.x(3.0), [[0, 0], [0, 0]])
        assert_equal(self.x(5.0), [[0, 0], [0, 0]])

    def test_view(self):
        x = self.x.view(1, 2, 2)
        assert_equal(x(0.0), [[[10, 20], [30, 40]]])

    def test_adjoint(self):
        times = torch.tensor([0.0, 1.0, 2.0])
        values = torch.tensor([10, 100])
        tensor = torch.tensor([[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]])
        x = _PWCTimeTensor(times, values, tensor)
        x = x.adjoint()
        res = torch.tensor([[10 - 10j, 30 - 30j], [20 - 20j, 40 - 40j]])
        assert torch.equal(x(0.0), res)

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-10, -20], [-30, -40]])

    def test_mul(self):
        # test type `Number`
        x = self.x * 2
        assert_equal(x(0.0), [[20, 40], [60, 80]])

        # test type `Tensor`
        x = self.x * torch.tensor([2])
        assert_equal(x(0.0), [[20, 40], [60, 80]])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [[20, 40], [60, 80]])

        # test type `Tensor`
        x = torch.tensor([2]) * self.x
        assert_equal(x(0.0), [[20, 40], [60, 80]])


class TestPWCTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        times = torch.tensor([0.0, 1.0, 2.0, 3.0])
        values = torch.tensor([10, 100, 1000])
        tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.complex64)
        pwc1 = _PWCTimeTensor(times, values, tensor)

        times = torch.tensor([1.0, 2.0, 3.0, 4.0])
        values = torch.tensor([1, 1, 1])
        tensor = torch.tensor([[1j, 2j], [3j, 4j]])
        pwc2 = _PWCTimeTensor(times, values, tensor)

        static = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)
        self.x = PWCTimeTensor([pwc1, pwc2], static=static)

    def test_init(self):
        assert torch.equal(self.x.times, torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0]))

    def test_call(self):
        assert_equal(self.x(-0.1), [[1, 1], [1, 1]])
        assert_equal(self.x(0.0), [[11, 21], [31, 41]])
        assert_equal(self.x(0.5), [[11, 21], [31, 41]])
        assert_equal(self.x(0.999), [[11, 21], [31, 41]])
        assert_equal(self.x(1.0), [[101 + 1j, 201 + 2j], [301 + 3j, 401 + 4j]])
        assert_equal(self.x(1.999), [[101 + 1j, 201 + 2j], [301 + 3j, 401 + 4j]])
        assert_equal(self.x(3.0), [[1 + 1j, 1 + 2j], [1 + 3j, 1 + 4j]])
        assert_equal(self.x(5.0), [[1, 1], [1, 1]])

    def test_view(self):
        x = self.x.view(1, 2, 2)
        assert_equal(x(0.0), [[[11, 21], [31, 41]]])

    def test_adjoint(self):
        x = self.x.adjoint()
        assert_equal(x(1.0), [[101 - 1j, 301 - 3j], [201 - 2j, 401 - 4j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(1.0), [[-101 - 1j, -201 - 2j], [-301 - 3j, -401 - 4j]])

    def test_mul(self):
        res = [[202 + 2j, 402 + 4j], [602 + 6j, 802 + 8j]]

        # test type `Number`
        x = self.x * 2
        assert_equal(x(1.0), res)

        # test type `Tensor`
        x = self.x * torch.tensor([2])
        assert_equal(x(1.0), res)

    def test_rmul(self):
        res = [[202 + 2j, 402 + 4j], [602 + 6j, 802 + 8j]]

        # test type `Number`
        x = 2 * self.x
        assert_equal(x(1.0), res)

        # test type `Tensor`
        x = torch.tensor([2]) * self.x
        assert_equal(x(1.0), res)

    def test_add(self):
        tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)

        # test type `Tensor`
        x = self.x + tensor
        assert isinstance(x, PWCTimeTensor)
        assert_equal(x(1.0), [[102 + 1j, 202 + 2j], [302 + 3j, 402 + 4j]])

        # test type `PWCTimeTensor`
        x = self.x + self.x
        assert isinstance(x, PWCTimeTensor)
        assert_equal(x(1.0), [[202 + 2j, 402 + 4j], [602 + 6j, 802 + 8j]])

    def test_radd(self):
        tensor = torch.tensor([[1, 1], [1, 1]], dtype=torch.complex64)

        # test type `Tensor`
        x = tensor + self.x
        assert isinstance(x, PWCTimeTensor)
        assert_equal(x(1.0), [[102 + 1j, 202 + 2j], [302 + 3j, 402 + 4j]])
