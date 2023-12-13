import pytest
import torch
from torch import Tensor

from dynamiqs.time_tensor import CallableTimeTensor, ConstantTimeTensor


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
