import pytest
import torch
from torch import Tensor

from dynamiqs.time_tensor import CallableTimeTensor, ConstantTimeTensor, to_time_tensor


def assert_equal(xt: Tensor, y: list):
    assert torch.equal(xt, torch.tensor(y, dtype=torch.float32))


class TestConstantTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = to_time_tensor(torch.tensor([1, 2], dtype=torch.float32))
        self.tensor = torch.tensor([0, 1], dtype=torch.float32)

    def test_call(self):
        assert_equal(self.x(0.0), [1, 2])
        assert_equal(self.x(1.0), [1, 2])

    def test_view(self):
        x = self.x.view(1, 2)
        assert_equal(x(0.0), [[1, 2]])

    def test_adjoint(self):
        x = to_time_tensor(
            torch.tensor([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]), dtype=torch.complex64
        )
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
        x = self.x * self.tensor
        assert_equal(x(0.0), [0, 2])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [2, 4])

        # test type `Tensor`
        x = self.tensor * self.x
        assert_equal(x(0.0), [0, 2])


class TestCallableTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.x = to_time_tensor(lambda t: t * torch.tensor([1, 2]), dtype=torch.float32)
        self.tensor = torch.tensor([0, 1], dtype=torch.float32)

    def test_call(self):
        assert_equal(self.x(0.0), [0, 0])
        assert_equal(self.x(1.0), [1, 2])

    def test_view(self):
        x = self.x.view(1, 2)
        assert_equal(x(0.0), [[0, 0]])
        assert_equal(x(1.0), [[1, 2]])

    def test_adjoint(self):
        x = to_time_tensor(
            lambda t: t * torch.tensor([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]),
            dtype=torch.complex64,
        )
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
        x = self.x * self.tensor
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [0, 2])

    def test_rmul(self):
        # test type `Number`
        x = 2 * self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

        # test type `Tensor`
        x = self.tensor * self.x
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [0, 2])


class TestAddTimeTensor:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.xc = to_time_tensor(torch.tensor([1, 2], dtype=torch.float32))
        self.xf = to_time_tensor(
            lambda t: t * torch.tensor([1, 2]), dtype=torch.float32
        )
        self.tensor = torch.tensor([0, 1], dtype=torch.float32)

    def test_add(self):
        # === `ConstantTimeTensor`
        # test type `Tensor`
        x = self.xc + self.tensor
        assert isinstance(x, ConstantTimeTensor)
        assert_equal(x(0.0), [1, 3])

        # test type `ConstantTimeTensor`
        x = self.xc + self.xc
        assert isinstance(x, ConstantTimeTensor)
        assert_equal(x(0.0), [2, 4])

        # test type `CallableTimeTensor`
        x = self.xc + self.xf
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [1, 2])
        assert_equal(x(1.0), [2, 4])

        # === `CallableTimeTensor`
        # test type `Tensor`
        x = self.xf + self.tensor
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])

        # test type `ConstantTimeTensor`
        x = self.xf + self.xc
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [1, 2])
        assert_equal(x(1.0), [2, 4])

        # test type `CallableTimeTensor`
        x = self.xf + self.xf
        assert isinstance(x, CallableTimeTensor)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [2, 4])

    def test_radd(self):
        # === `ConstantTimeTensor`
        # test type `Tensor`
        x = self.tensor + self.xc
        assert_equal(x(0.0), [1, 3])

        # === `CallableTimeTensor`
        # test type `Tensor`
        x = self.tensor + self.xf
        assert_equal(x(0.0), [0, 1])
        assert_equal(x(1.0), [1, 3])
