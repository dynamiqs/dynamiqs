import timeit

import jax
import jax.numpy as jnp
import pytest

from dynamiqs import QArray, asqarray
from dynamiqs.time_qarray import (
    CallableTimeQArray,
    ConstantTimeQArray,
    ModulatedTimeQArray,
    PWCTimeQArray,
    SummedTimeQArray,
    constant,
    modulated,
    pwc,
    timecallable,
)

from .order import TEST_SHORT


def assert_equal(x, y):
    if isinstance(x, QArray):
        x = x.to_jax()
    if isinstance(y, QArray):
        y = y.to_jax()
    assert jnp.array_equal(x, y)


@pytest.mark.run(order=TEST_SHORT)
class TestConstantTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        self.x = constant(jnp.arange(4).reshape(2, 2))

    @pytest.mark.skip('broken test')
    def test_jit(self):
        # we don't test speed here, just that it works
        x = jax.jit(self.x)
        assert_equal(x(0.0), [0, 1, 2, 3])

    def test_call(self):
        assert_equal(self.x(0.0), [[0, 1], [2, 3]])
        assert_equal(self.x(1.0), [[0, 1], [2, 3]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[0, 1], [2, 3]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(x(0.0), [[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    def test_conj(self):
        x = ConstantTimeQArray(jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]]))
        x = x.conj()
        assert_equal(x(0.0), [[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-0, -1], [-2, -3]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[0, 2], [4, 6]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[0, 2], [4, 6]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + jnp.ones_like(self.x)
        assert isinstance(x, ConstantTimeQArray)
        assert_equal(x(0.0), [[1, 2], [3, 4]])

        # test type `ConstantTimeQArray`
        x = self.x + self.x
        assert isinstance(x, ConstantTimeQArray)
        assert_equal(x(0.0), [[0, 2], [4, 6]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, ConstantTimeQArray)
        assert_equal(x(0.0), [[1, 2], [3, 4]])

    def test_clip(self):
        # clip the array to [0.25, 1.25]
        x_clipped = self.x.clip(0.25, 1.25)
        assert isinstance(x_clipped, ConstantTimeQArray)
        assert x_clipped.tstart == 0.25
        assert x_clipped.tend == 1.25

        # check inside window
        assert_equal(x_clipped(0.5), [[0, 1], [2, 3]])
        # check outside window
        assert_equal(x_clipped(0.0), [[0, 0], [0, 0]])
        assert_equal(x_clipped(1.5), [[0, 0], [0, 0]])

    def test_shift(self):
        # test unclipped operator
        t_shift = 0.5
        x_shifted = self.x.shift(t_shift)
        assert isinstance(x_shifted, ConstantTimeQArray)
        assert x_shifted.tstart is None
        assert x_shifted.tend is None

        # check values (should be invariant for constant, but we test the API)
        # x(t) == x_original(t - shift)
        assert_equal(x_shifted(1.0), self.x(1.0 - t_shift))

        # test clipped operator
        x_clipped = self.x.clip(0.25, 1.25)
        x_clipped_shifted = x_clipped.shift(t_shift)
        assert isinstance(x_clipped_shifted, ConstantTimeQArray)
        assert x_clipped_shifted.tstart == x_clipped.tstart + t_shift
        assert x_clipped_shifted.tend == x_clipped.tend + t_shift

        t_inside = 1.0  # inside shifted window [0.75, 1.75]
        t_outside = 0.0  # outside shifted window
        assert_equal(x_clipped_shifted(t_inside), x_clipped(t_inside - t_shift))
        assert_equal(x_clipped_shifted(t_outside), [[0, 0], [0, 0]])


@pytest.mark.run(order=TEST_SHORT)
class TestCallableTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        f = lambda t: t * asqarray(jnp.arange(4).reshape(2, 2))
        self.x = timecallable(f)

    @pytest.mark.skip('broken test')
    def test_jit(self):
        x = jax.jit(self.x)
        assert_equal(x(0.0), [0, 0])
        assert_equal(x(1.0), [1, 2])

        t1 = timeit.timeit(lambda: x(1.0), number=1000)
        t2 = timeit.timeit(lambda: self.x(1.0), number=1000)
        assert t1 < t2

    def test_call(self):
        assert_equal(self.x(0.0), [[0, 0], [0, 0]])
        assert_equal(self.x(1.0), [[0, 1], [2, 3]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[0, 0], [0, 0]]])
        assert_equal(x(1.0), [[[0, 1], [2, 3]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(x(0.0), [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        assert_equal(x(1.0), [[[0, 1], [2, 3]], [[0, 1], [2, 3]]])

    def test_conj(self):
        f = lambda t: t * jnp.array([[1 + 1j, 2 + 2j], [3 + 3j, 4 + 4j]])
        x = timecallable(f)
        x = x.conj()
        assert_equal(x(1.0), [[1 - 1j, 2 - 2j], [3 - 3j, 4 - 4j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, -1], [-2, -3]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + constant(jnp.ones_like(self.x))
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1, 1], [1, 1]])
        assert_equal(x(1.0), [[1, 2], [3, 4]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.arange(4).reshape(2, 2))
        x = self.x + y
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[0, 1], [2, 3]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

        # test type `CallableTimeQArray` (skipped for now)
        x = self.x + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[0, 0], [0, 0]])
        assert_equal(x(1.0), [[0, 2], [4, 6]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1, 1], [1, 1]])
        assert_equal(x(1.0), [[1.0, 2.0], [3.0, 4.0]])

        # test type `ConstantTimeQArray`
        x = constant(jnp.ones_like(x)) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1, 1], [1, 1]])
        assert_equal(x(1.0), [[1, 2], [3, 4]])

    def test_clip(self):
        f = lambda t: asqarray(jnp.array([[t, 0.0], [0.0, 1.0 + t]]))
        x = timecallable(f)
        x_clipped = x.clip(0.25, 1.25)

        assert isinstance(x_clipped, CallableTimeQArray)
        assert x_clipped.tstart == 0.25
        assert x_clipped.tend == 1.25

        # inside window
        assert_equal(x_clipped(0.5), x(0.5))
        # outside window
        assert_equal(x_clipped(0.0), [[0, 0], [0, 0]])

    def test_shift(self):
        f = lambda t: asqarray(jnp.array([[t, 0.0], [0.0, 1.0 + t]]))
        x = timecallable(f)
        t_shift = 0.5

        # test unclipped operator
        x_shifted = x.shift(t_shift)
        assert isinstance(x_shifted, CallableTimeQArray)
        assert x_shifted.tstart is None
        assert x_shifted.tend is None
        # x(t) == x_original(t - shift)
        assert_equal(x_shifted(1.0), x(1.0 - t_shift))

        # test clipped operator
        x_clipped = x.clip(0.25, 1.25)
        x_clipped_shifted = x_clipped.shift(t_shift)
        assert isinstance(x_clipped_shifted, CallableTimeQArray)
        assert x_clipped_shifted.tstart == x_clipped.tstart + t_shift
        assert x_clipped_shifted.tend == x_clipped.tend + t_shift

        t_inside = 1.0
        assert_equal(x_clipped_shifted(t_inside), x_clipped(t_inside - t_shift))
        assert_equal(x_clipped_shifted(0.0), [[0, 0], [0, 0]])

    def test_shift_composes(self):
        f = lambda t: asqarray(jnp.array([[t, 0.0], [0.0, 1.0 - t]]))
        x = timecallable(f).clip(0.0, 2.0)

        x_shifted_twice = x.shift(0.25).shift(0.5)
        x_shifted_direct = x.shift(0.75)

        assert x_shifted_twice.tstart == x_shifted_direct.tstart
        assert x_shifted_twice.tend == x_shifted_direct.tend

        t_inside = 1.0
        assert_equal(x_shifted_twice(t_inside), x_shifted_direct(t_inside))


@pytest.mark.run(order=TEST_SHORT)
class TestPWCTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        times = jnp.array([0, 1, 2, 3])
        values = jnp.array([1, 10 + 1j, 100])
        array = jnp.array([[1, 2], [3, 4]])

        self.x = pwc(times, values, array)  # shape at t: (2, 2)

    @pytest.mark.skip('broken test')
    def test_jit(self):
        x = jax.jit(self.x)
        assert_equal(x(-0.1), [[0, 0], [0, 0]])
        assert_equal(x(0.0), [[1, 2], [3, 4]])
        assert_equal(x(1.0), [[10 + 1j, 20 + 1j], [30 + 1j, 40 + 1j]])

        t1 = timeit.timeit(lambda: x(1.0), number=1000)
        t2 = timeit.timeit(lambda: self.x(1.0), number=1000)
        assert t1 < t2

    def test_call(self):
        assert_equal(self.x(-0.1), [[0, 0], [0, 0]])
        assert_equal(self.x(0.0), [[1, 2], [3, 4]])
        assert_equal(self.x(0.5), [[1, 2], [3, 4]])
        assert_equal(self.x(0.999), [[1, 2], [3, 4]])
        assert_equal(self.x(1.0), [[10 + 1j, 20 + 2j], [30 + 3j, 40 + 4j]])
        assert_equal(self.x(1.999), [[10 + 1j, 20 + 2j], [30 + 3j, 40 + 4j]])
        assert_equal(self.x(3.0), [[0, 0], [0, 0]])
        assert_equal(self.x(5.0), [[0, 0], [0, 0]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(-0.1), [[[0, 0], [0, 0]]])
        assert_equal(x(0.0), [[[1, 2], [3, 4]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(x(-0.1), [[[0, 0], [0, 0]], [[0, 0], [0, 0]]])
        assert_equal(x(0.0), [[[1, 2], [3, 4]], [[1, 2], [3, 4]]])

    def test_conj(self):
        x = self.x.conj()
        assert_equal(x(1.0), [[10 - 1j, 20 - 2j], [30 - 3j, 40 - 4j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1, -2], [-3, -4]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + jnp.ones_like(self.x)
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = self.x + y
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `PWCTimeQArray`
        x = self.x + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2, 4], [6, 8]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2, 3], [4, 5]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = y + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(-0.1), [[1, 1], [1, 1]])
        assert_equal(x(0.0), [[2, 3], [4, 5]])

    def test_clip(self):
        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        x = pwc(times, values, qarray)

        # clip [0.25, 1.25]
        x_clipped = x.clip(0.25, 1.25)
        assert isinstance(x_clipped, PWCTimeQArray)
        assert x_clipped.tstart == 0.25
        assert x_clipped.tend == 1.25

        # check values
        assert_equal(x_clipped(0.5), x(0.5))
        assert_equal(x_clipped(0.0), [[0, 0], [0, 0]])

    def test_shift(self):
        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        x = pwc(times, values, qarray)
        t_shift = 0.5

        # test unclipped operator
        x_shifted = x.shift(t_shift)
        assert isinstance(x_shifted, PWCTimeQArray)
        assert x_shifted.tstart is None
        assert x_shifted.tend is None

        # check values (PWC is defined on [0, 2], so shifted is defined on [0.5, 2.5])
        assert_equal(x_shifted(1.0), x(1.0 - t_shift))  # Inside
        assert_equal(x_shifted(0.0), [[0, 0], [0, 0]])  # Outside (left)

        # test clipped operator
        x_clipped = x.clip(0.25, 1.25)
        x_clipped_shifted = x_clipped.shift(t_shift)
        assert isinstance(x_clipped_shifted, PWCTimeQArray)
        assert x_clipped_shifted.tstart == x_clipped.tstart + t_shift
        assert x_clipped_shifted.tend == x_clipped.tend + t_shift

        t_inside = 1.0
        assert_equal(x_clipped_shifted(t_inside), x_clipped(t_inside - t_shift))

    def test_shift_vmap_tshift(self):
        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        x = pwc(times, values, qarray)

        t = 0.8
        t_shifts = jnp.array([0.0, 0.4, -0.6])

        out = jax.vmap(lambda s: x.shift(s)(t))(t_shifts)
        expected = jax.vmap(lambda s: x(t - s))(t_shifts)
        assert out.shape == (t_shifts.shape[0], *x(t).shape)
        assert_equal(out, expected)


@pytest.mark.run(order=TEST_SHORT)
class TestModulatedTimeQArray:
    @pytest.fixture(autouse=True)
    def _setup(self):
        one = jnp.array(1.0)
        eps = lambda t: (0.5 * t + 1.0j) * one
        array = jnp.array([[1, 2], [3, 4]])

        self.x = modulated(eps, array)

    def test_call(self):
        assert_equal(self.x(0.0), [[1.0j, 2.0j], [3.0j, 4.0j]])
        assert_equal(self.x(2.0), [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]])

    def test_reshape(self):
        x = self.x.reshape(1, 2, 2)
        assert_equal(x(0.0), [[[1.0j, 2.0j], [3.0j, 4.0j]]])
        assert_equal(x(2.0), [[[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]]])

    def test_broadcast(self):
        x = self.x.broadcast_to(2, 2, 2)
        assert_equal(
            x(0.0), [[[1.0j, 2.0j], [3.0j, 4.0j]], [[1.0j, 2.0j], [3.0j, 4.0j]]]
        )
        assert_equal(
            x(2.0),
            [
                [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]],
                [[1.0 + 1.0j, 2.0 + 2.0j], [3.0 + 3.0j, 4.0 + 4.0j]],
            ],
        )

    def test_conj(self):
        x = self.x.conj()
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_neg(self):
        x = -self.x
        assert_equal(x(0.0), [[-1.0j, -2.0j], [-3.0j, -4.0j]])

    def test_mul(self):
        x = self.x * 2
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_rmul(self):
        x = 2 * self.x
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_add(self):
        # test type `ArrayLike`
        x = self.x + jnp.ones_like(self.x)
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = self.x + y
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ModulatedTimeQArray`
        x = self.x + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[2.0j, 4.0j], [6.0j, 8.0j]])

    def test_radd(self):
        # test type `ArrayLike`
        x = jnp.ones_like(self.x) + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

        # test type `ConstantTimeQArray`
        y = constant(jnp.array([[1, 1], [1, 1]]))
        x = y + self.x
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(0.0), [[1.0 + 1.0j, 1.0 + 2.0j], [1.0 + 3.0j, 1.0 + 4.0j]])

    def test_clip(self):
        f = lambda t: 1.0 + 2.0 * t
        qarray = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x = modulated(f, qarray)

        x_clipped = x.clip(0.25, 1.25)
        assert isinstance(x_clipped, ModulatedTimeQArray)
        assert x_clipped.tstart == 0.25
        assert x_clipped.tend == 1.25

        assert_equal(x_clipped(0.5), x(0.5))
        assert_equal(x_clipped(0.0), [[0, 0], [0, 0]])

    def test_shift(self):
        f = lambda t: 1.0 + 2.0 * t
        qarray = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x = modulated(f, qarray)
        t_shift = 0.5

        # test unclipped operator
        x_shifted = x.shift(t_shift)
        assert isinstance(x_shifted, ModulatedTimeQArray)
        assert x_shifted.tstart is None
        assert x_shifted.tend is None
        assert_equal(x_shifted(1.0), x(1.0 - t_shift))

        # test clipped operator
        x_clipped = x.clip(0.25, 1.25)
        x_clipped_shifted = x_clipped.shift(t_shift)
        assert isinstance(x_clipped_shifted, ModulatedTimeQArray)
        assert x_clipped_shifted.tstart == x_clipped.tstart + t_shift
        assert x_clipped_shifted.tend == x_clipped.tend + t_shift
        assert_equal(x_clipped_shifted(1.0), x_clipped(1.0 - t_shift))


@pytest.mark.run(order=TEST_SHORT)
class TestSummedQArray:
    def test_add(self):
        f = lambda t: t * asqarray(jnp.arange(4).reshape(2, 2))
        f = timecallable(f)

        a = constant(jnp.ones_like(f))

        x = f + a
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(1.0), [[1, 2], [3, 4]])

        y = f + a
        assert isinstance(x, SummedTimeQArray)
        assert_equal(x(1.0), [[1, 2], [3, 4]])

        z = x + y
        assert isinstance(z, SummedTimeQArray)
        assert isinstance(z.timeqarrays[0], CallableTimeQArray)
        assert isinstance(z.timeqarrays[1], ConstantTimeQArray)
        assert isinstance(z.timeqarrays[2], CallableTimeQArray)
        assert isinstance(z.timeqarrays[3], ConstantTimeQArray)
        assert_equal(z(1.0), [[2, 4], [6, 8]])

    def test_clip(self):
        # create a modulated and pwc TimeQArray to sum
        f = lambda t: 1.0 + 2.0 * t
        qarray_modulated = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x_modulated = modulated(f, qarray_modulated)

        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray_pwc = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        x_pwc = pwc(times, values, qarray_pwc)

        # check that summing then clipping works
        x = x_modulated + x_pwc
        x_clipped = x.clip(0.25, 1.25)

        assert isinstance(x_clipped, SummedTimeQArray)
        # check inside window
        assert_equal(x_clipped(0.5), x(0.5))
        # check outside window
        assert_equal(x_clipped(0.0), [[0, 0], [0, 0]])

        # check that clipping then summing works
        x_modulated_clipped = x_modulated.clip(0.25, 1.25)
        x_pwc_clipped = x_pwc.clip(0.75, 1.75)
        x_clipped = x_modulated_clipped + x_pwc_clipped

        assert isinstance(x_clipped, SummedTimeQArray)
        # check inside both windows
        assert_equal(x_clipped(1.0), x_modulated_clipped(1.0) + x_pwc_clipped(1.0))
        # check inside modulated window
        assert_equal(x_clipped(0.5), x_modulated_clipped(0.5))
        # check inside pwc window
        assert_equal(x_clipped(1.5), x_pwc_clipped(1.5))
        # check outside both windows
        assert_equal(x_clipped(0.0), [[0, 0], [0, 0]])
        assert_equal(x_clipped(2.0), [[0, 0], [0, 0]])

    def test_shift(self):
        f = lambda t: 1.0 + 2.0 * t
        qarray_modulated = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        x_modulated = modulated(f, qarray_modulated)

        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray_pwc = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        x_pwc = pwc(times, values, qarray_pwc)

        x = x_modulated + x_pwc
        t_shift = 0.5

        # test unclipped
        x_shifted = x.shift(t_shift)
        assert isinstance(x_shifted, SummedTimeQArray)
        assert_equal(x_shifted(1.0), x(1.0 - t_shift))

        # test clipped
        x_clipped = x.clip(0.25, 1.25)
        x_clipped_shifted = x_clipped.shift(t_shift)
        assert isinstance(x_clipped_shifted, SummedTimeQArray)

        t_inside = 1.0
        assert_equal(x_clipped_shifted(t_inside), x_clipped(t_inside - t_shift))
        assert_equal(x_clipped_shifted(0.0), [[0, 0], [0, 0]])
