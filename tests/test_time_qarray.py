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


@pytest.mark.run(order=TEST_SHORT)
class TestTimeQArrayShift:
    def test_shift_constant(self):
        # test clipped operator
        qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        operator = constant(qarray).clip(0.25, 1.25)
        t_shift = 0.5
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, ConstantTimeQArray)

        assert shifted.tstart == operator.tstart + t_shift
        assert shifted.tend == operator.tend + t_shift

        t_inside = 1.0
        t_edge = 0.5
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

        # test unclipped operator
        operator = constant(qarray)
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, ConstantTimeQArray)
        assert shifted.tstart is None
        assert shifted.tend is None
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

    def test_shift_pwc(self):
        # test clipped operator
        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        operator = pwc(times, values, qarray).clip(0.25, 1.25)
        t_shift = 0.5
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, PWCTimeQArray)

        assert shifted.tstart == operator.tstart + t_shift
        assert shifted.tend == operator.tend + t_shift

        t_inside = 1.0
        t_edge = 0.5
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

        # test unclipped operator
        operator = pwc(times, values, qarray)
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, PWCTimeQArray)
        assert shifted.tstart is None
        assert shifted.tend is None
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

    def test_shift_modulated(self):
        # test clipped operator
        f = lambda t: 1.0 + 2.0 * t
        qarray = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        operator = modulated(f, qarray).clip(0.25, 1.25)
        t_shift = 0.5
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, ModulatedTimeQArray)

        assert shifted.tstart == operator.tstart + t_shift
        assert shifted.tend == operator.tend + t_shift

        t_inside = 1.0
        t_edge = 0.5
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

        # test unclipped operator
        operator = modulated(f, qarray)
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, ModulatedTimeQArray)
        assert shifted.tstart is None
        assert shifted.tend is None
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

    def test_shift_callable(self):
        # test clipped operator
        f = lambda t: asqarray(jnp.array([[t, 0.0], [0.0, 1.0 + t]]))
        operator = timecallable(f).clip(0.25, 1.25)
        t_shift = 0.5
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, CallableTimeQArray)

        assert shifted.tstart == operator.tstart + t_shift
        assert shifted.tend == operator.tend + t_shift

        t_inside = 1.0
        t_edge = 0.5
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

        # test unclipped operator
        operator = timecallable(f)
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, CallableTimeQArray)
        assert shifted.tstart is None
        assert shifted.tend is None
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

    def test_shift_summed(self):
        # test clipped operator
        f = lambda t: 1.0 + 2.0 * t
        qarray = jnp.array([[1.0, 2.0], [3.0, 4.0]])
        modulated_op = modulated(f, qarray).clip(0.25, 1.25)

        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        pwc_qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        pwc_op = pwc(times, values, pwc_qarray).clip(0.25, 1.25)

        operator = modulated_op + pwc_op
        t_shift = 0.5
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, SummedTimeQArray)

        t_inside = 1.0
        t_edge = 0.5
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

        # test unclipped operator
        modulated_op = modulated(f, qarray)
        pwc_op = pwc(times, values, pwc_qarray)
        operator = modulated_op + pwc_op
        shifted = operator.shift(t_shift)
        assert isinstance(shifted, SummedTimeQArray)
        assert shifted.tstart is None
        assert shifted.tend is None
        assert_equal(shifted(t_inside), operator(t_inside - t_shift))
        assert_equal(shifted(t_edge), operator(t_edge - t_shift))

    def test_shift_vmap_tshift(self):
        times = jnp.array([0.0, 1.0, 2.0])
        values = jnp.array([1.0, -1.0])
        qarray = jnp.array([[1.0, 0.0], [0.0, -1.0]])
        operator = pwc(times, values, qarray)

        t = 0.8
        t_shifts = jnp.array([0.0, 0.4, -0.6])

        out = jax.vmap(lambda s: operator.shift(s)(t))(t_shifts)
        expected = jax.vmap(lambda s: operator(t - s))(t_shifts)
        assert out.shape == (t_shifts.shape[0], *operator(t).shape)
        assert_equal(out, expected)

    def test_shift_composes(self):
        f = lambda t: asqarray(jnp.array([[t, 0.0], [0.0, 1.0 - t]]))
        operator = timecallable(f).clip(0.0, 2.0)

        shifted_twice = operator.shift(0.25).shift(0.5)
        shifted_direct = operator.shift(0.75)

        assert shifted_twice.tstart == shifted_direct.tstart
        assert shifted_twice.tend == shifted_direct.tend

        t_inside = 1.0
        t_edge = 0.75
        assert_equal(shifted_twice(t_inside), shifted_direct(t_inside))
        assert_equal(shifted_twice(t_edge), shifted_direct(t_edge))
