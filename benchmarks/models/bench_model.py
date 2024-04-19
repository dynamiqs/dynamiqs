from __future__ import annotations

from abc import ABC, abstractmethod
from timeit import Timer
from typing import Literal

import jax.numpy as jnp
import qutip as qt
from jaxtyping import Array

import dynamiqs as dq


class BenchModel(ABC):
    def __init__(self, *args, **kwargs):
        self.init_qutip(*args, **kwargs)
        self.init_dynamiqs(*args, **kwargs)

    @abstractmethod
    def init_qutip(self, *args, **kwargs):
        """Initialize qutip arguments."""

    @abstractmethod
    def init_dynamiqs(self, *args, **kwargs):
        """Initialize dynamiqs arguments."""

    def benchmark(
        self,
        library: Literal['dynamiqs', 'qutip'],
        backend: Literal['cpu', 'gpu'],
        repeat: int = 5,
        tmin: float = 0.05,
    ) -> float:
        """Benchmark the model and returns the runtime in seconds."""
        if (library, backend) == ('dynamiqs', 'cpu'):
            dq.set_device('cpu')
            fn = lambda: self.fn_dynamiqs(**self.kwargs_dynamiqs)
        elif (library, backend) == ('dynamiqs', 'gpu'):
            dq.set_device('gpu')
            fn = lambda: self.fn_dynamiqs(**self.kwargs_dynamiqs)
        elif (library, backend) == ('qutip', 'cpu'):
            fn = lambda: self.fn_qutip(**self.kwargs_qutip)
        else:
            raise ValueError(
                f'Invalid combination of library and backend: {library}, {backend}'
            )

        # run once for jit
        if library == 'dynamiqs':
            fn()

        return _auto_timeit(fn, repeat=repeat, tmin=tmin)

    def check_args(self):
        """Check that all input arguments are equal between qutip and dynamiqs."""
        if 'H' in self.kwargs_qutip and 'H' in self.kwargs_dynamiqs:
            self._check_operators_equal(
                self.kwargs_qutip['H'], self.kwargs_dynamiqs['H']
            )
        if 'psi0' in self.kwargs_qutip and 'psi0' in self.kwargs_dynamiqs:
            self._check_operators_equal(
                self.kwargs_qutip['psi0'], self.kwargs_dynamiqs['psi0']
            )
        if 'rho0' in self.kwargs_qutip and 'rho0' in self.kwargs_dynamiqs:
            self._check_operators_equal(
                self.kwargs_qutip['rho0'], self.kwargs_dynamiqs['rho0']
            )
        if 'tlist' in self.kwargs_qutip and 'tsave' in self.kwargs_dynamiqs:
            assert jnp.allclose(
                self.kwargs_qutip['tlist'], self.kwargs_dynamiqs['tsave']
            )
        if 'c_ops' in self.kwargs_qutip and 'jump_ops' in self.kwargs_dynamiqs:
            for c_op, jump_op in zip(
                self.kwargs_qutip['c_ops'], self.kwargs_dynamiqs['jump_ops']
            ):
                self._check_operators_equal(c_op, jump_op)

    def _check_operators_equal(
        self, op_qutip: qt.Qobj | list, op_dynamiqs: Array | dq.TimeArray
    ):
        if isinstance(op_qutip, qt.Qobj) and isinstance(op_dynamiqs, Array):
            assert jnp.allclose(op_qutip.full(), op_dynamiqs)
        elif isinstance(op_qutip, list) and isinstance(op_dynamiqs, dq.TimeArray):
            # check that operators are equal when evaluated at t=0.0 and t=0.23113
            op_qutip = qt.QobjEvo(op_qutip)
            assert jnp.allclose(op_qutip(0.0).full(), op_dynamiqs(0.0))
            assert jnp.allclose(op_qutip(0.23113).full(), op_dynamiqs(0.23113))

    def convert_data_format(self, data_format: Literal['dense', 'csr', 'dia']):
        """Convert qutip data format to `data_format`."""
        if 'H' in self.kwargs_qutip:
            self.kwargs_qutip['H'] = self._convert_operator(
                self.kwargs_qutip['H'], data_format
            )
        if 'psi0' in self.kwargs_qutip:
            self.kwargs_qutip['psi0'] = self._convert_operator(
                self.kwargs_qutip['psi0'], data_format
            )
        if 'rho0' in self.kwargs_qutip:
            self.kwargs_qutip['rho0'] = self._convert_operator(
                self.kwargs_qutip['rho0'], data_format
            )
        if 'c_ops' in self.kwargs_qutip:
            self.kwargs_qutip['c_ops'] = [
                self._convert_operator(c_op, data_format)
                for c_op in self.kwargs_qutip['c_ops']
            ]

    def _convert_operator(
        self, op: qt.Qobj | list | callable, data_format: Literal['dense', 'csr', 'dia']
    ) -> qt.Qobj | list:
        """Convert a qutip operator to a dynamiqs operator."""
        if isinstance(op, qt.Qobj):
            return op.to(data_format)
        elif isinstance(op, list):
            return [self._convert_operator(subop, data_format) for subop in op]
        elif callable(op):
            return op
        else:
            raise TypeError(f'Invalid type for qutip operator: {type(op)}')


def _auto_timeit(fn: callable, repeat: int = 5, tmin: float = 0.05) -> float:
    """Run timeit for at least `tmin` seconds to get a reliable runtime benchmark. This
    is similar behavior as the `%timeit` ipython magic.
    """
    timer = Timer(fn)
    n = 1
    t = timer.timeit(number=n)
    while t < tmin:
        n *= 10
        t = timer.timeit(number=n)

    for _ in range(repeat - 1):
        t += timer.timeit(number=n)

    return t / n / repeat
