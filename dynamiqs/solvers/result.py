from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

from torch import Tensor

from ..gradient import Gradient
from ..solver import Solver
from .options import Options


def memory_bytes(x: Tensor) -> int:
    return x.element_size() * x.numel()


def memory_str(x: Tensor) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.2f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.2f} Mb'
    else:
        return f'{mem / 1024**3:.2f} Gb'


def tensor_str(x: Tensor) -> str:
    return f'Tensor {tuple(x.shape)} | {memory_str(x)}'


class Result:
    def __init__(
        self,
        options: Options,
        ysave: Tensor,
        tsave: Tensor,
        exp_save: Tensor | None,
        meas_save: Tensor | None = None,
        tmeas: Tensor | None = None,
    ):
        self._options = options
        self.ysave = ysave
        self.tsave = tsave
        self.exp_save = exp_save
        self.meas_save = meas_save
        self.tmeas = tmeas
        self.start_time: float | None = None
        self.end_time: float | None = None

    @property
    def solver(self) -> Solver:
        return self._options.solver

    @property
    def gradient(self) -> Gradient:
        return self._options.gradient

    @property
    def options(self) -> dict[str, Any]:
        return self._options.options

    @property
    def states(self) -> Tensor:
        # alias for ysave
        return self.ysave

    @property
    def times(self) -> Tensor:
        # alias for tsave
        return self.tsave

    @property
    def expects(self) -> Tensor | None:
        # alias for exp_save
        return self.exp_save

    @property
    def measurements(self) -> Tensor | None:
        # alias for meas_save
        return self.meas_save

    @property
    def start_datetime(self) -> datetime | None:
        if self.start_time is None:
            return None
        return datetime.fromtimestamp(self.start_time)

    @property
    def end_datetime(self) -> datetime | None:
        if self.end_time is None:
            return None
        return datetime.fromtimestamp(self.end_time)

    @property
    def total_time(self) -> timedelta | None:
        if self.start_datetime is None or self.end_datetime is None:
            return None
        return self.end_datetime - self.start_datetime

    def __str__(self) -> str:
        parts = {
            'Solver': type(self.solver).__name__,
            'Gradient': type(self.gradient).__name__ if self.gradient else None,
            'Start': self.start_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'End': self.end_datetime.strftime('%Y-%m-%d %H:%M:%S'),
            'Total time': f'{self.total_time.total_seconds():.2f} s',
            'States': tensor_str(self.states),
            'Expects': tensor_str(self.expects) if self.expects else None,
            'Measurements': (
                tensor_str(self.measurements) if self.measurements else None
            ),
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        padding = max(len(k) for k in parts.keys()) + 1
        parts_str = '\n'.join(f'{k:<{padding}}: {v}' for k, v in parts.items())
        return '==== Result ====\n' + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str) -> Result:
        raise NotImplementedError
