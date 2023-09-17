from __future__ import annotations

from datetime import datetime, timedelta

from torch import Tensor

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
        exp_save: Tensor,
        meas_save: Tensor | None = None,
    ):
        self.options = options
        self.ysave = ysave
        self.tsave = tsave
        self.exp_save = exp_save
        self.meas_save = meas_save
        self.start_time: float | None = None
        self.end_time: float | None = None

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
    def solver_str(self) -> str:
        return self.options.__class__.__name__

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

    def __str__(self):
        tmp = (
            '==== Result ====\n'
            f'Method       : {self.solver_str}\n'
            f'Start        : {self.start_datetime.strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'End          : {self.end_datetime.strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Total time   : {self.total_time.total_seconds():.2f} s\n'
            f'states       : {tensor_str(self.states)}'
        )
        if self.expects is not None:
            tmp += f'\nexpects      : {tensor_str(self.expects)}'
        if self.measurements is not None:
            tmp += f'\nmeasurements : {tensor_str(self.measurements)}'
        return tmp

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str) -> Result:
        raise NotImplementedError
