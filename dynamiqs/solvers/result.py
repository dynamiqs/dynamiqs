from __future__ import annotations

from datetime import datetime, timedelta

from torch import Tensor

from ..options import Options


def memory_bytes(x: Tensor) -> int:
    return x.element_size() * x.numel()


def memory_str(x: Tensor) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.2f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.2f} Mb'

    return f'{mem / 1024**3:.2f} Gb'


def tensor_str(x: Tensor) -> str:
    return f'tensor {tuple(x.size())} [{memory_str(x)}]'


class Result:
    def __init__(self, options: Options, y_save: Tensor, exp_save: Tensor):
        self.options = options
        self.y_save = y_save
        self.exp_save = exp_save
        self.start_time: float | None = None
        self.end_time: float | None = None

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
            '==== Result ===\n'
            f'Options    : {self.solver_str}\n'
            f'Start      : {self.start_datetime.strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'End        : {self.end_datetime.strftime("%Y-%m-%d %H:%M:%S")}\n'
            f'Total time : {self.total_time.total_seconds():.2f} s\n'
            f'y_save     : {tensor_str(self.y_save)}'
        )
        if self.exp_save is not None:
            tmp += f'\nexp_save   : {tensor_str(self.exp_save)}'
        return tmp

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError

    def save(self, filename: str):
        raise NotImplementedError

    def load(self, filename: str) -> Result:
        raise NotImplementedError
