from abc import abstractmethod

import diffrax as dx
import equinox as eqx
import tqdm

__all__ = [
    'AbstractProgressMeter',
    'NoProgressMeter',
    'TextProgressMeter',
    'TqdmProgressMeter',
]


class AbstractProgressMeter(eqx.Module):
    @abstractmethod
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        pass


class NoProgressMeter(AbstractProgressMeter):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.NoProgressMeter()


class TextProgressMeter(AbstractProgressMeter):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.TextProgressMeter()


class _DiffraxTqdmProgressMeter(dx.TqdmProgressMeter):
    @staticmethod
    def _init_bar() -> tqdm.tqdm:
        bar_format = (
            '|{bar}| {percentage:5.1f}% ◆ total {elapsed} ◆ remaining {remaining}'
        )
        return tqdm.tqdm(total=100, unit='%', bar_format=bar_format)


class TqdmProgressMeter(AbstractProgressMeter):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return _DiffraxTqdmProgressMeter()
