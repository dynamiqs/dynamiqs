import diffrax as dx
import equinox as eqx

__all__ = [
    'AbstractProgressMeter',
    'NoProgressMeter',
    'TextProgressMeter',
    'TqdmProgressMeter',
]


class AbstractProgressMeter(eqx.Module):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.AbstractProgressMeter()


class NoProgressMeter(AbstractProgressMeter):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.NoProgressMeter()


class TextProgressMeter(AbstractProgressMeter):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.TextProgressMeter()


class TqdmProgressMeter(AbstractProgressMeter):
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.TqdmProgressMeter()
