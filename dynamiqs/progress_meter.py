import diffrax as dx
import equinox as eqx

__all__ = [
    'AbstractProgressMeter',
    'NoProgressMeter',
    'TextProgressMeter',
    'TqdmProgressMeter',
]


class AbstractProgressMeter(eqx.Module):
    def into_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.AbstractProgressMeter()


class NoProgressMeter(AbstractProgressMeter):
    def into_diffrax(self) -> dx.NoProgressMeter:
        return dx.NoProgressMeter()


class TextProgressMeter(AbstractProgressMeter):
    def into_diffrax(self) -> dx.TextProgressMeter:
        return dx.TextProgressMeter()


class TqdmProgressMeter(AbstractProgressMeter):
    def into_diffrax(self) -> dx.TqdmProgressMeter:
        return dx.TqdmProgressMeter()
