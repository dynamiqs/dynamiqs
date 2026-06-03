from abc import abstractmethod

import diffrax as dx
import equinox as eqx
from tqdm import tqdm

__all__ = ['NoProgressMeter', 'TextProgressMeter', 'TqdmProgressMeter']


class AbstractProgressMeter(eqx.Module):
    """Abstract base class for progress meters.

    Implement this class to define a custom progress meter. Pass an instance to any
    solver via the `progress_meter` argument.
    """

    @abstractmethod
    def to_diffrax(self) -> dx.AbstractProgressMeter:
        pass


class NoProgressMeter(AbstractProgressMeter):
    """Progress meter that produces no output.

    Pass to any solver via `progress_meter=dq.NoProgressMeter()` (or equivalently
    `progress_meter=False`) to suppress all progress output.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass

    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.NoProgressMeter()


class TextProgressMeter(AbstractProgressMeter):
    """Plain-text progress meter that prints to standard output.

    Prints a simple text line updated in place during the integration. Useful in
    environments where tqdm is not available or not desired.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass

    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return dx.TextProgressMeter()


def _format_duration(duration_s: float) -> str:
    hours, remainder = divmod(duration_s, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = seconds * 1000

    if hours > 0:  # e.g. 34h07m37s
        return f'{hours:02.0f}h{minutes:02.0f}m{seconds:02.0f}s'
    elif minutes > 0:  # e.g. 07m37s
        return f'{minutes:02.0f}m{seconds:02.0f}s'
    elif seconds >= 1:  # e.g. 56.79s
        return f'{seconds:.2f}s'
    else:  # e.g. 789.12ms
        return f'{milliseconds:.2f}ms'


class _TqdmCustom(tqdm):
    @property
    def format_dict(self):  # noqa: ANN202
        d = super().format_dict

        # compute and format remaining
        n, total, rate, elapsed = (d[k] for k in ['n', 'total', 'rate', 'elapsed'])
        if elapsed == 0:
            remaining_custom = '?'
        else:
            remaining = (total - n) / rate if rate and total else 0
            remaining_custom = _format_duration(remaining)

        # format elapsed
        elapsed_custom = _format_duration(elapsed)

        # update dict
        d.update(elapsed_custom=elapsed_custom, remaining_custom=remaining_custom)

        return d


class _DiffraxTqdmProgressMeter(dx.TqdmProgressMeter):
    @staticmethod
    def _init_bar() -> tqdm:
        bar_format = (
            '|{bar}| {percentage:5.1f}% ◆ elapsed {elapsed_custom} '
            '◆ remaining {remaining_custom}'
        )
        return _TqdmCustom(total=100, unit='%', bar_format=bar_format)


class TqdmProgressMeter(AbstractProgressMeter):
    """tqdm progress bar (default progress meter).

    Displays a rich progress bar powered by [tqdm](https://github.com/tqdm/tqdm),
    showing percentage completion, elapsed time, and estimated remaining time.

    This is the default progress meter used by all solvers when `progress_meter=None`
    (unless overridden globally with
    [`dq.set_progress_meter()`][dynamiqs.set_progress_meter]).

    Pass `progress_meter=True` or `progress_meter=dq.TqdmProgressMeter()` explicitly
    to force it.
    """

    # dummy init to have the signature in the documentation
    def __init__(self):
        pass

    def to_diffrax(self) -> dx.AbstractProgressMeter:
        return _DiffraxTqdmProgressMeter()
