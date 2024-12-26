from __future__ import annotations

from ._utils import obj_type_str
from .qarrays import QArray
from .time_qarray import TimeQArray

__all__ = ['hc']


class HermitianConjugate:
    def __add__(self, _):  # noqa
        raise TypeError(
            'The symbol `dq.hq` can only be right-added to a qarray or a time-qarray.'
        )

    def __radd__(self, y: QArray | TimeQArray) -> QArray | TimeQArray:
        if isinstance(y, (QArray, TimeQArray)):
            return y + y.dag()
        else:
            raise TypeError(
                f'The symbol `dq.hq` can only be right-added to a qarray or a '
                f'time-qarray, but it was added to a {obj_type_str(y)}.'
            )


hc = HermitianConjugate()
