from ._utils import obj_type_str
from .qarrays import QArray

__all__ = ['hc']


class HermitianConjugate:
    def __radd__(self, y: QArray) -> QArray:
        if isinstance(y, QArray):
            return y + y.dag()
        else:
            raise TypeError(
                f'The symbol `dq.hq` can only be right-added to a qarray, but it was'
                f' added to a {obj_type_str(y)}.'
            )


hc = HermitianConjugate()
