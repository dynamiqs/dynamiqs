from typing import get_args

from jaxtyping import ArrayLike

from .qarrays import QArray, QArrayLike

__all__ = ['hc']


class HermitianConjugate:
    def __radd__(self, y: QArrayLike) -> QArrayLike:
        if isinstance(y, QArray):
            return y + y.dag()
        elif isinstance(y, get_args(ArrayLike)):
            return y + y.mT.conj()
        else:
            raise TypeError(
                f'`dq.hq` can only be right-added '
                f'to a QArrayLike, {y.__class__} given.'
            )


hc = HermitianConjugate()
