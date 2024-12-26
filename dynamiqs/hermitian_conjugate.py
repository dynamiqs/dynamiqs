from .qarrays import QArray

__all__ = ['hc']


class HermitianConjugate:
    def __radd__(self, y: QArray) -> QArray:
        if isinstance(y, QArray):
            return y + y.dag()
        else:
            raise TypeError(
                f'`dq.hq` can only be right-added '
                f'to a QArrayLike, {y.__class__} given.'
            )


hc = HermitianConjugate()
