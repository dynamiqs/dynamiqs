from jaxtyping import ArrayLike
from qutip import Qobj

from .qarrays import QArray, QArrayLike, asqarray


class HermitianConjugate:
    def __radd__(self, y: QArrayLike) -> QArrayLike:
        if isinstance(y, Qobj):
            y = asqarray(y)

        if isinstance(y, QArray):
            return y + y.dag()
        elif isinstance(y, ArrayLike):
            return y + y.mT.conj()
        else:
            raise TypeError(
                f'`dq.hq` can only be right-added '
                f'to a QArrayLike, {y.__class__} given.'
            )


hc = HermitianConjugate()
