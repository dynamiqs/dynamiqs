from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike
from qutip import Qobj

from .dataarray import IndexType
from .layout import Layout
from .materialized_qarray import MaterializedQArray
from .qarray import QArray, QArrayLike

__all__ = []


class CompositeTerm(eqx.Module):
    r"""One separable term in a :class:`CompositeQArray`.

    Represents a single term of the form

    $$
        c \, A_0 \otimes A_1 \otimes \cdots \otimes A_{N-1}
    $$

    where $c$ is a scalar coefficient and each $A_k$ is a local operator acting
    on subsystem $k$.

    Attributes:
        operators: Local operators $(A_0, \ldots, A_{N-1})$, one per subsystem.
            Each is a square :class:`MaterializedQArray` of shape
            $(\ldots, d_k, d_k)$.
        coeff: Scalar coefficient $c$ multiplying the full tensor-product operator.
            Can be a Python scalar or a broadcastable JAX array for batched use.
            Defaults to $1$.
    """

    operators: tuple[MaterializedQArray, ...]
    coeff: ArrayLike = 1.0


class CompositeQArray(QArray):
    r"""Lazy sum of separable tensor-product operators.

    Represents an operator acting on a composite Hilbert space
    $\mathcal{H} = \mathcal{H}_0 \otimes \cdots \otimes \mathcal{H}_{N-1}$
    of total dimension $n = \prod_k d_k$, written as a sum of separable terms:

    $$
        H = \sum_{j} c_j \, A_{j,0} \otimes A_{j,1} \otimes \cdots \otimes A_{j,N-1}
    $$

    where each term $j$ is a :class:`CompositeTerm`.

    Storing the operator in this factored form — rather than materializing the full
    $n \times n$ Kronecker product — enables efficient matrix-vector products via
    per-subsystem contractions, avoiding the exponential memory cost of the dense
    representation.

    Note:
        ``dims`` is inherited from the abstract :class:`QArray` base class.

    Attributes:
        terms: Tuple of :class:`CompositeTerm` objects whose sum defines the
            operator. All terms must have ``len(operators) == len(dims)``.
    """

    terms: tuple[CompositeTerm, ...]

    # === Lifecycle ===

    def __check_init__(self):
        pass

    # === Properties ===

    @property
    def dtype(self) -> jnp.dtype:
        pass

    @property
    def layout(self) -> Layout:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    def mT(self) -> QArray:
        pass

    @property
    def ndim(self) -> int:
        pass

    # === Array methods ===

    def conj(self) -> QArray:
        pass

    def reshape(self, *shape: int) -> QArray:
        pass

    def _reshape_unchecked(self, *shape: int) -> QArray:
        pass

    def broadcast_to(self, *shape: int) -> QArray:
        pass

    def powm(self, n: int) -> QArray:
        pass

    def expm(self, *, max_squarings: int = 16) -> QArray:
        pass

    def norm(self, *, psd: bool = False) -> Array:
        pass

    def trace(self) -> Array:
        pass

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        pass

    def _eig(self) -> tuple[Array, QArray]:
        pass

    def _eigh(self) -> tuple[Array, Array]:
        pass

    def _eigvals(self) -> Array:
        pass

    def _eigvalsh(self) -> Array:
        pass

    def devices(self) -> set[Device]:
        pass

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
        pass

    def block_until_ready(self) -> QArray:
        pass

    # === Quantum methods ===

    def ptrace(self, *keep: int) -> QArray:
        pass

    # === Conversion methods ===

    def to_qutip(self) -> Qobj | list[Qobj]:
        pass

    def to_jax(self) -> Array:
        pass

    def to_numpy(self) -> np.ndarray:
        pass

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        pass

    def asdense(self) -> QArray:
        pass

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> QArray:
        pass

    # === Repr ===

    def __repr__(self) -> str:
        pass

    # === Arithmetic operations ===

    def __mul__(self, y: ArrayLike) -> QArray:
        pass

    def __add__(self, y: QArrayLike) -> QArray:
        pass

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        pass

    def __rmatmul__(self, y: QArrayLike) -> QArray:
        pass

    def __and__(self, y: QArray) -> QArray:
        pass

    # === Element-wise operations ===

    def addscalar(self, y: ArrayLike) -> QArray:
        pass

    def elmul(self, y: QArrayLike) -> QArray:
        pass

    def elpow(self, power: int) -> QArray:
        pass

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> QArray:
        pass
