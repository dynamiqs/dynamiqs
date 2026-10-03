from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from functools import reduce
from math import prod
from typing import cast, overload

import equinox as eqx
import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import Array, Device
from jaxtyping import ArrayLike, PyTree
from qutip import Qobj

from .._utils import is_batched_scalar
from .dataarray import IndexType
from .layout import Layout, dia
from .materialized_qarray import MaterializedQArray
from .qarray import QArray, QArrayLike
from .sparsedia_dataarray import SparseDIADataArray

__all__ = []


def _devices(x: PyTree) -> set[Device]:
    # devices of every array leaf; tracers are skipped since they have no device
    return {
        device
        for leaf in jax.tree_util.tree_leaves(x)
        if isinstance(leaf, Array) and not isinstance(leaf, jax.core.Tracer)
        for device in leaf.devices()
    }


def _coeff_batch_shape(coeff: ArrayLike) -> tuple[int, ...]:
    # batch shape of a batched scalar, i.e. `()`, `(1,)` or `(*batch, 1, 1)`
    coeff = jnp.asarray(coeff)
    return coeff.shape[:-2] if coeff.ndim >= 2 else ()


class CompositeTerm(eqx.Module):
    r"""One separable term $c \, A_0 \otimes \cdots \otimes A_{N-1}$ in a
    :class:`CompositeQArray`.  Holds the bulk of the lazy logic; most ``LAZY``
    methods on :class:`CompositeQArray` delegate to a corresponding method here.

    Attributes:
        operators: Per-subsystem local operators (one square :class:`MaterializedQArray`
            per subsystem).
        coeff: Scalar coefficient; may be a JAX array for batched use. Defaults to 1.
    """

    operators: tuple[MaterializedQArray, ...]
    coeff: ArrayLike = 1.0

    # === Lifecycle ===

    def __check_init__(self):
        # ensure `operators` is a non-empty tuple
        if not isinstance(self.operators, tuple):
            raise TypeError(
                'Argument `operators` of `CompositeTerm` must be a tuple, but got '
                f'`{type(self.operators).__name__}`.'
            )
        if len(self.operators) == 0:
            raise ValueError(
                'Argument `operators` of `CompositeTerm` must contain at least one '
                'operator, but got `operators=()`.'
            )

        # ensure every operator is a materialized qarray, as assumed by every
        # lazy formula that reaches into `op.data`, `op.dims`, etc.
        if not all(isinstance(op, MaterializedQArray) for op in self.operators):
            raise TypeError(
                'Argument `operators` of `CompositeTerm` must contain only '
                '`MaterializedQArray` instances, but got types '
                f'{tuple(type(op).__name__ for op in self.operators)}.'
            )

        # === ensure the operators are square, as assumed by every lazy formula
        if any(op.shape[-1] != op.shape[-2] for op in self.operators):
            raise ValueError(
                'Argument `operators` of `CompositeTerm` must contain square qarrays, '
                f'but got shapes {tuple(op.shape for op in self.operators)}.'
            )

        # === ensure `coeff` is a scalar or a batched scalar of shape (..., 1, 1),
        # as assumed by `shape`, `_materialize` and every lazy formula
        if not is_batched_scalar(self.coeff):
            raise ValueError(
                'Argument `coeff` of `CompositeTerm` must be a scalar or a batched '
                f'scalar of shape (..., 1, 1), but got shape {jnp.shape(self.coeff)}.'
            )

        # === ensure a single layout and device, so that they can be reported
        # without materializing
        layouts = {op.layout for op in self.operators}
        if len(layouts) > 1:
            raise ValueError(
                'Argument `operators` of `CompositeTerm` must contain qarrays with '
                f'a common layout, but got layouts {tuple(layouts)}.'
            )
        devices = _devices(self)
        if len(devices) > 1:
            raise ValueError(
                'A `CompositeTerm` must be stored on a single device, but got '
                f'devices {tuple(devices)}.'
            )

    # === Materialization ===

    def _materialize(self) -> MaterializedQArray:
        tensor_product = reduce(lambda x, y: x & y, self.operators)
        return cast(MaterializedQArray, tensor_product * self.coeff)

    def _offsets(self) -> set[int]:
        # offsets of a Kronecker product combine as `o_0 * m_1 + o_1` (see
        # `and_sparsedia_sparsedia`); duplicates are merged, never dropped
        offsets = set(cast(SparseDIADataArray, self.operators[0].data).offsets)
        for op in self.operators[1:]:
            op_offsets = cast(SparseDIADataArray, op.data).offsets
            offsets = {o * op.shape[-1] + p for o in offsets for p in op_offsets}
        return offsets

    # === Properties ===

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.result_type(*(op.dtype for op in self.operators), self.coeff)

    @property
    def shape(self) -> tuple[int, ...]:
        batch = jnp.broadcast_shapes(
            *(op.shape[:-2] for op in self.operators), _coeff_batch_shape(self.coeff)
        )
        n = prod(op.shape[-1] for op in self.operators)
        return (*batch, n, n)

    @property
    def layout(self) -> Layout:
        return self.operators[0].layout

    @property
    def mT(self) -> CompositeTerm:
        # (A (x) B)^T = A^T (x) B^T
        return replace(self, operators=tuple(op.mT for op in self.operators))

    # === Array methods ===

    def conj(self) -> CompositeTerm:
        # conj(c·⊗A_k) = conj(c)·⊗conj(A_k) → each op's .conj() + jnp.conj(coeff).
        raise NotImplementedError

    def broadcast_to(self, *shape: int) -> CompositeTerm:
        # batch axes only → each op's .broadcast_to() + jnp.broadcast_to(coeff, ...).
        raise NotImplementedError

    def trace(self) -> Array:
        # tr(c·⊗A_k) = c·Π_k tr(A_k) → each op's .trace().
        raise NotImplementedError

    def sum(self, axis: int | tuple[int, ...] | None = None) -> CompositeTerm:
        # MATERIALIZE → _materialize().sum(axis).
        raise NotImplementedError

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> CompositeTerm:
        # batch axes only → each op's .squeeze(axis) + jnp.squeeze(coeff, axis).
        raise NotImplementedError

    def powm(self, n: int) -> CompositeTerm:
        # (c·⊗A_k)^n = c^n·⊗A_k^n → each op's .powm(n).
        raise NotImplementedError

    def expm(self, *, max_squarings: int = 16) -> MaterializedQArray:
        # exp(c·⊗A_k) = (⊗V_k)·diag(exp(c·∏λ_k))·(⊗V_k)^†; returns MaterializedQArray.
        # → each op's ._eigh().
        raise NotImplementedError

    def norm(self, *, psd: bool = False) -> Array:
        # LAZY if psd=False: ‖c·⊗A_k‖_F = |c|·Π_k‖A_k‖_F.
        # psd=True: trace shortcut only if known PSD; otherwise materialize.
        raise NotImplementedError

    def _eig(self) -> tuple[Array, MaterializedQArray]:
        # eigenvalues = c·Cartesian(λ_k), eigenvectors = ⊗V_k (materialized)
        # → each op's ._eig().
        raise NotImplementedError

    def _eigh(self) -> tuple[Array, Array]:
        # Hermitian variant; returns raw JAX arrays → each op's ._eigh().
        raise NotImplementedError

    def _eigvals(self) -> Array:
        # c · Cartesian product of per-op eigenvalues → each op's ._eigvals().
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        # Hermitian variant → each op's ._eigvalsh().
        raise NotImplementedError

    def devices(self) -> set[Device]:
        # delegate, so that a term reports the devices its materialized form would
        # and refuses the same way under tracing. `_devices` is for the invariants
        # that must hold while traced, not for this.
        return set().union(*(op.devices() for op in self.operators))

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
        # Sufficient (not necessary): coeff real AND all ops .isherm().
        # False here is not conclusive for multi-term CompositeQArray.
        raise NotImplementedError

    def block_until_ready(self) -> CompositeTerm:
        for op in self.operators:
            op.block_until_ready()
        if isinstance(self.coeff, Array):
            self.coeff.block_until_ready()
        return self

    # === Layout conversion ===

    def asdense(self) -> CompositeTerm:
        return replace(self, operators=tuple(op.asdense() for op in self.operators))

    def assparsedia(
        self, offsets: tuple[int, ...] | None = None
    ) -> CompositeTerm | MaterializedQArray:
        # `offsets` designates diagonals of the full matrix, which do not decompose
        # into per-operator offsets
        if offsets is not None:
            return cast(
                MaterializedQArray, self._materialize().assparsedia(offsets=offsets)
            )
        return replace(self, operators=tuple(op.assparsedia() for op in self.operators))

    # === Quantum methods ===

    def ptrace(self, keep: tuple[int, ...]) -> CompositeTerm:
        # ptrace_{∉keep}(c·⊗A_j) = c·(Π_{j∉keep} tr(A_j))·⊗_{k∈keep} A_k
        # → .trace() on each traced-out op.
        raise NotImplementedError

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> CompositeTerm:
        # batch axes only → each op's __getitem__.
        # Matrix-axis keys: caller materializes.
        raise NotImplementedError

    # === Arithmetic ===

    def __mul__(self, y: QArrayLike) -> CompositeTerm:
        # y·(c·⊗A_k) = (y·c)·⊗A_k; only touches coeff.
        raise NotImplementedError

    def __matmul__(self, other: CompositeTerm) -> CompositeTerm:
        # is the main mpoint of the feature
        raise NotImplementedError

    def __and__(self, other: CompositeTerm) -> CompositeTerm:
        # (c·⊗A_k)⊗(d·⊗B_l) = (c·d)·(A_*,B_*); tuple concat + coeff multiply.
        raise NotImplementedError


class CompositeQArray(QArray):
    r"""Lazy sum of separable tensor-product operators.

    $H = \sum_j c_j A_{j,0} \otimes \cdots \otimes A_{j,N-1}$, stored in factored form
    to avoid the exponential cost of the full $n \times n$ matrix.

    ``dims`` is inherited from :class:`QArray`.

    Strategy tags used in method comments:

    - ``LAZY``: implemented term-wise; no full matrix built.
    - ``MATERIALIZE``: falls back to ``_materialize().<method>(...)``.
    - ``MIXED``: LAZY for batch axes, MATERIALIZE for matrix axes.
    - ``1-term``: single-term fast path that skips full materialization.
    - ``★``: big-win lazy methods (core motivation for this class).

    Attributes:
        terms: Tuple of :class:`CompositeTerm` objects that sum to the operator.
    """

    terms: tuple[CompositeTerm, ...]

    # === Lifecycle ===

    def __check_init__(self):
        super().__check_init__()

        # === ensure `dims` describes multiple subsystems; a single-subsystem
        # operator is a `MaterializedQArray`
        if len(self.dims) < 2:
            raise ValueError(
                'Argument `dims` of `CompositeQArray` must describe at least two '
                f'subsystems, but got `dims={self.dims}`. For a single-subsystem '
                'operator, use a materialized qarray instead.'
            )

        # === ensure `terms` is a non-empty tuple
        if not isinstance(self.terms, tuple):
            raise TypeError(
                'Argument `terms` of `CompositeQArray` must be a tuple, but got '
                f'`{type(self.terms).__name__}`.'
            )
        if len(self.terms) == 0:
            raise ValueError(
                'Argument `terms` of `CompositeQArray` must contain at least one '
                'term, but got `terms=()`.'
            )

        # === ensure each term factorizes `dims`
        for j, term in enumerate(self.terms):
            term_dims = tuple(d for op in term.operators for d in op.dims)
            if term_dims != self.dims:
                raise ValueError(
                    f'Term {j} of `CompositeQArray` has operators whose dims '
                    f'concatenate to {term_dims}, which does not match '
                    f'`dims={self.dims}`.'
                )

        # === ensure a single layout and device, so that they can be reported
        # without materializing
        layouts = {term.layout for term in self.terms}
        if len(layouts) > 1:
            raise ValueError(
                'Argument `terms` of `CompositeQArray` must contain terms with a '
                f'common layout, but got layouts {tuple(layouts)}.'
            )
        devices = _devices(self)
        if len(devices) > 1:
            raise ValueError(
                'A `CompositeQArray` must be stored on a single device, but got '
                f'devices {tuple(devices)}.'
            )

    # === Materialization ===

    def _materialize(self) -> MaterializedQArray:
        return reduce(lambda x, y: x + y, (term._materialize() for term in self.terms))

    # === Properties ===

    @property
    def dtype(self) -> jnp.dtype:
        return jnp.result_type(*(term.dtype for term in self.terms))

    @property
    def layout(self) -> Layout:
        return self.terms[0].layout

    @property
    def shape(self) -> tuple[int, ...]:
        shapes = [term.shape for term in self.terms]
        batch = jnp.broadcast_shapes(*(shape[:-2] for shape in shapes))
        return (*batch, *shapes[0][-2:])

    @property
    def mT(self) -> QArray:
        return replace(self, terms=tuple(term.mT for term in self.terms))

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def ndiags(self) -> int:
        """Number of stored diagonals (only for sparse diagonal layout)."""
        if self.layout is not dia:
            raise AttributeError(
                f"Attribute 'ndiags' is only defined for sparse diagonal layouts; "
                f'got layout {self.layout!r}.'
            )
        # the offsets of a sum are the union of the offsets of its terms (see
        # `add_sparsedia_sparsedia`), and no diagonal is ever dropped
        return len(set().union(*(term._offsets() for term in self.terms)))

    # === Array methods ===

    def conj(self) -> QArray:
        # LAZY → term.conj().
        raise NotImplementedError

    def reshape(self, *shape: int) -> QArray:
        # MATERIALIZE → _materialize().reshape(*shape).
        raise NotImplementedError

    def _reshape_unchecked(self, *shape: int) -> QArray:
        # MATERIALIZE → _materialize()._reshape_unchecked(*shape).
        raise NotImplementedError

    def broadcast_to(self, *shape: int) -> QArray:
        # LAZY batch axes only → term.broadcast_to(...).
        raise NotImplementedError

    def swapaxes(self, axis1: int, axis2: int) -> QArray:
        # LAZY batch axes only → term.swapaxes(axis1, axis2).
        raise NotImplementedError

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> QArray:
        # LAZY batch axes only → term.moveaxis(source, destination).
        raise NotImplementedError

    def expand_dims(self, axis: int | Sequence[int]) -> QArray:
        # LAZY batch axes only → term.expand_dims(axis).
        raise NotImplementedError

    def powm(self, n: int) -> QArray:
        # MATERIALIZE | 1-term (c·⊗A_k)^n=c^n·⊗A_k^n → term.powm(n).
        raise NotImplementedError

    def expm(self, *, max_squarings: int = 16) -> QArray:
        # MATERIALIZE | 1-term per-factor spectral path → term.expm(...).
        raise NotImplementedError

    def norm(self, *, psd: bool = False) -> Array:
        # LAZY if psd=False: Gram sum over term pairs using local traces.
        # psd=True: trace shortcut only if known PSD; otherwise materialize.
        # can be unstable
        raise NotImplementedError

    def trace(self) -> Array:
        # LAZY tr(c·⊗A_k)=c·Π tr(A_k) → sum(term.trace()).
        raise NotImplementedError

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # MATERIALIZE → _materialize().sum(axis).
        raise NotImplementedError

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # LAZY → term.squeeze(axis).
        raise NotImplementedError

    def _eig(self) -> tuple[Array, QArray]:
        # MATERIALIZE | 1-term eigenvalues=c·Cartesian(λ_k), eigenvecs=⊗V_k
        # → term._eig().
        raise NotImplementedError

    def _eigh(self) -> tuple[Array, Array]:
        # MATERIALIZE | 1-term Hermitian variant → term._eigh().
        raise NotImplementedError

    def _eigvals(self) -> Array:
        # MATERIALIZE | 1-term → term._eigvals().
        raise NotImplementedError

    def _eigvalsh(self) -> Array:
        # MATERIALIZE | 1-term → term._eigvalsh().
        raise NotImplementedError

    def devices(self) -> set[Device]:
        # see `CompositeTerm.devices`
        return set().union(*(term.devices() for term in self.terms))

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
        # MATERIALIZE | 1-term sufficient check → term.isherm(rtol, atol).
        raise NotImplementedError

    def block_until_ready(self) -> QArray:
        for term in self.terms:
            term.block_until_ready()
        return self

    # === Quantum methods ===

    def ptrace(self, *keep: int) -> QArray:
        # LAZY → term.ptrace(keep).
        raise NotImplementedError

    # === Conversion ===

    def to_qutip(self) -> Qobj | list[Qobj]:
        return self._materialize().to_qutip()

    def to_jax(self) -> Array:
        return self._materialize().to_jax()

    def to_numpy(self) -> np.ndarray:
        return self._materialize().to_numpy()

    def __array__(self, dtype=None, copy=None) -> np.ndarray:  # noqa: ANN001
        return self._materialize().__array__(dtype=dtype, copy=copy)

    def asdense(self) -> QArray:
        """Converts to a dense layout.

        Returns:
            A qarray with dense data layout.
        """
        return replace(self, terms=tuple(term.asdense() for term in self.terms))

    def assparsedia(self, offsets: tuple[int, ...] | None = None) -> QArray:
        """Converts to a sparse diagonal layout.

        Args:
            offsets: Offsets of the stored diagonals. If `None`, offsets are determined
                automatically from the matrix structure. This argument can also be
                explicitly specified to ensure compatibility with JAX transformations,
                which require static offset values.

        Returns:
            A qarray with sparse diagonal data layout.
        """
        # `offsets` designates diagonals of the full matrix, which do not decompose
        # into per-operator offsets
        if offsets is not None:
            return self._materialize().assparsedia(offsets=offsets)
        return replace(self, terms=tuple(term.assparsedia() for term in self.terms))

    # === Repr ===

    def __repr__(self) -> str:
        # deliberately does not materialize: printing the full matrix would defeat
        # the purpose of the class
        res = (
            f'CompositeQArray: shape={self.shape}, dims={self.dims}, '
            f'dtype={self.dtype}, layout={self.layout}, n_terms={len(self.terms)}'
        )
        for j, term in enumerate(self.terms):
            coeff = jnp.asarray(term.coeff)
            coeff_str = (
                repr(coeff.reshape(()).item())
                if coeff.size == 1
                else f'<shape={coeff.shape}>'
            )
            op_dims = tuple(op.dims for op in term.operators)
            res += f'\n  term[{j}]: coeff={coeff_str}, op_dims={op_dims}'
        return res

    # === Arithmetic ===

    def __mul__(self, y: QArrayLike) -> QArray:
        # LAZY y·Σc_j⊗A_{jk}=Σ(y·c_j)⊗A_{jk} → term.__mul__(y).
        raise NotImplementedError

    def __add__(self, y: QArrayLike) -> QArray:
        # LAZY ★ two composites: self.terms + other.terms.
        # Non-composite y: wrap as single-operator CompositeTerm first.
        raise NotImplementedError

    @overload
    def __matmul__(self, y: QArray) -> QArray: ...

    @overload
    def __matmul__(self, y: ArrayLike) -> Array: ...

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        # LAZY ★ (Σc_j⊗A_jk)·(Σd_l⊗B_lk)=Σ_{j,l}(c_j·d_l)⊗(A_jk·B_lk) → term_j @ term_l.
        raise NotImplementedError

    @overload
    def __rmatmul__(self, y: QArray) -> QArray: ...

    @overload
    def __rmatmul__(self, y: ArrayLike) -> Array: ...

    def __rmatmul__(self, y: QArrayLike) -> QArray | Array:
        # LAZY symmetric to __matmul__ → term_other @ term_self.
        raise NotImplementedError

    def __and__(self, y: QArray) -> QArray:
        # LAZY ★ (Σc_j⊗A_jk)⊗(Σd_l⊗B_lk)=Σ_{j,l}(c_j·d_l)⊗(A_j*,B_l*) → term_j & term_l.
        raise NotImplementedError

    # === Element-wise ===

    def addscalar(self, y: ArrayLike) -> QArray:
        # MATERIALIZE → _materialize().addscalar(y).
        raise NotImplementedError

    def elmul(self, y: QArrayLike) -> QArray:
        # MATERIALIZE → _materialize().elmul(y).
        raise NotImplementedError

    def elpow(self, power: int) -> QArray:
        # MATERIALIZE → _materialize().elpow(power).
        raise NotImplementedError

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> QArray:
        # MIXED batch: term[key] | matrix: _materialize()[key].
        raise NotImplementedError
