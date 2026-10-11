from __future__ import annotations

import warnings
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
from .dataarray import IndexType, in_last_two_dims, key_touches_last_two_dims
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


def _key_in_batch_dims(key: IndexType, ndim: int) -> bool:
    # whether `key` only indexes batch axes, and thus leaves the tensor-product
    # structure intact
    return not key_touches_last_two_dims(key, ndim)


def _isherm(
    sufficient: Array, x: CompositeTerm | CompositeQArray, rtol: float, atol: float
) -> Array:
    # `sufficient` only proves Hermiticity, so a negative needs confirming on the
    # full matrix; short-circuiting on it needs a concrete value, hence the tracer
    # guard below (under tracing the matrix is always built)
    if not isinstance(sufficient, jax.core.Tracer) and bool(sufficient):
        return sufficient
    return x._materialize().isherm(rtol=rtol, atol=atol)


def _squeezable_batch_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
    # batch-axis positions of size 1, i.e. the axes an unqualified `squeeze()`
    # removes without touching the matrix axes
    return tuple(i for i, n in enumerate(shape[:-2]) if n == 1)


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

    def _aligned(self) -> CompositeTerm:
        # term whose operators and coefficient all carry the full batch shape, so
        # that batch axes line up positionally across leaves
        return self.broadcast_to(*self.shape)

    def _coeff(self) -> Array:
        # coefficient with the `(1, 1)` matrix-dummy axes stripped, to broadcast
        # against per-factor arrays whose trailing axis is not a matrix axis
        coeff = jnp.asarray(self.coeff)
        if coeff.ndim >= 2:
            return coeff.reshape(coeff.shape[:-2])
        return coeff.reshape(()) if coeff.shape == (1,) else coeff

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
        return replace(
            self,
            operators=tuple(op.conj() for op in self.operators),
            coeff=jnp.conj(jnp.asarray(self.coeff)),
        )

    def reshape(self, *shape: int) -> CompositeTerm:
        """Reshapes the term's batch axes."""
        term = self._aligned()
        operators = tuple(
            op.reshape(*shape[:-2], *op.shape[-2:]) for op in term.operators
        )
        coeff = jnp.asarray(term.coeff).reshape(*shape[:-2], 1, 1)
        return replace(term, operators=operators, coeff=coeff)

    def broadcast_to(self, *shape: int) -> CompositeTerm:
        """Broadcasts the term's batch axes to a new shape."""
        # unlike other batch-axis methods, no `_aligned()` call is needed
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because the last two dimensions '
                f'do not match current shape dimensions, {self.shape}.'
            )
        operators = tuple(
            op.broadcast_to(*shape[:-2], *op.shape[-2:]) for op in self.operators
        )
        coeff = jnp.broadcast_to(jnp.asarray(self.coeff), (*shape[:-2], 1, 1))
        return replace(self, operators=operators, coeff=coeff)

    def swapaxes(self, axis1: int, axis2: int) -> CompositeTerm:
        """Interchanges two batch axes of the term."""
        term = self._aligned()
        operators = tuple(op.swapaxes(axis1, axis2) for op in term.operators)
        coeff = jnp.swapaxes(jnp.asarray(term.coeff), axis1, axis2)
        return replace(term, operators=operators, coeff=coeff)

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> CompositeTerm:
        """Moves batch axes of the term to new positions."""
        term = self._aligned()
        operators = tuple(op.moveaxis(source, destination) for op in term.operators)
        coeff = jnp.moveaxis(jnp.asarray(term.coeff), source, destination)
        return replace(term, operators=operators, coeff=coeff)

    def expand_dims(self, axis: int | Sequence[int]) -> CompositeTerm:
        """Expands the term's batch axes by inserting new axes."""
        term = self._aligned()
        operators = tuple(op.expand_dims(axis) for op in term.operators)
        coeff = jnp.expand_dims(jnp.asarray(term.coeff), axis)
        return replace(term, operators=operators, coeff=coeff)

    def trace(self) -> Array:
        return self._coeff() * reduce(
            jnp.multiply, (op.trace() for op in self.operators)
        )

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        return self._materialize().sum(axis=axis)

    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> CompositeTerm | QArray | Array:
        # MIXED: an axis touching the matrix dims destroys the tensor-product
        # structure. `axis=None` only does so if one of them is size 1.
        if axis is None:
            if self.shape[-2] == 1 or self.shape[-1] == 1:
                return self._materialize().squeeze(axis=None)
            axis = _squeezable_batch_axes(self.shape)
        elif in_last_two_dims(axis, len(self.shape)):
            return self._materialize().squeeze(axis=axis)

        term = self._aligned()
        operators = tuple(op.squeeze(axis=axis) for op in term.operators)
        coeff = jnp.squeeze(jnp.asarray(term.coeff), axis=axis)
        return replace(term, operators=operators, coeff=coeff)

    def powm(self, n: int) -> CompositeTerm:
        return replace(
            self,
            operators=tuple(op.powm(n) for op in self.operators),
            coeff=jnp.asarray(self.coeff) ** n,
        )

    def expm(self, *, max_squarings: int = 16) -> MaterializedQArray:
        # not implemented: `CompositeQArray.expm` always materializes, so there is
        # no caller for a term-level formula
        raise NotImplementedError

    def norm(self, *, psd: bool = False) -> Array:
        # not implemented: `CompositeQArray.norm` is built directly from `trace`
        # and `_eigvalsh`, so there is no caller for a term-level formula
        raise NotImplementedError

    def _kron_evals(self, evals: Sequence[Array]) -> Array:
        # eigenvalues of a Kronecker product are the products of the per-factor
        # eigenvalues, flattened row-major to match the Kronecker convention
        out = evals[0]
        for lam in evals[1:]:
            out = (out[..., :, None] * lam[..., None, :]).reshape(*out.shape[:-1], -1)
        return self._coeff()[..., None] * out

    def _eig(self) -> tuple[Array, MaterializedQArray]:
        # each factor is diagonalized with `_eig` rather than `_eigh`, since a
        # Hermitian term can have non-Hermitian factors, e.g. iY⊗iY
        evals, evecs = zip(*(op._eig() for op in self.operators), strict=True)
        scaled_evals = self._kron_evals(evals)
        V = reduce(lambda x, y: x & y, evecs)
        # the coefficient may carry batch axes the operators do not
        if scaled_evals.shape[:-1] != V.shape[:-2]:
            V = V.broadcast_to(*scaled_evals.shape[:-1], *V.shape[-2:])
        return scaled_evals, cast(MaterializedQArray, V)

    def _eigh(self) -> tuple[Array, Array]:
        # not implemented: the tensor product of the per-factor `_eig` eigenvectors
        # is not orthonormal when a factor has a degenerate eigenvalue, so it
        # cannot satisfy the eigh contract; see `CompositeQArray._eigh`
        raise NotImplementedError

    def _eigvals(self) -> Array:
        return self._kron_evals([op._eigvals() for op in self.operators])

    def _eigvalsh(self) -> Array:
        return jnp.sort(self._eigvals().real, axis=-1)

    def devices(self) -> set[Device]:
        # delegate, so that a term reports the devices its materialized form would
        # and refuses the same way under tracing. `_devices` is for the invariants
        # that must hold while traced, not for this.
        return set().union(*(op.devices() for op in self.operators))

    def _isherm_sufficient(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
        # a real coefficient and Hermitian factors are enough, but not necessary:
        # e.g. iY⊗iY = -Y⊗Y is Hermitian with non-Hermitian factors
        return reduce(
            jnp.logical_and,
            (op.isherm(rtol=rtol, atol=atol) for op in self.operators),
            jnp.allclose(jnp.imag(jnp.asarray(self.coeff)), 0.0, atol=atol),
        )

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
        return _isherm(self._isherm_sufficient(rtol, atol), self, rtol, atol)

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
        # each operator spans `len(op.dims)` consecutive subsystems of the term:
        # fold the trace of fully traced-out operators into the coefficient, keep
        # fully kept operators as-is, and partial-trace the rest
        coeff = jnp.asarray(self.coeff)
        operators = []
        offset = 0
        for op in self.operators:
            local_keep = tuple(
                k - offset for k in keep if offset <= k < offset + len(op.dims)
            )
            if len(local_keep) == 0:
                coeff = coeff * op.trace()[..., None, None]
            elif len(local_keep) == len(op.dims):
                operators.append(op)
            else:
                operators.append(cast(MaterializedQArray, op.ptrace(*local_keep)))
            offset += len(op.dims)
        return replace(self, operators=tuple(operators), coeff=coeff)

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> CompositeTerm | MaterializedQArray:
        # MIXED: a key touching the matrix axes destroys the tensor-product
        # structure.
        if not _key_in_batch_dims(key, len(self.shape)):
            return cast(MaterializedQArray, self._materialize()[key])

        term = self._aligned()
        operators = tuple(op[key] for op in term.operators)
        coeff = jnp.asarray(term.coeff)[key]
        return replace(term, operators=operators, coeff=coeff)

    # === Arithmetic ===

    def __mul__(self, y: QArrayLike) -> CompositeTerm:
        if not is_batched_scalar(y):
            raise NotImplementedError(
                'Element-wise multiplication of two qarrays with the `*` operator '
                'is not supported. For matrix multiplication, use `x @ y`. For '
                'element-wise multiplication, use `x.elmul(y)`.'
            )
        return replace(self, coeff=y * jnp.asarray(self.coeff))

    def elpow(self, power: int) -> CompositeTerm:
        return replace(
            self,
            operators=tuple(
                cast(MaterializedQArray, op.elpow(power)) for op in self.operators
            ),
            coeff=jnp.asarray(self.coeff) ** power,
        )

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

    def _aligned(self) -> CompositeQArray:
        # see `CompositeTerm._aligned`
        return cast(CompositeQArray, self.broadcast_to(*self.shape))

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
        """Returns the element-wise complex conjugate of the qarray.

        Returns:
            New qarray with element-wise complex conjuguated values.
        """
        return replace(self, terms=tuple(term.conj() for term in self.terms))

    def reshape(self, *shape: int) -> QArray:
        """Returns a reshaped copy of a qarray.

        Args:
            *shape: New shape, which must match the original size.

        Returns:
            New qarray with the given shape.
        """
        # LAZY: the check below rejects any shape that would change the matrix
        # dims
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot reshape to shape {shape} because the last two dimensions do '
                f'not match current shape dimensions, {self.shape}.'
            )
        terms = tuple(term.reshape(*shape) for term in self._aligned().terms)
        return replace(self, terms=terms)

    def _reshape_unchecked(self, *shape: int) -> QArray:
        return self._materialize()._reshape_unchecked(*shape)

    def broadcast_to(self, *shape: int) -> QArray:
        """Broadcasts a qarray to a new shape.

        Args:
            *shape: New shape, which must be compatible with the original shape.

        Returns:
            New qarray with the given shape.
        """
        return replace(
            self, terms=tuple(term.broadcast_to(*shape) for term in self.terms)
        )

    def swapaxes(self, axis1: int, axis2: int) -> QArray:
        """Interchange two axes of a qarray."""
        terms = tuple(term.swapaxes(axis1, axis2) for term in self._aligned().terms)
        return replace(self, terms=terms)

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> QArray:
        """Move axes of a qarray to new positions."""
        terms = tuple(
            term.moveaxis(source, destination) for term in self._aligned().terms
        )
        return replace(self, terms=terms)

    def expand_dims(self, axis: int | Sequence[int]) -> QArray:
        """Expand the shape of a qarray by inserting new axes."""
        terms = tuple(term.expand_dims(axis) for term in self._aligned().terms)
        return replace(self, terms=terms)

    def powm(self, n: int) -> QArray:
        # the multinomial expansion of `(Σ_j T_j)^n` has `len(terms)^n` cross
        # terms, none of them separable in general
        if len(self.terms) > 1:
            return self._materialize().powm(n)
        return replace(self, terms=(self.terms[0].powm(n),))

    def expm(self, *, max_squarings: int = 16) -> QArray:
        # exp destroys the tensor-product structure, and the per-factor spectral
        # formula exp(c·⊗A_k) = V·diag(exp(c·λ))·V^-1 is both O(n^3) like Padé and
        # wrong for non-diagonalizable factors
        return self._materialize().expm(max_squarings=max_squarings)

    def norm(self, *, psd: bool = False) -> Array:
        # matches `dq.norm`: tr(H) for a PSD operator, Σ|λ_i| otherwise
        if psd:
            return self.trace().real

        from .._checks import check_hermitian  # noqa: PLC0415

        return jnp.abs(check_hermitian(self, 'x')._eigvalsh()).sum(-1)

    def trace(self) -> Array:
        return reduce(jnp.add, (term.trace() for term in self.terms))

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # MATERIALIZE per term (forced, see `CompositeTerm.sum`), but each term is
        # reduced along `axis` before combining
        return reduce(
            lambda x, y: x + y, (term.sum(axis=axis) for term in self._aligned().terms)
        )

    def squeeze(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # MIXED: an axis touching the matrix dims destroys the tensor-product
        # structure. `axis=None` only does so if one of them is size 1.
        if axis is None:
            if self.shape[-2] == 1 or self.shape[-1] == 1:
                return self._materialize().squeeze(axis=None)
            axis = _squeezable_batch_axes(self.shape)
        elif in_last_two_dims(axis, self.ndim):
            return self._materialize().squeeze(axis=axis)
        terms = tuple(term.squeeze(axis=axis) for term in self._aligned().terms)
        return replace(self, terms=cast(tuple[CompositeTerm, ...], terms))

    def _eig(self) -> tuple[Array, QArray]:
        # eigenvalues are not additive across terms
        if len(self.terms) > 1:
            return self._materialize()._eig()
        return self.terms[0]._eig()

    def _eigh(self) -> tuple[Array, Array]:
        # always materializes: the eigh contract requires an orthonormal basis,
        # which the per-factor `_eig` construction used by `_eig`/`_eigvals` does
        # not guarantee when a factor has a degenerate eigenvalue
        return self._materialize()._eigh()

    def _eigvals(self) -> Array:
        if len(self.terms) > 1:
            return self._materialize()._eigvals()
        return self.terms[0]._eigvals()

    def _eigvalsh(self) -> Array:
        if len(self.terms) > 1:
            return self._materialize()._eigvalsh()
        return self.terms[0]._eigvalsh()

    def devices(self) -> set[Device]:
        # see `CompositeTerm.devices`
        return set().union(*(term.devices() for term in self.terms))

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
        # every term sufficient is enough, but not necessary: cross-term
        # cancellations can also make the sum Hermitian, e.g. A⊗B + B⊗A
        sufficient = reduce(
            jnp.logical_and,
            (term._isherm_sufficient(rtol, atol) for term in self.terms),
        )
        return _isherm(sufficient, self, rtol, atol)

    def block_until_ready(self) -> QArray:
        for term in self.terms:
            term.block_until_ready()
        return self

    # === Quantum methods ===

    def ptrace(self, *keep: int) -> QArray:
        if len(keep) == 0:
            raise ValueError(
                '`ptrace` requires at least one subsystem to keep, but got `keep=()`.'
            )
        if len(set(keep)) != len(keep):
            raise ValueError(f'Argument `keep={keep}` must not contain duplicates.')
        if any(k < 0 or k >= len(self.dims) for k in keep):
            raise ValueError(
                f'Argument `keep={keep}` must match the Hilbert space structure '
                f'specified by `dims={self.dims}`.'
            )

        keep = tuple(sorted(keep))
        terms = tuple(term.ptrace(keep) for term in self.terms)
        dims = tuple(self.dims[k] for k in keep)

        # a single kept subsystem cannot be a `CompositeQArray` (`dims` must
        # describe at least two subsystems), so the reduced terms are combined
        # into one materialized qarray instead
        if len(dims) < 2:
            return reduce(lambda x, y: x + y, (term._materialize() for term in terms))
        return replace(self, dims=dims, terms=terms)

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
        return replace(self, terms=tuple(term * y for term in self.terms))

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
        """Adds a scalar.

        Args:
            y: Scalar to add, whose shape should be broadcastable with the qarray.

        Returns:
            New qarray resulting from the addition with the scalar.
        """
        # a shape that is not a batched scalar broadcasts against the matrix axes,
        # which no separable term can represent
        if not is_batched_scalar(y):
            return self._materialize().addscalar(y)

        # the all-ones factors below are fully dense, so a sparse layout cannot
        # represent them (and could not build their offsets under tracing)
        if self.layout is dia:
            warnings.warn(
                'A sparse qarray has been converted to dense layout due to the '
                'addition of a scalar.',
                # 3, not 2: equinox wraps the method call in a frame of its own
                stacklevel=3,
            )
            return self.asdense().addscalar(y)

        # H + y·J_n, with J_n = ⊗_k J_{d_k} the all-ones matrix (itself separable
        # since J_a⊗J_b = J_ab), added as one extra term
        from .utils import asqarray  # noqa: PLC0415  (avoid import cycle)

        devices = _devices(self)
        operators = []
        for d in self.dims:
            ones = jnp.ones((d, d), dtype=self.dtype)
            if len(devices) == 1:
                ones = jax.device_put(ones, next(iter(devices)))
            operators.append(
                cast(MaterializedQArray, asqarray(ones, dims=(d,), layout=self.layout))
            )

        term = CompositeTerm(operators=tuple(operators), coeff=y)
        return replace(self, terms=(*self.terms, term))

    def elmul(self, y: QArrayLike) -> QArray:
        # MATERIALIZE → _materialize().elmul(y).
        raise NotImplementedError

    def elpow(self, power: int) -> QArray:
        """Computes the element-wise power.

        Args:
            power: Power to raise to.

        Returns:
            New qarray with elements raised to the specified power.
        """
        # the element-wise power does not distribute over a sum in general
        if len(self.terms) > 1:
            return self._materialize().elpow(power)
        return replace(self, terms=(self.terms[0].elpow(power),))

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> QArray | Array:
        # MIXED: a key touching the matrix axes destroys the tensor-product
        # structure.
        if not _key_in_batch_dims(key, self.ndim):
            return self._materialize()[key]
        terms = tuple(term[key] for term in self._aligned().terms)
        return replace(self, terms=cast(tuple[CompositeTerm, ...], terms))
