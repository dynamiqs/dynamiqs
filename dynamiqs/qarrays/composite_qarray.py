from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import replace
from functools import reduce
from itertools import accumulate
from math import prod
from numbers import Complex, Integral
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
from .dense_dataarray import DenseDataArray
from .layout import Layout, dia
from .materialized_qarray import MaterializedQArray
from .qarray import QArray, QArrayLike, check_compatible_dims, isqarraylike, to_jax
from .sparsedia_dataarray import SparseDIADataArray

__all__ = []

# sparse diagonal offsets of each operator of a term, `None` to find them from data
_TermOffsets = tuple[tuple[int, ...] | None, ...]

_DEFAULT_COMPOSITE = False


def set_global_composite(composite: bool):
    global _DEFAULT_COMPOSITE  # noqa: PLW0603
    _DEFAULT_COMPOSITE = composite


def get_composite(composite: bool | None = None) -> bool:
    if composite is None:
        return _DEFAULT_COMPOSITE
    elif isinstance(composite, bool):
        return composite
    else:
        raise TypeError(
            f'Argument `composite` must be `True`, `False` or `None`, but is'
            f' `{composite}`.'
        )


def _devices(x: PyTree) -> set[Device]:
    # devices of every array leaf; tracers are skipped since they have no device
    return {
        device
        for leaf in jax.tree_util.tree_leaves(x)
        if isinstance(leaf, Array) and not isinstance(leaf, jax.core.Tracer)
        for device in leaf.devices()
    }


def _squeezable_batch_axes(shape: tuple[int, ...]) -> tuple[int, ...]:
    # batch-axis positions of size 1, i.e. the axes an unqualified `squeeze()`
    # removes without touching the matrix axes
    return tuple(i for i, n in enumerate(shape[:-2]) if n == 1)


def _batch_broadcastable(x_shape: tuple[int, ...], y_shape: tuple[int, ...]) -> bool:
    # aligned from the right, two batch sizes are equal or one is 1
    x_batch, y_batch = x_shape[:-2][::-1], y_shape[:-2][::-1]
    return all(a == b or 1 in (a, b) for a, b in zip(x_batch, y_batch, strict=False))


def _separable_dense_term(operators: tuple[Array, ...], y_tensor: Array) -> Array:
    # y in tensor form (d_0, ..., d_{N-1}, k): operator i contracts axis i
    tensor = y_tensor
    for axis, op in enumerate(operators):
        tensor = jnp.tensordot(op, tensor, axes=(1, axis))
        tensor = jnp.moveaxis(tensor, 0, axis)
    return tensor


def _separable_sparse_term(
    operators: tuple[SparseDIADataArray, ...], y_tensor: Array
) -> Array:
    # as `_separable_dense_term`, with the dia kernel on (d_i, rest); the axis order
    # is tracked, so that the data is transposed back only once
    tensor = y_tensor
    axis_order = list(range(y_tensor.ndim))
    for original_axis, op in enumerate(operators):
        current_axis = axis_order.index(original_axis)
        tensor = jnp.moveaxis(tensor, current_axis, 0)
        axis_order.insert(0, axis_order.pop(current_axis))

        old_shape = tensor.shape
        tensor = (op @ tensor.reshape(old_shape[0], -1)).to_jax()
        tensor = tensor.reshape(old_shape)

    inverse_perm = [axis_order.index(i) for i in range(y_tensor.ndim)]
    return jnp.transpose(tensor, inverse_perm)


def _coeff_from_batched_scalar(y: ArrayLike) -> Array:
    # a batched scalar has shape (), (1,) or (..., 1, 1), a coefficient shape (...)
    y = jnp.asarray(y)
    return y[..., 0, 0] if y.ndim > 1 else y.reshape(())


class CompositeTerm(eqx.Module):
    r"""One separable term $c \, A_0 \otimes \cdots \otimes A_{N-1}$ in a
    :class:`CompositeQArray`.  Holds the bulk of the lazy logic; most ``LAZY``
    methods on :class:`CompositeQArray` delegate to a corresponding method here.

    Attributes:
        operators: Per-subsystem local operators (one square :class:`MaterializedQArray`
            per subsystem).
        coeff: Coefficient, a scalar or an array of shape (...) for batched use.
            Defaults to 1.
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

        # === ensure the batch shapes of the operators and of `coeff` broadcast
        # together, as assumed by `shape` and every lazy formula
        batch_shapes = (
            *(op.shape[:-2] for op in self.operators),
            jnp.shape(self.coeff),
        )
        try:
            jnp.broadcast_shapes(*batch_shapes)
        except ValueError as err:
            raise ValueError(
                'Arguments `operators` and `coeff` of `CompositeTerm` must have '
                'broadcastable batch shapes, but got operator batch shapes '
                f'{batch_shapes[:-1]} and `coeff` shape {batch_shapes[-1]}.'
            ) from err

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
        return cast(MaterializedQArray, tensor_product * self._coeff_with_matrix_axes())

    def _coeff_with_matrix_axes(self) -> Array:
        # coefficient of shape (..., 1, 1), so that axes counted on the full shape of
        # the term, negative ones included, designate the same axes on it as on the
        # operators: e.g. for a term of shape (3, 4, 10, 10) with a coefficient of
        # shape (3, 4), axis -3 is the axis of size 4, which a coefficient of shape
        # (3, 4) does not have, but one of shape (3, 4, 1, 1) does
        return jnp.asarray(self.coeff)[..., None, None]

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
            *(op.shape[:-2] for op in self.operators), jnp.shape(self.coeff)
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
        # conj(c A ⊗ B) = conj(c) conj(A) ⊗ conj(B)
        operators = tuple(op.conj() for op in self.operators)
        return replace(self, operators=operators, coeff=jnp.conj(self.coeff))

    def reshape(self, *shape: int) -> CompositeTerm:
        """Reshapes the term's batch axes."""
        # with operators of shapes (3, 4, 2, 2) and (4, 5, 5), the term has shape
        # (3, 4, 10, 10) and `reshape(6, 2, 10, 10)` is valid, but the second
        # operator can only be reshaped to (6, 2, 5, 5) once broadcast to
        # (3, 4, 5, 5)
        term = self.broadcast_to(*self.shape)
        operators = tuple(
            op.reshape(*shape[:-2], *op.shape[-2:]) for op in term.operators
        )
        coeff = jnp.asarray(term.coeff).reshape(shape[:-2])
        return replace(term, operators=operators, coeff=coeff)

    def broadcast_to(self, *shape: int) -> CompositeTerm:
        """Broadcasts the term's batch axes to a new shape."""
        # each leaf is broadcast with its own matrix axes: for operators of shapes
        # (3, 4, 2, 2) and (4, 5, 5), `broadcast_to(2, 3, 4, 10, 10)` gives operators
        # of shapes (2, 3, 4, 2, 2) and (2, 3, 4, 5, 5). Broadcasting aligns axes
        # from the right, so unlike the other batch-axis methods, the leaves need no
        # prior broadcast
        if shape[-2:] != self.shape[-2:]:
            raise ValueError(
                f'Cannot broadcast to shape {shape} because the last two dimensions '
                f'do not match current shape dimensions, {self.shape}.'
            )
        operators = tuple(
            op.broadcast_to(*shape[:-2], *op.shape[-2:]) for op in self.operators
        )
        coeff = jnp.broadcast_to(jnp.asarray(self.coeff), shape[:-2])
        return replace(self, operators=operators, coeff=coeff)

    def swapaxes(self, axis1: int, axis2: int) -> CompositeTerm:
        """Interchanges two batch axes of the term."""
        # operators can have fewer batch axes than the term, e.g. (3, 4, 2, 2) and
        # (4, 5, 5) for a term of shape (3, 4, 10, 10): axis 0 of the second
        # operator is then the axis of size 4, not 3. Once both are broadcast to
        # the batch shape (3, 4), `swapaxes(0, 1)` gives (4, 3, 2, 2) and
        # (4, 3, 5, 5). `moveaxis`, `expand_dims`, `squeeze` and `__getitem__`
        # broadcast first for the same reason.
        term = self.broadcast_to(*self.shape)
        operators = tuple(op.swapaxes(axis1, axis2) for op in term.operators)
        coeff = jnp.swapaxes(term._coeff_with_matrix_axes(), axis1, axis2)
        return replace(term, operators=operators, coeff=coeff[..., 0, 0])

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> CompositeTerm:
        """Moves batch axes of the term to new positions."""
        # broadcast first, see `swapaxes`: for a term of shape (3, 4, 10, 10),
        # `moveaxis(0, 1)` gives (4, 3, 10, 10)
        term = self.broadcast_to(*self.shape)
        operators = tuple(op.moveaxis(source, destination) for op in term.operators)
        coeff = jnp.moveaxis(term._coeff_with_matrix_axes(), source, destination)
        return replace(term, operators=operators, coeff=coeff[..., 0, 0])

    def expand_dims(self, axis: int | Sequence[int]) -> CompositeTerm:
        """Expands the term's batch axes by inserting new axes."""
        # broadcast first, see `swapaxes`: for a term of shape (3, 4, 10, 10),
        # `expand_dims(1)` gives (3, 1, 4, 10, 10)
        term = self.broadcast_to(*self.shape)
        operators = tuple(op.expand_dims(axis) for op in term.operators)
        coeff = jnp.expand_dims(term._coeff_with_matrix_axes(), axis)
        return replace(term, operators=operators, coeff=coeff[..., 0, 0])

    def trace(self) -> Array:
        # tr(c A ⊗ B) = c tr(A) tr(B)
        traces = (op.trace() for op in self.operators)
        return reduce(jnp.multiply, traces, jnp.asarray(self.coeff))

    def sum(
        self, axis: int | tuple[int, ...] | None = None
    ) -> CompositeTerm | QArray | Array:
        # MIXED: summing over a batch axis keeps a single term only if at most one
        # leaf (operator or coefficient) varies along it, in which case the sum
        # factorizes: Σ_jk c_j·A_k ⊗ B = (Σ_j c_j)·(Σ_k A_k) ⊗ B. Summing over a
        # matrix axis, or over a batch axis along which several leaves vary, e.g.
        # Σ_k A_k ⊗ B_k, materializes. For operators of shapes (3, 4, 2, 2) and
        # (4, 5, 5), only the first varies along axis 0, so `sum(0)` gives a term of
        # shape (4, 10, 10), but both vary along axis 1, so `sum(1)` materializes.
        ndim = len(self.shape)
        if axis is None or in_last_two_dims(axis, ndim):
            return self._materialize().sum(axis=axis)
        axes = {a % ndim for a in ((axis,) if isinstance(axis, int) else axis)}

        # leaves are broadcast against the term from the right, so axis `a` of the
        # term is axis `a - pad` of a leaf with `pad` fewer batch axes
        coeff = jnp.asarray(self.coeff)
        batch_shapes = [op.shape[:-2] for op in self.operators] + [coeff.shape]
        pads = [ndim - 2 - len(batch_shape) for batch_shape in batch_shapes]
        varying_axes = [
            {a for a in axes if a >= pad and batch_shape[a - pad] > 1}
            for batch_shape, pad in zip(batch_shapes, pads, strict=True)
        ]
        if any(sum(a in varying for varying in varying_axes) > 1 for a in axes):
            return self._materialize().sum(axis=axis)

        leaf_axes = [tuple(a - pad for a in axes if a >= pad) for pad in pads]
        operators = tuple(
            op.sum(axis=op_axes) if op_axes else op
            for op, op_axes in zip(self.operators, leaf_axes[:-1], strict=True)
        )
        return replace(self, operators=operators, coeff=coeff.sum(leaf_axes[-1]))

    def squeeze(
        self, axis: int | tuple[int, ...] | None = None
    ) -> CompositeTerm | QArray | Array:
        # MIXED: an axis touching the matrix dims destroys the tensor-product
        # structure. `axis=None` only does so if one of them is size 1, which
        # happens only if every operator has size 1. For a term of shape
        # (1, 3, 1, 10, 10), `squeeze()` gives a term of shape (3, 10, 10) and
        # `squeeze(0)` one of shape (3, 1, 10, 10).
        if axis is None:
            if self.shape[-2] == 1 or self.shape[-1] == 1:
                return self._materialize().squeeze(axis=None)
            axis = _squeezable_batch_axes(self.shape)
        elif in_last_two_dims(axis, len(self.shape)):
            return self._materialize().squeeze(axis=axis)

        term = self.broadcast_to(*self.shape)
        operators = tuple(op.squeeze(axis=axis) for op in term.operators)
        coeff = jnp.squeeze(term._coeff_with_matrix_axes(), axis=axis)
        return replace(term, operators=operators, coeff=coeff[..., 0, 0])

    def powm(self, n: int) -> CompositeTerm:
        # (c A ⊗ B)^n = c^n A^n ⊗ B^n
        operators = tuple(cast(MaterializedQArray, op.powm(n)) for op in self.operators)
        return replace(self, operators=operators, coeff=jnp.asarray(self.coeff) ** n)

    def elpow(self, power: int) -> CompositeTerm:
        # element-wise: (c A ⊗ B)^p = c^p A^p ⊗ B^p
        operators = tuple(
            cast(MaterializedQArray, op.elpow(power)) for op in self.operators
        )
        coeff = jnp.asarray(self.coeff) ** power
        return replace(self, operators=operators, coeff=coeff)

    def _eig(self) -> tuple[Array, MaterializedQArray]:
        # (A ⊗ B)(v ⊗ w) = λμ v ⊗ w. Per-factor `_eig`, not `_eigh`: a Hermitian
        # term can have non-Hermitian factors (iY ⊗ iY)
        eigvals, eigvecs = zip(*(op._eig() for op in self.operators), strict=True)
        eigvals = self._kron_eigvals(eigvals)
        V = reduce(lambda x, y: x & y, eigvecs)
        # the coefficient can carry batch axes that the operators do not
        V = V.broadcast_to(*eigvals.shape[:-1], *V.shape[-2:])
        return eigvals, cast(MaterializedQArray, V)

    def _eigvals(self) -> Array:
        return self._kron_eigvals([op._eigvals() for op in self.operators])

    def _eigvalsh(self) -> Array:
        # real, even when those of the factors are not (±i for iY ⊗ iY)
        return jnp.sort(self._eigvals().real, axis=-1)

    def _kron_eigvals(self, eigvals: Sequence[Array]) -> Array:
        # eigenvalues of A ⊗ B are the products λ_i μ_j, in the row-major order of
        # the Kronecker product: (3, 2) and (4,) give (3, 8)
        def kron(x: Array, y: Array) -> Array:
            xy = x[..., :, None] * y[..., None, :]
            return xy.reshape(*xy.shape[:-2], -1)

        return jnp.asarray(self.coeff)[..., None] * reduce(kron, eigvals)

    def devices(self) -> set[Device]:
        # delegate, so that a term reports the devices its materialized form would
        # and refuses the same way under tracing. `_devices` is for the invariants
        # that must hold while traced, not for this.
        return set().union(*(op.devices() for op in self.operators))

    def _has_hermitian_factors(self) -> Array:
        # exact, so that the Kronecker product is exactly Hermitian too and the
        # full check agrees whatever its tolerance
        hermitian = [op.isherm(rtol=0.0, atol=0.0) for op in self.operators]
        real = jnp.all(jnp.imag(self.coeff) == 0)
        return jnp.all(jnp.stack([*hermitian, real]))

    def block_until_ready(self) -> CompositeTerm:
        for op in self.operators:
            op.block_until_ready()
        if isinstance(self.coeff, Array):
            self.coeff.block_until_ready()
        return self

    # === Layout conversion ===

    def asdense(self) -> CompositeTerm:
        return replace(self, operators=tuple(op.asdense() for op in self.operators))

    def assparsedia(self, offsets: _TermOffsets | None = None) -> CompositeTerm:
        # finding offsets from data needs concrete values, so under `jax.jit` every
        # dense operator needs static offsets
        if offsets is None:
            offsets = (None,) * len(self.operators)
        if len(offsets) != len(self.operators):
            raise ValueError(
                'Argument `offsets` of `CompositeTerm.assparsedia` must contain one '
                f'entry per operator ({len(self.operators)}), but got '
                f'{len(offsets)} entries.'
            )
        operators = tuple(
            op.assparsedia(offsets=op_offsets)
            for op, op_offsets in zip(self.operators, offsets, strict=True)
        )
        return replace(self, operators=operators)

    # === Quantum methods ===

    def ptrace(self, keep: tuple[int, ...]) -> CompositeTerm:
        # tr_1(c A ⊗ B ⊗ C) = c tr(B) A ⊗ C. For operators of dims (2, 3) and (2,),
        # `keep=(0, 2)` keeps the second one and partial-traces the first
        coeff = jnp.asarray(self.coeff)
        operators = []
        start = 0
        for op in self.operators:
            stop = start + len(op.dims)
            op_keep = tuple(k - start for k in keep if start <= k < stop)
            if len(op_keep) == 0:
                coeff = coeff * op.trace()
            elif len(op_keep) == len(op.dims):
                operators.append(op)
            else:
                operators.append(cast(MaterializedQArray, op.ptrace(*op_keep)))
            start = stop
        return replace(self, operators=tuple(operators), coeff=coeff)

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> CompositeTerm | MaterializedQArray:
        # MIXED: a key touching the matrix axes destroys the tensor-product
        # structure. For a term of shape (3, 4, 10, 10), `x[0]` and `x[:, 1:3]` stay
        # lazy, with shapes (4, 10, 10) and (3, 2, 10, 10), but `x[..., 0, 1]`
        # materializes. The key also applies to the coefficient with its matrix
        # axes, so that a key like `x[..., 2, :, :]` indexes its axis of size 4.
        if key_touches_last_two_dims(key, len(self.shape)):
            return cast(MaterializedQArray, self._materialize()[key])

        term = self.broadcast_to(*self.shape)
        operators = tuple(op[key] for op in term.operators)
        coeff = term._coeff_with_matrix_axes()[key][..., 0, 0]
        return replace(term, operators=operators, coeff=coeff)

    # === Arithmetic ===

    def __mul__(self, y: Array) -> CompositeTerm:
        # `y` is a coefficient of shape (...), see `CompositeQArray.__mul__`
        return replace(self, coeff=y * self.coeff)

    def __matmul__(self, other: CompositeTerm) -> CompositeTerm:
        # (c A ⊗ B)(d C ⊗ D) = cd AC ⊗ BD, once both terms factorize `dims` alike
        x, y = self._coarsened_to(other), other._coarsened_to(self)
        operators = tuple(
            cast(MaterializedQArray, A @ B)
            for A, B in zip(x.operators, y.operators, strict=True)
        )
        return replace(x, operators=operators, coeff=jnp.asarray(x.coeff) * y.coeff)

    def elmul(self, other: CompositeTerm) -> CompositeTerm:
        # (c A ⊗ B) ∘ (d C ⊗ D) = cd (A ∘ C) ⊗ (B ∘ D), see `__matmul__`
        x, y = self._coarsened_to(other), other._coarsened_to(self)
        operators = tuple(
            cast(MaterializedQArray, A.elmul(B))
            for A, B in zip(x.operators, y.operators, strict=True)
        )
        return replace(x, operators=operators, coeff=jnp.asarray(x.coeff) * y.coeff)

    def _coarsened_to(self, other: CompositeTerm) -> CompositeTerm:
        # keeps the boundaries shared with `other`: against dims (2, 3)(4,), operators
        # of dims (2,)(3,)(4,) become (2, 3)(4,), and (2,)(3, 4) become (2, 3, 4)
        starts = set(self._operator_starts()) & set(other._operator_starts())
        groups = []
        for start, op in zip(self._operator_starts(), self.operators, strict=True):
            if start in starts:
                groups.append([op])
            else:
                groups[-1].append(op)
        operators = tuple(reduce(lambda x, y: x & y, group) for group in groups)
        return replace(self, operators=cast(tuple[MaterializedQArray, ...], operators))

    def _operator_starts(self) -> list[int]:
        # first subsystem of each operator: dims (2, 3)(4,) give [0, 2]
        return list(accumulate((len(op.dims) for op in self.operators[:-1]), initial=0))

    def _shares_inner_boundary(self, other: CompositeTerm) -> bool:
        # dims (2, 3)(4,) and (2,)(3, 4) share only the start 0: both coarsen to one
        # operator on the full space, whose product costs the materialized one
        starts = set(self._operator_starts()) & set(other._operator_starts())
        return len(starts) > 1

    def __and__(self, other: CompositeTerm) -> CompositeTerm:
        # (c A ⊗ B) ⊗ (d C) = cd A ⊗ B ⊗ C
        operators = (*self.operators, *other.operators)
        coeff = jnp.asarray(self.coeff) * other.coeff
        return replace(self, operators=operators, coeff=coeff)

    def _matmul_array(self, y: Array) -> Array:
        # ★ (A ⊗ B) y without A ⊗ B, on y in tensor form (d_A, d_B, k); `vmap` maps
        # the batch of the operators and y, not of `coeff`, which only scales: shapes
        # (3, 2, 2) and (5, 5) with y of shape (10, 4) map over 3 copies of y and A
        batch = jnp.broadcast_shapes(
            *(op.shape[:-2] for op in self.operators), y.shape[:-2]
        )
        batch_size, (n, k) = prod(batch), y.shape[-2:]
        y = jnp.broadcast_to(y, (*batch, n, k))
        y_tensor = y.reshape(batch_size, *(op.shape[-1] for op in self.operators), k)

        operators = [
            op
            if op.ndim == 2
            else op.broadcast_to(*batch, *op.shape[-2:]).reshape(-1, *op.shape[-2:])
            for op in self.operators
        ]
        operators = cast(list[MaterializedQArray], operators)
        in_axes = tuple(None if op.ndim == 2 else 0 for op in operators)
        if self.layout is dia:
            sparse_ops = tuple(cast(SparseDIADataArray, op.data) for op in operators)
            term = jax.vmap(_separable_sparse_term, in_axes=(in_axes, 0))
            Hy = term(sparse_ops, y_tensor)
        else:
            dense_ops = tuple(op.to_jax() for op in operators)
            term = jax.vmap(_separable_dense_term, in_axes=(in_axes, 0))
            Hy = term(dense_ops, y_tensor)
        return self._coeff_with_matrix_axes() * Hy.reshape(*batch, n, k)


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
        # terms can have different batch shapes, e.g. (3,) and (2, 1) for a qarray
        # of batch shape (2, 3), so they are broadcast to (2, 3) before
        # `reshape(6, n, n)`, as operators are within a term (see
        # `CompositeTerm.reshape`). `swapaxes`, `moveaxis`, `expand_dims`,
        # `squeeze` and `__getitem__` broadcast first for the same reason.
        terms = self.broadcast_to(*self.shape).terms
        return replace(self, terms=tuple(term.reshape(*shape) for term in terms))

    def _reshape_unchecked(self, *shape: int) -> QArray:
        return self._materialize()._reshape_unchecked(*shape)

    def broadcast_to(self, *shape: int) -> CompositeQArray:
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
        terms = self.broadcast_to(*self.shape).terms
        return replace(self, terms=tuple(term.swapaxes(axis1, axis2) for term in terms))

    def moveaxis(
        self, source: int | Sequence[int], destination: int | Sequence[int]
    ) -> QArray:
        """Move axes of a qarray to new positions."""
        terms = self.broadcast_to(*self.shape).terms
        terms = tuple(term.moveaxis(source, destination) for term in terms)
        return replace(self, terms=terms)

    def expand_dims(self, axis: int | Sequence[int]) -> QArray:
        """Expand the shape of a qarray by inserting new axes."""
        terms = self.broadcast_to(*self.shape).terms
        return replace(self, terms=tuple(term.expand_dims(axis) for term in terms))

    def powm(self, n: int) -> QArray:
        # 1-term: (T_0 + T_1)^n expands into 2^n products of terms, so the lazy
        # route grows exponentially with n
        if len(self.terms) > 1:
            return self._materialize().powm(n)
        return replace(self, terms=(self.terms[0].powm(n),))

    def expm(self, *, max_squarings: int = 16) -> QArray:
        # MATERIALIZE: exp(c A ⊗ B) is not separable
        return self._materialize().expm(max_squarings=max_squarings)

    def norm(self, *, psd: bool = False) -> Array:
        # through `dq.norm`, which only calls `trace`, `isherm` and `_eigvalsh`
        from ..utils.general import norm  # noqa: PLC0415

        return norm(self, psd=psd)

    def trace(self) -> Array:
        return reduce(jnp.add, (term.trace() for term in self.terms))

    def sum(self, axis: int | tuple[int, ...] | None = None) -> QArray | Array:
        # MIXED: LAZY if every term stays a single term (see `CompositeTerm.sum`),
        # MATERIALIZE otherwise
        if axis is None or in_last_two_dims(axis, self.ndim):
            return self._materialize().sum(axis=axis)
        axes = {a % self.ndim for a in ((axis,) if isinstance(axis, int) else axis)}

        summed_terms = []
        for term in self.terms:
            # terms are broadcast against the qarray from the right, so a term with
            # `pad` fewer batch axes, or of size 1 along a summed axis of size n, is
            # constant along it and counted n times: e.g. with terms of batch shapes
            # (2, 3) and (3,), `sum(0)` counts the second term twice
            pad = self.ndim - len(term.shape)
            count = prod(
                self.shape[a] for a in axes if a < pad or term.shape[a - pad] == 1
            )
            counted = replace(term, coeff=count * jnp.asarray(term.coeff))
            term_axes = tuple(a - pad for a in axes if a >= pad)
            summed_terms.append(counted.sum(axis=term_axes) if term_axes else counted)

        if all(isinstance(term, CompositeTerm) for term in summed_terms):
            return replace(self, terms=tuple(summed_terms))
        return reduce(
            lambda x, y: x + y,
            (
                term._materialize() if isinstance(term, CompositeTerm) else term
                for term in summed_terms
            ),
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
        terms = self.broadcast_to(*self.shape).terms
        terms = tuple(term.squeeze(axis=axis) for term in terms)
        return replace(self, terms=cast(tuple[CompositeTerm, ...], terms))

    def _eig(self) -> tuple[Array, QArray]:
        # 1-term: the spectrum of a sum is not given by those of its terms
        if len(self.terms) > 1:
            return self._materialize()._eig()
        return self.terms[0]._eig()

    def _eigh(self) -> tuple[Array, Array]:
        # MATERIALIZE: per-factor `_eigh` is wrong for non-Hermitian factors, and
        # per-factor `_eig` is not orthonormal within a degenerate eigenspace
        return self._materialize()._eigh()

    def _eigvals(self) -> Array:
        # 1-term, see `_eig`
        if len(self.terms) > 1:
            return self._materialize()._eigvals()
        return self.terms[0]._eigvals()

    def _eigvalsh(self) -> Array:
        # 1-term, see `_eig`
        if len(self.terms) > 1:
            return self._materialize()._eigvalsh()
        return self.terms[0]._eigvalsh()

    def devices(self) -> set[Device]:
        # see `CompositeTerm.devices`
        return set().union(*(term.devices() for term in self.terms))

    def isherm(self, rtol: float = 1e-5, atol: float = 1e-8) -> Array:
        # SUFFICIENT: Hermitian factors prove it, but a Hermitian sum may have none
        # (iY ⊗ iY, A ⊗ B + B ⊗ A), so a negative or a tracer checks the matrix
        sufficient = jnp.all(
            jnp.stack([term._has_hermitian_factors() for term in self.terms])
        )
        if not isinstance(sufficient, jax.core.Tracer) and sufficient:
            return sufficient
        return self._materialize().isherm(rtol=rtol, atol=atol)

    def block_until_ready(self) -> QArray:
        for term in self.terms:
            term.block_until_ready()
        return self

    # === Quantum methods ===

    def ptrace(self, *keep: int) -> QArray:
        if len(keep) == 0:
            raise ValueError(
                'Argument `keep` must contain at least one subsystem, but got'
                ' `keep=()`.'
            )
        if len(set(keep)) != len(keep):
            raise ValueError(
                f'Argument `keep` must not contain duplicates, but got `keep={keep}`.'
            )
        if any(k < 0 or k > len(self.dims) - 1 for k in keep):
            raise ValueError(
                'Argument `keep` must match the Hilbert space structure specified by'
                ' `dims`.'
            )

        # the partial trace of a materialized qarray is always dense
        keep = tuple(sorted(keep))
        terms = tuple(term.asdense().ptrace(keep) for term in self.terms)
        dims = tuple(self.dims[k] for k in keep)
        if len(dims) == 1:
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

    def asdense(self) -> CompositeQArray:
        """Converts to a dense layout.

        Returns:
            A qarray with dense data layout.
        """
        return replace(self, terms=tuple(term.asdense() for term in self.terms))

    def assparsedia(
        self, offsets: tuple[int, ...] | tuple[_TermOffsets | None, ...] | None = None
    ) -> CompositeQArray:
        """Converts to a sparse diagonal layout.

        Args:
            offsets: Offsets of the stored diagonals of each operator of each term,
                e.g. `((offsets_A0, offsets_B0), (offsets_A1, offsets_B1))`, with
                `None` for any term or operator whose offsets should be determined
                automatically from its structure. If `None`, all offsets are
                determined automatically. Static offsets ensure compatibility with
                JAX transformations. Offsets of an operator that is already in sparse
                diagonal layout are ignored.

        Returns:
            A qarray with sparse diagonal data layout.
        """
        # the signature accepts `tuple[int, ...]` to match `QArray.assparsedia`, but
        # diagonals of the full matrix do not decompose into per-operator offsets
        if offsets is None:
            offsets = (None,) * len(self.terms)
        if any(isinstance(o, Integral) for o in offsets):
            raise ValueError(
                'Argument `offsets` of `CompositeQArray.assparsedia` must contain the '
                'offsets of each operator of each term, but got integers, which '
                'designate diagonals of the full matrix.'
            )
        if len(offsets) != len(self.terms):
            raise ValueError(
                'Argument `offsets` of `CompositeQArray.assparsedia` must contain one '
                f'entry per term ({len(self.terms)}), but got {len(offsets)} entries.'
            )
        term_offsets = cast(tuple[_TermOffsets | None, ...], offsets)
        terms = tuple(
            term.assparsedia(offsets=o)
            for term, o in zip(self.terms, term_offsets, strict=True)
        )
        return replace(self, terms=terms)

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
        if not is_batched_scalar(y):
            if not isqarraylike(y):
                return NotImplemented
            raise NotImplementedError(
                'Element-wise multiplication of two qarrays with the `*` operator is '
                'not supported. For matrix multiplication, use `x @ y`. For '
                'element-wise multiplication, use `x.elmul(y)`.'
            )
        coeff = _coeff_from_batched_scalar(y)
        return replace(self, terms=tuple(term * coeff for term in self.terms))

    def __add__(self, y: QArrayLike) -> QArray:
        # no dtype promotion: `x + 0j` keeps a real `x` real
        if isinstance(y, Complex) and y == 0:
            return self

        if is_batched_scalar(y):
            raise NotImplementedError(
                'Adding a scalar to a qarray with the `+` operator is not supported. '
                'To add a scaled identity matrix, use `x + scalar * dq.eye_like(x)`.'
                ' To add a scalar, use `x.addscalar(scalar)`.'
            )

        if not isqarraylike(y):
            return NotImplemented

        # MATERIALIZE an array-like, which may broadcast against the matrix axes
        if not isinstance(y, QArray):
            return self._materialize() + y

        check_compatible_dims(self.dims, y.dims)
        if isinstance(y, MaterializedQArray):
            # a ket or a bra adds element-wise
            if y.shape[-2:] != self.shape[-2:]:
                return self._materialize() + y
            y = replace(self, terms=(CompositeTerm((y,)),))
        y = cast(CompositeQArray, y)
        if not _batch_broadcastable(self.shape, y.shape):
            return self._materialize() + y._materialize()

        x = self
        if x.layout is not y.layout:
            warnings.warn(
                'A sparse data array has been converted to dense layout due to '
                'element-wise addition with a dense data array.',
                stacklevel=3,
            )
            x, y = x.asdense(), y.asdense()

        # LAZY ★ Σ_j T_j + Σ_l T'_l
        return replace(x, terms=x.terms + y.terms)

    @overload
    def __matmul__(self, y: QArray) -> QArray: ...

    @overload
    def __matmul__(self, y: ArrayLike) -> Array: ...

    def __matmul__(self, y: QArrayLike) -> QArray | Array:
        if is_batched_scalar(y):
            raise TypeError('Attempted matrix product between a scalar and a qarray.')

        if isinstance(y, CompositeQArray):
            check_compatible_dims(self.dims, y.dims)
            # MATERIALIZE if a pair of terms would cost a full product, once per pair
            if not _batch_broadcastable(self.shape, y.shape) or not all(
                tj._shares_inner_boundary(tl) for tj in self.terms for tl in y.terms
            ):
                return self._materialize() @ y._materialize()
            # LAZY ★ (Σ_j T_j)(Σ_l T'_l) = Σ_jl T_j T'_l
            terms = tuple(tj @ tl for tj in self.terms for tl in y.terms)
            return replace(self, terms=terms)

        if isinstance(y, MaterializedQArray):
            check_compatible_dims(self.dims, y.dims)
            # dia @ dia is dia, and the full matrix of a dia composite is sparse
            if self.layout is dia and y.layout is dia:
                return self._materialize() @ y
            y_data = y.to_jax()
        elif isqarraylike(y):
            y_data = to_jax(y)
        else:
            return NotImplemented

        if (
            y_data.ndim < 2
            or y_data.shape[-2] != self.shape[-1]
            or not _batch_broadcastable(self.shape, y_data.shape)
        ):
            return self._materialize() @ y_data

        # ★ mode by mode, dense like the materialized product
        data = DenseDataArray(self._matmul_array(y_data))
        return MaterializedQArray(dims=self.dims, vectorized=False, data=data)

    @overload
    def __rmatmul__(self, y: QArray) -> QArray: ...

    @overload
    def __rmatmul__(self, y: ArrayLike) -> Array: ...

    def __rmatmul__(self, y: QArrayLike) -> QArray | Array:
        if is_batched_scalar(y):
            raise TypeError('Attempted matrix product between a scalar and a qarray.')

        # reached for `y @ x` with a materialized `y`, see `__matmul__`
        if isinstance(y, MaterializedQArray):
            check_compatible_dims(y.dims, self.dims)
            if y.layout is dia and self.layout is dia:
                return y @ self._materialize()
            y_data = y.to_jax()
        elif isqarraylike(y):
            y_data = to_jax(y)
        else:
            return NotImplemented

        if (
            y_data.ndim < 2
            or y_data.shape[-1] != self.shape[-2]
            or not _batch_broadcastable(self.shape, y_data.shape)
        ):
            return y_data @ self._materialize()

        # ★ y H = (H^T y^T)^T, and `mT` is lazy
        H_T = cast(CompositeQArray, self.mT)
        data = DenseDataArray(H_T._matmul_array(y_data.mT).mT)
        return MaterializedQArray(dims=self.dims, vectorized=False, data=data)

    def _matmul_array(self, y: Array) -> Array:
        return reduce(jnp.add, (term._matmul_array(y) for term in self.terms))

    def __and__(self, y: QArray) -> QArray:
        if not isinstance(y, MaterializedQArray | CompositeQArray):
            return NotImplemented

        # dense ⊗ dia is dense, without a warning
        if self.layout is not y.layout:
            return self.asdense() & y.asdense()

        # LAZY ★ (Σ_j T_j) ⊗ (Σ_l T'_l) = Σ_jl T_j ⊗ T'_l
        if isinstance(y, MaterializedQArray):
            # a ket or a superoperator cannot be a term operator
            if (
                y.vectorized
                or not y.isop()
                or not _batch_broadcastable(self.shape, y.shape)
            ):
                return self._materialize() & y
            terms = tuple(term & CompositeTerm((y,)) for term in self.terms)
        else:
            if not _batch_broadcastable(self.shape, y.shape):
                return self._materialize() & y._materialize()
            terms = tuple(tj & tl for tj in self.terms for tl in y.terms)
        return replace(self, dims=self.dims + y.dims, terms=terms)

    def __rand__(self, y: QArray) -> QArray:
        # reached for `y & x` with a materialized `y`, see `__and__`
        if not isinstance(y, MaterializedQArray):
            return NotImplemented

        if self.layout is not y.layout:
            return y.asdense() & self.asdense()

        if (
            y.vectorized
            or not y.isop()
            or not _batch_broadcastable(self.shape, y.shape)
        ):
            return y & self._materialize()
        terms = tuple(CompositeTerm((y,)) & term for term in self.terms)
        return replace(self, dims=y.dims + self.dims, terms=terms)

    # === Element-wise ===

    def addscalar(self, y: ArrayLike) -> QArray:
        """Adds a scalar.

        Args:
            y: Scalar to add, whose shape should be broadcastable with the qarray.

        Returns:
            New qarray resulting from the addition with the scalar.
        """
        # MATERIALIZE if `y` broadcasts against the matrix axes, e.g. shape (n, n)
        if not is_batched_scalar(y):
            return self._materialize().addscalar(y)

        if self.layout is dia:
            warnings.warn(
                'A sparse data array has been converted to dense layout due to '
                'element-wise addition with a raw array.',
                # 3, not 2: equinox wraps the method call in a frame of its own
                stacklevel=3,
            )
            return self.asdense().addscalar(y)

        # x + y J, with J the all-ones matrix, added as one more term since
        # J_a ⊗ J_b = J_ab
        from .utils import asqarray  # noqa: PLC0415

        devices = _devices(self)
        operators = []
        for d in self.dims:
            J = jnp.ones((d, d), dtype=self.dtype)
            if len(devices) == 1:
                J = jax.device_put(J, next(iter(devices)))
            operators.append(cast(MaterializedQArray, asqarray(J, dims=(d,))))
        term = CompositeTerm(tuple(operators), _coeff_from_batched_scalar(y))
        return replace(self, terms=(*self.terms, term))

    def elmul(self, y: QArrayLike) -> QArray:
        """Computes the element-wise multiplication.

        Args:
            y: Qarray-like to multiply with element-wise.

        Returns:
            New qarray resulting from the element-wise multiplication.
        """
        # MATERIALIZE unless `y` is composite: (A ⊗ B) ∘ M is not separable
        if not isinstance(y, CompositeQArray):
            return self._materialize().elmul(y)

        check_compatible_dims(self.dims, y.dims)
        # MATERIALIZE if a pair of terms would cost a full product, see `__matmul__`
        if not _batch_broadcastable(self.shape, y.shape) or not all(
            tj._shares_inner_boundary(tl) for tj in self.terms for tl in y.terms
        ):
            return self._materialize().elmul(y._materialize())

        # LAZY (Σ_j T_j) ∘ (Σ_l T'_l) = Σ_jl T_j ∘ T'_l
        terms = tuple(tj.elmul(tl) for tj in self.terms for tl in y.terms)
        return replace(self, terms=terms)

    def elpow(self, power: int) -> QArray:
        """Computes the element-wise power.

        Args:
            power: Power to raise to.

        Returns:
            New qarray with elements raised to the specified power.
        """
        # 1-term, see `powm`, and an integer power >= 0: (ab)^p != a^p b^p for a
        # fractional p (branch cut), and 0^-1 gives nan in a product instead of inf
        if len(self.terms) > 1 or not isinstance(power, int) or power < 0:
            return self._materialize().elpow(power)
        return replace(self, terms=(self.terms[0].elpow(power),))

    # === Indexing ===

    def __getitem__(self, key: IndexType) -> QArray | Array:
        # MIXED: a key touching the matrix axes destroys the tensor-product
        # structure.
        if key_touches_last_two_dims(key, self.ndim):
            return self._materialize()[key]
        terms = tuple(term[key] for term in self.broadcast_to(*self.shape).terms)
        return replace(self, terms=cast(tuple[CompositeTerm, ...], terms))
