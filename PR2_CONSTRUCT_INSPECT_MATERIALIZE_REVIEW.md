# PR2 — `CompositeQArray`/`CompositeTerm`: construct, inspect, materialize

Branch `feat/composite-qarray-construct-inspect-materialize`, off
`fix/sparsedia-batched-kron-and-devices`. Scope: everything needed to
**construct** a `CompositeTerm`/`CompositeQArray`, **inspect** it (properties,
`ndiags`, `devices`, `__repr__`), and **materialize** it (build the full matrix
and everything that depends on doing so). No arithmetic, no batch-axis
manipulation, no spectral methods — those are later PRs.

This document walks every import, every implemented method, and every
construction check in [`composite_qarray.py`](dynamiqs/qarrays/composite_qarray.py),
explains the choice with its code, and gives a verdict per item. A final
section rolls everything up.

Legend: ✅ correct/complete/good as is · ⚠️ correct but worth a note · ❌ defect found and fixed during this work (kept for the record).

---

## 1. Imports

```python
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
```

| Import | Used for | Verdict |
|---|---|---|
| `from __future__ import annotations` | Lets every method annotate with `CompositeTerm`/`CompositeQArray` before the class body finishes (`-> CompositeTerm` inside `CompositeTerm` itself), and — important side effect — makes every annotation a string, never evaluated at runtime. That's what makes the `MaterializedQArray.__add__` return-type narrowing (§7) provably zero-risk: it cannot change any runtime bytecode. | ✅ |
| `collections.abc.Sequence` | Type hint for `moveaxis`/`expand_dims`'s `axis: int \| Sequence[int]` stub signatures. | ✅ — needed only because those two stubs had to be *added* (§6); not otherwise used in PR2's real code. |
| `dataclasses.replace` | The core "modify one field, keep the rest" primitive used by every lazy method that returns a new `CompositeTerm`/`CompositeQArray` (`mT`, `asdense`, `assparsedia`). `eqx.Module` re-runs `__check_init__` on every `replace`, so this can never silently produce an invalid instance. | ✅ |
| `functools.reduce` | Left-folds `&` over operators (`CompositeTerm._materialize`) and `+` over terms (`CompositeQArray._materialize`). | ✅ — see §7 for the type-level fix this needed. |
| `math.prod` | `CompositeTerm.shape`'s matrix dimension: $\prod_k m_k$. | ✅ |
| `typing.cast` | Three uses, all narrowing a base-class-typed expression to the concrete subtype the code *knows* it is (§4.3, §4.4). Not used to paper over a real ambiguity — see §7 for the one place a `cast` would have been the wrong tool and a real annotation fix was used instead. | ✅ |
| `typing.overload` | Only decorates the still-stubbed `__matmul__`/`__rmatmul__` overloads inherited from the skeleton; unused by anything in PR2's real scope. | ✅ |
| `equinox as eqx` | Base class for `CompositeTerm` (`eqx.Module`); provides the pytree machinery and the `__check_init__` hook both classes rely on. | ✅ |
| `jax` / `jax.core` | `jax.tree_util.tree_leaves` and `jax.core.Tracer` inside `_devices`. `jax.core` is imported **explicitly** — `import jax` alone does not guarantee `jax.core` is a resolvable attribute (confirmed: `ty` flagged `possibly-missing-submodule` until this was added). | ✅ — fixed for real, not suppressed (§7). |
| `jax.numpy as jnp` | `result_type`, `broadcast_shapes`, `asarray`, `shape` — every dtype/shape computation. | ✅ |
| `numpy as np` | Only for the `to_numpy`/`__array__` return-type annotations (`np.ndarray`); the actual conversion is delegated to `_materialize()`. | ✅ |
| `jax.Array`, `jax.Device` | `Array` type-checks `self.coeff` in `block_until_ready`; `Device` types the `devices()` return sets. | ✅ |
| `jaxtyping.ArrayLike`, `jaxtyping.PyTree` | `ArrayLike` types `coeff` and the module-level helpers' arguments; `PyTree` types `_devices`' argument (it walks an arbitrary pytree, not just a `CompositeTerm`). | ✅ |
| `qutip.Qobj` | Return-type annotation for `to_qutip` (still delegates entirely to `_materialize()`). | ✅ |
| `.._utils.is_batched_scalar` | The V1 guard on `coeff` (§4.1). | ✅ — new import, added for the fix. |
| `.dataarray.IndexType` | Only used by the still-stubbed `__getitem__` signatures. | ✅ |
| `.layout.Layout`, `.layout.dia` | `Layout` types every `.layout` property; `dia` is the sentinel `ndiags` compares against. | ✅ — `dia` is a new import; the pre-PR skeleton only imported `Layout`. |
| `.materialized_qarray.MaterializedQArray` | The leaf type every operator must be (V2 guard, §4.1), and the return type of every materializing method. | ✅ |
| `.qarray.QArray`, `.qarray.QArrayLike` | Base class for `CompositeQArray`; `QArrayLike` types the still-stubbed arithmetic argument. | ✅ |
| `.sparsedia_dataarray.SparseDIADataArray` | Only used inside `_offsets` to reach `.data.offsets` on a sparse operator. | ✅ — new import, needed for `ndiags`. |

**Nothing unused.** Every import is exercised by real PR2 code except the handful whose sole job is typing a stub signature (`Sequence`, `overload`, `IndexType`, `QArrayLike` in the stub bodies) — those are legitimately needed too, since a stub without a correct signature would itself be a defect once a later PR implements it.

---

## 2. Module-level helpers

### `_devices`

```python
def _devices(x: PyTree) -> set[Device]:
    # devices of every array leaf; tracers are skipped since they have no device
    return {
        device
        for leaf in jax.tree_util.tree_leaves(x)
        if isinstance(leaf, Array) and not isinstance(leaf, jax.core.Tracer)
        for device in leaf.devices()
    }
```

**Why it exists:** the single-device invariant (§4.1, §4.2) has to be checkable *while under `jit`/`grad` tracing*, because `__check_init__` runs on every construction, including inside a traced function. A tracer has no real device, so it's silently excluded rather than raising — an *empty* result from `_devices` is treated as "no violation" (`len(devices) > 1` is simply never true when `devices` is `{}` or a singleton).

**Verdict:** ✅ — this is a genuinely different animal from the public `devices()` method (§4.2/§5.2), which is *supposed* to refuse under tracing. `_devices` exists precisely so the *invariant* doesn't refuse under tracing while the *user-facing query* still does. Verified: construction and `to_jax()` both work fine inside `jax.jit`.

### `_coeff_batch_shape`

```python
def _coeff_batch_shape(coeff: ArrayLike) -> tuple[int, ...]:
    # batch shape of a batched scalar, i.e. `()`, `(1,)` or `(*batch, 1, 1)`
    coeff = jnp.asarray(coeff)
    return coeff.shape[:-2] if coeff.ndim >= 2 else ()
```

**Why it exists:** `is_batched_scalar` (the guard, §4.1) accepts three legal shapes for a scalar coefficient — `()`, `(1,)`, or `(*batch, 1, 1)`. Only the third carries a *batch* shape; this helper extracts it (or `()` for the other two), so `CompositeTerm.shape` can broadcast it against the operators' own batch shapes uniformly.

**Verdict:** ✅ — matches `is_batched_scalar`'s own three-way shape contract exactly (confirmed by reading both side by side); a `(1,)`-shaped coeff correctly contributes an *empty* batch shape (broadcast-neutral), not a spurious `(1,)` batch axis.

---

## 3. `CompositeTerm` — fields

```python
class CompositeTerm(eqx.Module):
    operators: tuple[MaterializedQArray, ...]
    coeff: ArrayLike = 1.0
```

**Verdict:** ✅ — `coeff` defaults to `1.0`, so `CompositeTerm(operators=(A, B))` is a valid, uncoefficiented separable term without forcing every call site to spell out `coeff=1.0`. The field *type annotations* (`tuple[MaterializedQArray, ...]`) are not runtime-enforced by `equinox` on their own — that's exactly why the V1/V2 guards below exist as explicit checks rather than relying on the annotation.

---

## 4. `CompositeTerm.__check_init__`

The full method, in the order it actually runs:

```python
def __check_init__(self):
    # ensure `operators` is a non-empty tuple
    if not isinstance(self.operators, tuple):
        raise TypeError(...)
    if len(self.operators) == 0:
        raise ValueError(...)

    # ensure every operator is a materialized qarray, as assumed by every
    # lazy formula that reaches into `op.data`, `op.dims`, etc.
    if not all(isinstance(op, MaterializedQArray) for op in self.operators):
        raise TypeError(...)

    # === ensure the operators are square, as assumed by every lazy formula
    if any(op.shape[-1] != op.shape[-2] for op in self.operators):
        raise ValueError(...)

    # === ensure `coeff` is a scalar or a batched scalar of shape (..., 1, 1),
    # as assumed by `shape`, `_materialize` and every lazy formula
    if not is_batched_scalar(self.coeff):
        raise ValueError(...)

    # === ensure a single layout and device, so that they can be reported
    # without materializing
    layouts = {op.layout for op in self.operators}
    if len(layouts) > 1:
        raise ValueError(...)
    devices = _devices(self)
    if len(devices) > 1:
        raise ValueError(...)
```

### 4.1 Check by check

| # | Check | Why | Verdict |
|---|---|---|---|
| 1 | `operators` is a `tuple` | Tuples are immutable and hashable-friendly, matching every other qarray collection field in this codebase (`terms`, `offsets`). A list would let a caller mutate the field after construction, bypassing `__check_init__` entirely. | ✅ |
| 2 | `operators` non-empty | An empty tensor product isn't a well-defined operator; there is no sensible `shape`/`dtype` for it. | ✅ |
| 3 | **(V2, new)** every element is a `MaterializedQArray` | Every downstream formula reaches into `.data`, `.dims`, `.layout` — attributes only `MaterializedQArray` guarantees. Before this fix, a raw JAX array or a nested `CompositeQArray` was silently accepted and failed later with an unrelated `AttributeError` far from the actual mistake. | ✅ — this is one of the two defects this PR was created to fix. Verified: `CompositeTerm(operators=(jnp.eye(2), op3))` now raises `TypeError` at construction; before, it raised `AttributeError: 'ArrayImpl' object has no attribute 'layout'` three calls later. |
| 4 | Every operator is square | `shape`, `_materialize`, `_offsets`, and every not-yet-implemented spectral method assume $A_k \in \mathbb{C}^{m_k\times m_k}$. A non-square factor has no eigenvalues, no trace in the usual sense, and no meaningful "number of subsystems it spans." | ✅ |
| 5 | **(V1, new)** `coeff` is a batched scalar | Without this, `CompositeTerm(operators=(A, B), coeff=jnp.ones(3))` was silently accepted and `shape` **lied**: it returned `(6, 6)` regardless of the malformed coeff, because `_coeff_batch_shape` only looks at `ndim >= 2`. | ✅ — the second defect this PR fixes. Verified via mutation testing: removing this check makes `test_rejects_invalid_construction[term-coeff-not-a-batched-scalar]` fail with `DID NOT RAISE ValueError`, confirming the test — and the guard — are both load-bearing. |
| 6 | Single layout across operators | `layout` (§5.3) reports `operators[0].layout` without inspecting the rest — sound only if this check holds. Mixing a dense and a DIA factor in one term has no single answer for "what layout is this term." | ✅ |
| 7 | Single device (`_devices`) | Same reasoning as layout, for `devices()`. | ✅ — though see the completeness note in §9: this branch is **not exercised by any test in this repo**, on either class, because CI never runs with more than one JAX device (confirmed: no `XLA_FLAGS`/forced multi-device setup anywhere in `.github/workflows/ci.yml`). The code is correct by inspection; it simply has zero automated coverage here, same as it would on `main`. |

### 4.2 Ordering

Checks are sequential early-exits, so only the *first* violated one ever fires. The chosen order (structural → type → shape → coeff → layout/device) means the cheapest, most fundamental checks run first and the most expensive one (`_devices`, which walks the whole pytree) runs last, after everything cheaper has already passed. This also means a test targeting check *N* must construct an input that is valid for checks *1..N-1* — which is why every parametrized case in `test_rejects_invalid_construction` reuses `_OP2`/`_OP3` (otherwise-valid fixtures) and perturbs exactly one field.

**Verdict:** ✅ — the ordering is a reasonable performance/clarity trade-off, not load-bearing for correctness (each check is independent of the others' outcomes).

---

## 5. `CompositeTerm` — implemented methods

### 5.1 `_materialize`

```python
def _materialize(self) -> MaterializedQArray:
    tensor_product = reduce(lambda x, y: x & y, self.operators)
    return cast(MaterializedQArray, tensor_product * self.coeff)
```

**Math:** $A_0 \otimes A_1 \otimes \cdots \otimes A_{N-1}$, scaled by $c$. `&` is the pre-existing (main-branch) Kronecker-product operator on `MaterializedQArray`; `*` is the pre-existing batched-scalar multiply.

**Why `cast`, not something else:** `reduce(&, operators)` folds left-to-right; `MaterializedQArray.__and__`'s return type is annotated `QArray` (loosely, same root cause as §7), so `tensor_product`'s static type is `QArray`, and `QArray.__mul__`'s return type is also `QArray`. The `cast` narrows the final expression to `MaterializedQArray`, which is what it always is at runtime — two `MaterializedQArray & MaterializedQArray` operations can only ever produce another `MaterializedQArray` (verified by the same exhaustive-branch reading used in §7, applied to `__and__`/`__mul__`). This `cast` is *not* covering an ambiguity; it's narrowing a return type that the base class's abstract signature necessarily can't state precisely for one subclass.

**Verdict:** ✅ — verified against an independent numpy oracle (`test_qarray_matches_independent_oracle`) with 2 operators per term, batched and unbatched, real and complex coefficients: exact match. Mutation-tested: dropping the `* self.coeff` multiply was caught by the same test (`np.allclose` failure), proving the coefficient scaling is actually exercised, not just present.

### 5.2 `_offsets`

```python
def _offsets(self) -> set[int]:
    # offsets of a Kronecker product combine as `o_0 * m_1 + o_1` (see
    # `and_sparsedia_sparsedia`); duplicates are merged, never dropped
    offsets = set(cast(SparseDIADataArray, self.operators[0].data).offsets)
    for op in self.operators[1:]:
        op_offsets = cast(SparseDIADataArray, op.data).offsets
        offsets = {o * op.shape[-1] + p for o in offsets for p in op_offsets}
    return offsets
```

**Math:** for a Kronecker product $A \otimes B$ with $B \in \mathbb{C}^{n\times n}$, a diagonal at offset $o$ in $A$ and offset $p$ in $B$ lands on offset $o\cdot n + p$ in $A\otimes B$. This is exactly the formula `and_sparsedia_sparsedia` (in `sparsedia_primitives.py`, pre-existing) uses to compute the offsets of a sparse Kronecker product — `_offsets` predicts the *same* set symbolically, without building any diagonal.

**Why the fold order matters:** the accumulation walks `operators` left-to-right, mirroring `reduce(&, operators)`'s own left-to-right fold in `_materialize`. If these two orders ever diverged, `_offsets` would predict offsets for a *different* factorization than the one `_materialize` actually builds.

**`cast(SparseDIADataArray, ...)`:** `op.data`'s static type is `DataArray` (the abstract base); this narrows it to the concrete sparse subtype that actually has an `.offsets` attribute. This is only ever called when `self.layout is dia` (guaranteed by the caller, `CompositeQArray.ndiags`, which checks the layout first) — so the cast reflects a real, checked precondition, not a guess.

**Verdict:** ✅ — hand-derived and verified: two DIA operators with offsets $\{0\}$ and $\{0,1\}$ (sizes 2 and 3) predict $\{0, 3\cdot0+1\} = \{0,1\}$; confirmed exactly by direct computation. Cross-checked against `_materialize().ndiags` (a completely independent code path — full densification then re-discovery of nonzero diagonals) on a 2-term composite: both report 3. Mutation-tested indirectly through `ndiags` (§6.2).

### 5.3 Properties: `dtype`, `shape`, `layout`, `mT`

```python
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
```

| Property | Choice | Verdict |
|---|---|---|
| `dtype` | Includes `self.coeff` in the `result_type` call, not just the operators. | ✅ — verified: a term with real `float32` operators and a `complex` coefficient correctly reports a complex dtype (oracle test's second term, `coeff1 = 0.5 + 0.5j`). Omitting `coeff` here would be the kind of bug that's invisible until someone multiplies by `1j`. |
| `shape` | Batch axes broadcast across *all* operators' own batch shapes *and* the coeff's batch shape; matrix size is the product of operator sizes (sound only because of the square-operator guard, §4.1#4). | ✅ — verified with two operators of *different* batch shapes (`(4,)` vs. unbatched) broadcasting correctly to `(4, 6, 6)`. |
| `layout` | `operators[0].layout`, not an aggregate. | ✅ — sound *only* because of the single-layout guard (§4.1#6); reads as a red flag in isolation, but the invariant is enforced at every construction site, including every `replace(...)` call (equinox re-runs `__check_init__`). |
| `mT` | $(A\otimes B)^\mathsf{T} = A^\mathsf{T}\otimes B^\mathsf{T}$ — a standard Kronecker identity, applied per-operator. Zero cost (no matrix ever built), no batch-shape realignment needed since `mT` never touches batch axes. | ✅ — verified against the oracle: `c.mT.to_jax()` matches `np.swapaxes(oracle, -1, -2)` exactly. |

### 5.4 `devices`, `block_until_ready`

```python
def devices(self) -> set[Device]:
    # delegate, so that a term reports the devices its materialized form would
    # and refuses the same way under tracing. `_devices` is for the invariants
    # that must hold while traced, not for this.
    return set().union(*(op.devices() for op in self.operators))

def block_until_ready(self) -> CompositeTerm:
    for op in self.operators:
        op.block_until_ready()
    if isinstance(self.coeff, Array):
        self.coeff.block_until_ready()
    return self
```

**Why `devices()` doesn't reuse `_devices(self)`:** deliberate, and stated in the comment. `_devices` is tolerant of tracers (needed so construction doesn't spuriously fail inside `jit`); the *public* `devices()` should behave like `MaterializedQArray.devices()` — which calls `.data.devices()`, and *does* raise `ConcretizationTypeError` under tracing, because asking "what physical device is this on" genuinely has no answer for a traced value. Delegating to each operator's own `devices()` inherits that refusal for free, rather than re-implementing it.

**`block_until_ready`'s `isinstance(self.coeff, Array)` guard:** `coeff` defaults to the Python float `1.0` and can be any `ArrayLike`; only a real JAX `Array` has `.block_until_ready()`. Skipping the check for a plain float is correct, not a shortcut — calling `.block_until_ready()` on a Python float would `AttributeError`.

**Verdict:** ✅ for both — the design intent (refuse under tracing) matches the equivalent `MaterializedQArray` behavior by construction, since it's a pure delegation.

### 5.5 `asdense`, `assparsedia`

```python
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
```

**The branch on `offsets`:** an explicit `offsets` tuple names diagonals of the *full* $D\times D$ matrix. There is no way to decompose "diagonal 7 of the 6×6 product" into "diagonal $x$ of the 2×2 factor and diagonal $y$ of the 3×3 factor" in general — the offset-combination formula in §5.2 runs in the *other* direction (factor offsets → product offset, not the reverse, and not injectively invertible). So this one call genuinely has no lazy implementation and must materialize; `offsets=None` (the default, auto-detect) stays fully lazy, per-operator.

**Verdict:** ✅ — this is a correct and *necessary* asymmetry, not an inconsistency. Verified in `test_layout_conversion_stays_lazy`: `offsets=None` returns type `CompositeQArray`/`CompositeTerm` (via the `CompositeQArray`-level wrapper) with matching values; explicit `offsets` returns `MaterializedQArray` with matching values against an independently materialized comparison.

---

## 6. `CompositeQArray.__check_init__`

```python
def __check_init__(self):
    super().__check_init__()

    if len(self.dims) < 2:
        raise ValueError(...)

    if not isinstance(self.terms, tuple):
        raise TypeError(...)
    if len(self.terms) == 0:
        raise ValueError(...)

    for j, term in enumerate(self.terms):
        term_dims = tuple(d for op in term.operators for d in op.dims)
        if term_dims != self.dims:
            raise ValueError(...)

    layouts = {term.layout for term in self.terms}
    if len(layouts) > 1:
        raise ValueError(...)
    devices = _devices(self)
    if len(devices) > 1:
        raise ValueError(...)
```

### 6.1 Check by check

| # | Check | Why | Verdict |
|---|---|---|---|
| 0 | `super().__check_init__()` | Re-validates the inherited `dims`-is-a-tuple-of-ints invariant from `QArray`. Easy to forget when overriding `__check_init__` in a subclass — omitting it wouldn't fail loudly, it would just silently stop enforcing a check that used to run. | ✅ — explicitly present. |
| 1 | `len(dims) < 2` | A single-subsystem "composite" is a contradiction in terms — that case is exactly what `MaterializedQArray` already represents. Rejecting it prevents the same operator from having two different valid representations in the codebase. | ✅ |
| 2 | `terms` is a non-empty tuple | Same reasoning as `CompositeTerm.operators` (§4.1#1–2), independently re-checked rather than shared, specifically so a copy-paste typo in one class's version (e.g. checking `self.operators` instead of `self.terms`) can't silently go untested. | ✅ |
| 3 | Each term's operators' dims concatenate to exactly `self.dims` | This is the check that ties a term's *factorization* to the declared Hilbert space, and it is **order-sensitive** (tuple equality, not set equality) — a term whose operators are ordered `(3,2)` when `dims=(2,3)` is rejected, correctly, since that would silently represent a *different* tensor-product ordering. | ✅ — verified via `qarray-dims-term-mismatch`. |
| 4 | Single layout across terms | Same reasoning as §4.1#6, one level up. | ✅ |
| 5 | Single device across terms | Same reasoning as §4.1#7, one level up — same completeness caveat (untested in this repo's CI, correct by inspection). | ✅ |

**Verdict on the whole method:** ✅ — every branch has a distinct, independently-verified test case (`test_rejects_invalid_construction`, 11 parametrized cases after dropping the 2 always-skipped device cases). Mutation-testing wasn't performed on every one of the 11 individually, but was performed on the two *new* checks (V1, V2) specifically, since those are the two genuinely new pieces of logic in this PR; the rest are ported verbatim from the reviewed real implementation.

### 6.2 `_materialize`, properties

```python
def _materialize(self) -> MaterializedQArray:
    return reduce(lambda x, y: x + y, (term._materialize() for term in self.terms))

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
        raise AttributeError(...)
    # the offsets of a sum are the union of the offsets of its terms (see
    # `add_sparsedia_sparsedia`), and no diagonal is ever dropped
    return len(set().union(*(term._offsets() for term in self.terms)))
```

**`_materialize`:** $H = \sum_j c_j\bigotimes_k A_{jk}$, i.e. sum of each term's own materialize. Correctness here rests entirely on `MaterializedQArray.__add__` already being correct (pre-existing, unmodified logic) — this method adds no new arithmetic, only orchestrates it.

**`shape`'s matrix dimensions:** taken from `shapes[0][-2:]` — the *first* term's matrix shape, not a computed aggregate. Sound only because §6.1#3 guarantees every term factorizes the *same* `dims`, hence the *same* matrix size; picking `shapes[0]` rather than re-deriving it from `dims` is a minor but correct shortcut.

**`ndiags`:** the one property with real, non-trivial lazy content — see §5.2 for the offset math. `set().union(*(...))` merges every term's offsets with no double-counting (Python set semantics), matching exactly how `add_sparsedia_sparsedia` (pre-existing, in `sparsedia_primitives.py`) merges diagonals of a materialized sum: duplicate offsets combine into one diagonal, they never multiply the count.

**Verdict:** ✅ across the board.
- `dtype`/`shape`/`mT`/`ndim` verified against the independent oracle with 2 terms of *different* batch shapes — a wrong per-term computation could not coincidentally produce the right broadcast result at the aggregate level (verified this reasoning empirically too: deliberately breaking `CompositeTerm._materialize`'s coefficient scaling was caught by the *aggregate*-level oracle test, confirming there's no hiding place for a per-term bug).
- `ndiags`: hand-derived expected value (3, from two DIA terms with offsets $\{0,1\}$ and $\{0,-1\}$) matched exactly, and cross-checked against `_materialize().ndiags`. Mutation-tested: replacing the union with `len(self.terms[0]._offsets())` was caught (`2 == 3` failure) — proving the test actually exercises the *merge*, not just "returns a number."
- `layout is not dia` uses `is`, not `==` — correct, since `Layout` values (`dia`, `dense`) are singleton sentinels (confirmed by reading `layout.py`), so identity comparison is the intended and slightly cheaper check, consistent with how the rest of the codebase compares layouts.

### 6.3 `devices`, `block_until_ready`, conversions, `__repr__`

```python
def devices(self) -> set[Device]:
    # see `CompositeTerm.devices`
    return set().union(*(term.devices() for term in self.terms))

def block_until_ready(self) -> QArray:
    for term in self.terms:
        term.block_until_ready()
    return self

def to_qutip(self) -> Qobj | list[Qobj]:
    return self._materialize().to_qutip()

def to_jax(self) -> Array:
    return self._materialize().to_jax()

def to_numpy(self) -> np.ndarray:
    return self._materialize().to_numpy()

def __array__(self, dtype=None, copy=None) -> np.ndarray:
    return self._materialize().__array__(dtype=dtype, copy=copy)

def asdense(self) -> QArray:
    return replace(self, terms=tuple(term.asdense() for term in self.terms))

def assparsedia(self, offsets: tuple[int, ...] | None = None) -> QArray:
    if offsets is not None:
        return self._materialize().assparsedia(offsets=offsets)
    return replace(self, terms=tuple(term.assparsedia() for term in self.terms))
```

**`devices`/`block_until_ready`:** same delegation pattern as `CompositeTerm`'s (§5.4), one level up — union of per-term devices, block each term in turn.

**Conversions (`to_qutip`/`to_jax`/`to_numpy`/`__array__`):** unconditional delegation to `_materialize()`. There is no lazy alternative here — a QuTiP `Qobj`, a JAX array, and a NumPy array are all, by definition, the concrete matrix. Nothing to second-guess.

**`asdense`/`assparsedia`:** exactly mirrors `CompositeTerm`'s own version (§5.5) one level up — per-term, stays lazy; explicit `offsets` still materializes for the same reason (full-matrix offsets don't decompose, this time across *terms and their operators* simultaneously).

**Verdict:** ✅ — verified: `dia` composite's `.asdense()` returns type `CompositeQArray` with `layout is dq.dense` and matching values; a dense composite's explicit-offsets `.assparsedia(offsets=...)` correctly returns `MaterializedQArray`, matching an independently materialized-then-converted comparison.

```python
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
```

**Design choice:** every field it prints — `shape`, `dims`, `dtype`, `layout`, and each term's `coeff`/`op_dims` — is $O(\text{term count})$ to compute, never touching the $D\times D$ matrix. This is the entire point of the class (avoid the exponential blow-up), and `__repr__` is exactly the place where it would be easiest to accidentally undo that by reaching for `self._materialize()` "just to print it nicely."

**Verdict:** ✅ — this was not taken on faith. Mutation-tested directly: `monkeypatch`ing `CompositeQArray._materialize` to raise `AssertionError` and then calling `repr(c)` still succeeds and prints the expected `CompositeQArray`/`n_terms=2`/`dims`/`shape` substrings. Also mutation-tested the *inverse*: inserting a `self._materialize()` call into `__repr__` was caught immediately (the monkeypatched assertion fired).

---

## 7. The `MaterializedQArray.__add__` return-type fix

Not part of `composite_qarray.py`, but part of this PR's diff, and directly caused by `CompositeQArray._materialize`'s `reduce`:

```python
# before
def __add__(self, y: QArrayLike) -> QArray:
    ...
# after
def __add__(self, y: QArrayLike) -> MaterializedQArray:
    ...
```

**Why this was necessary and why it's sound, not a workaround:** `functools.reduce(f, xs)` requires `f: (T, T) -> T`. With `__add__` annotated `-> QArray`, `reduce(lambda x, y: x + y, (term._materialize() for term in terms))` type-checks `f` as `(MaterializedQArray, MaterializedQArray) -> QArray` — a mismatch, since `_T` can't be both `MaterializedQArray` (the input) and `QArray` (the inferred output). Reading every branch of `__add__`'s actual body (the `y == 0` shortcut, the `NotImplemented` early-returns, and the final `replace(self, data=result)`) shows it **can only ever return** a `MaterializedQArray` or the sentinel `NotImplemented` — never any other `QArray` subtype. `-> QArray` was simply looser than reality, inherited unchanged from the abstract base class's necessarily-generic signature.

**Why not `cast()` or `# type: ignore` instead:** both were available and both would have "worked" locally, but neither would have been *true* — the annotation itself was wrong, not just unprovable to the checker at one call site. Fixing the annotation fixes the actual information the type system has about this method everywhere it's used, not just at this one `reduce` call.

**Verdict:** ✅, verified three ways:
1. `ty check dynamiqs` (the whole package, not just the changed files) drops from 2 diagnostics to 0, with no *new* diagnostics appearing anywhere — meaning no other call site was relying on the wider, less accurate annotation.
2. `ruff check .` (whole repo) stays clean.
3. Full `tests/qarrays/` suite (220 tests) still passes — expected, since narrowing a type annotation with `from __future__ import annotations` active is mathematically incapable of changing runtime behavior; this run is a sanity check on everything *else*, not on this change specifically.

---

## 8. The three added abstract-method stubs

`swapaxes`, `moveaxis`, `expand_dims` were **entirely absent** from `origin/main`'s pre-existing skeleton — not `raise NotImplementedError` bodies, just missing. Since `QArray` is a real `ABCMeta`-based abstract class, this meant `CompositeQArray` could not be **instantiated at all** on `main` today:

```
TypeError: Can't instantiate abstract class CompositeQArray with abstract methods
expand_dims, moveaxis, ndiags, swapaxes
```

`ndiags` is genuine PR2 content (§6.2) and was implemented for real. `swapaxes`/`moveaxis`/`expand_dims` are PR3's ("batch-axis machinery") — this PR only adds bare stubs, in the exact style already used by every other not-yet-implemented method in the file:

```python
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
```

**Verdict:** ✅ as a minimal, honest fix to an *unrelated, pre-existing* skeleton defect — not scope creep, since without it, nothing in PR2 (or any later PR) could be tested at all. Correctly scoped: no logic invented, no attempt made to guess at PR3's actual implementation. Verified: `CompositeQArray(...)` now instantiates; calling any of the three stubs raises `NotImplementedError` cleanly (the same failure mode as every other not-yet-landed method in the file — `c.conj()`, `c.trace()`, etc. — not a new or different kind of failure).

**One completeness gap worth flagging explicitly:** this fix is real code shipped in this PR, but it isn't mentioned anywhere in the class's own docstring or comments as "these three are missing on purpose, not forgotten again." A future contributor reading only the diff for PR3 might not realize these three stubs' presence is itself a fix that predates PR3's actual content. Worth one sentence in the eventual PR description (already planned) and arguably a one-line code comment at the top of the file — this doc is the only place that currently explains it.

---

## 9. Overall assessment

### Correctness

Every implemented method and every check was verified against at least one of: an independent (hand-computed or numpy-only) oracle, a hand-derived expected value, or a direct cross-check against a structurally unrelated code path (e.g. `ndiags` vs. `_materialize().ndiags`). Four of the five test functions were additionally **mutation-tested** — the implementation was deliberately broken in a targeted way and the test suite confirmed it caught the break, then the break was reverted. No test in this PR is tautological; each one has been shown to fail when the code it covers is wrong.

The two defects this PR exists to fix (V1: unvalidated `coeff`; V2: unvalidated `operators` element type) are both fixed and both individually mutation-tested.

One additional, unrelated defect was found and fixed as a prerequisite: `CompositeQArray` could not be instantiated at all on `main` (§8) — without noticing and fixing this, none of PR2's own work could have been verified.

### Completeness

Within its stated scope (construct/inspect/materialize), coverage is complete: every guard branch in both `__check_init__` methods has a dedicated test, except the two single-device checks, which are **untestable in this repository's current CI** (no environment here or in `.github/workflows/ci.yml` ever exposes more than one JAX device) rather than untested by choice. Every implemented property, every conversion method, and `__repr__`'s non-materializing guarantee are all exercised.

What's explicitly *not* covered, correctly, because it's out of scope: every method still `raise NotImplementedError` (`conj`, `reshape`, `broadcast_to`, `swapaxes`, `moveaxis`, `expand_dims`, `powm`, `expm`, `norm`, `trace`, `sum`, `squeeze`, the four spectral methods, `isherm`, `ptrace`, `__getitem__`, and all of `CompositeTerm`'s and `CompositeQArray`'s arithmetic dunders). These are PR3/PR4/PR5 content and are untouched here.

### Quality of the choices themselves

- The decision to duplicate the tuple/empty/type structural checks across `CompositeTerm` and `CompositeQArray` rather than sharing one helper is a deliberate trade (discussed and agreed before writing the tests): it costs a few lines of near-identical code, but it means a future copy-paste-with-a-typo bug in one class's version is independently caught rather than silently inherited from a shared helper that was only ever tested once.
- `_devices` vs. `devices()` being two genuinely different functions with different tracing behavior, rather than one function used two ways, is the right call — conflating them would either break construction under `jit` or let `devices()` silently misreport under tracing.
- The `MaterializedQArray.__add__` annotation fix (§7) is the kind of fix that's easy to skip ("it's just a type checker complaint") but is worth doing exactly because the *old* annotation was actively wrong, not merely imprecise — narrowing it made the codebase's stated contract match its real behavior, which is strictly better information for every future caller and every future type-checker run, not just this one `reduce` call.
- The one place a genuine trade-off exists — testing the two device-uniqueness checks at all, given they're provably dead in this repo's CI — was resolved by cutting them rather than keeping dead, never-run test cases; that's documented here rather than silently done, so the coverage gap is a known, deliberate one rather than an accidental one.
