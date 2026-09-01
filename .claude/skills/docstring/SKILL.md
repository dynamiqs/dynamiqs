---
name: docstring
description: Write docstrings for Dynamiqs functions, classes and methods following the Dynamiqs conventions (Google-style with mkdocs quirks, KaTeX math, shape annotations, doctested examples). Use when writing or updating docstrings in Dynamiqs code.
---

# Dynamiqs Docstring Writing Guide

Dynamiqs docstrings are rendered into the documentation website by
[mkdocstrings](https://mkdocstrings.github.io/) and **executed as tests** by
[sybil](https://sybil.readthedocs.io/) (`task doctest-code`). A docstring is therefore
both documentation and a test — a wrong example output fails CI.

The conventions are Google-style with several Dynamiqs-specific quirks. Follow the
existing docstrings in `dynamiqs/utils/general.py`, `dynamiqs/utils/operators.py`, and
`dynamiqs/integrators/apis/mesolve.py` — they are the reference.

## General principles

- Use **raw strings** (`r"""..."""`) whenever the docstring contains LaTeX. In practice:
  always, for public functions.
- Headers, **in this order and with these exact names**: `Args`, `Returns`, `Raises`,
  `Examples`, `See also`.
- Admonitions available: `Note`, `Warning` (and their collapsed variants, below).
- Be concise. The website shows this text next to the signature; it is not a tutorial.
- Every public function gets at least one runnable example.

## Structure

### 1. Summary line

One line, imperative or third-person-singular, ending with a period. No signature
repetition (mkdocstrings renders the real signature from the annotations).

```python
def dag(x: QArrayLike) -> QArray:
    r"""Returns the adjoint (complex conjugate transpose) of a matrix."""
```

### 2. Extended description with math

Math is rendered with KaTeX. Use `$...$` inline and `$$...$$` for display math — **not**
Sphinx `:math:` roles, which do not exist here.

```python
    r"""Returns the norm of a ket, bra, density matrix, or Hermitian matrix.

    For a ket or a bra, the returned norm is $\sqrt{\braket{\psi|\psi}}$. For a
    Hermitian matrix, the returned norm is the trace norm defined by:
    $$
        \\|A\\|_1 = \tr{\sqrt{A^\dag A}} = \sum_i |\lambda_i|
    $$
    where $\lambda_i$ are the eigenvalues of $A$.
    """
```

`docs/javascripts/katex.js` defines the project macros — `\dag`, `\dd`, `\dt`, `\tr{}`,
`\kett{}` — on top of KaTeX built-ins such as `\braket{}`. Reuse the notation of
neighbouring docstrings; if you need a new macro, add it there rather than inlining a
one-off expansion.

For long equations with per-symbol commentary, the annotation extension is available —
see `dq.mesolve()`:

```
    equation
    { .annotate }

    1. With explicit time dependence:
        - $\rho\to\rho(t)$
        - $H\to H(t)$
```

### 3. Args

```python
    Args:
        x (qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n)): Ket, bra,
            density matrix, or Hermitian matrix.
        psd: Whether `x` is a positive semi-definite matrix. If `True`, returns the
            trace of `x`, otherwise computes the eigenvalues of `x`.
        *dims: Hilbert space dimension of each subsystem.
```

Rules:

- **Only add a type in parentheses when it carries information the signature does not.**
  The usual reason is a shape: `(qarray-like of shape (..., n, n))`. If the annotation
  is already clear (`psd: bool = False`), omit the type entirely.
- Shape vocabulary: `...` for batch dimensions, `n` for the Hilbert space dimension.
  Spell the type in lower case: `qarray-like`, `qarray`, `array`, not `QArray`.
- **Do not start a description with "The"**: write `x: Quantum state.`, not
  `x: The quantum state.`
- Mention the default in the prose only when it is non-obvious; mkdocstrings already
  renders defaults from the signature.
- Continuation lines are indented by 4 spaces.

### 4. Returns

```python
    Returns:
        (qarray of shape (..., n, m)): Adjoint of `x`.

    Returns:
        (array of shape (...)): Real-valued norm of `x`.

    Returns:
        (qarray of shape (n, n)): Identity operator, with _n = prod(dims)_.
```

The parenthesized type-and-shape comes first, then the description. Italics with
`_..._` for inline symbolic notes.

### 5. Admonitions

Two flavors, and the difference matters:

```python
    Note: Some title
        Rendered as an *open* admonition.

    Note-: Some title
        Rendered as a *collapsed* admonition — the trailing `-` closes it by default.
```

Use the collapsed form (`Note-:`, `Warning-:`) for asides most readers can skip, such as
equivalent syntax:

```python
    Note-: Equivalent syntax
        This function is equivalent to `x.mT.conj()`.
```

Use `Warning:` for numerical caveats, differentiability restrictions, and behavior that
will surprise a physicist (unit conventions, $\hbar=1$, normalization).

### 6. Examples — these are tests

Examples run under sybil in a namespace that already has `dq`, `np`, `plt`, `jax`,
`jnp`, and `qt` imported. **Never write `import dynamiqs as dq` in an example.**

```python
    Examples:
        Single-mode $I_4$:
        >>> dq.eye(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]
```

Rules:

- Use the `Examples:` header (Google style), **not** Sphinx's `Examples::`.
- Expected output must match exactly what the current code prints. Get it by running the
  snippet, not by writing it from memory — the `QArray` repr includes `layout`, `dims`
  and `ndiags`, and the printing options are `precision=3, suppress=True`.
- JAX defaults to float32/complex64. Outputs show `dtype=complex64` and 3 decimals.
- `...` is enabled as a doctest ellipsis (`optionflags=ELLIPSIS`); use it for volatile
  parts of the output (timings, addresses, long arrays).
- Short prose lines between snippets are encouraged to say what each block shows.
- For examples producing a figure, use the `renderfig` fixture pattern used in the plot
  docstrings.
- After writing examples, run them **on the file you touched only** — sybil collects
  docstring examples per file, so point pytest at the module:
  `uv run pytest dynamiqs/utils/general.py -q`. Do not run `task doctest-code` (the whole
  suite) unless the user asks; that is CI's job.

### 7. See also

Cross-references use mkdocs syntax, not Sphinx roles:

```python
    See also:
        - [dq.unit()][dynamiqs.unit]: normalize a quantum state.
        - [dq.Options][dynamiqs.Options]: solver options.
```

- Function: `[dq.sesolve()][dynamiqs.sesolve]` — keep the explicit `()` in the label.
- Class: `[dq.Options][dynamiqs.Options]` — no parentheses.
- Documentation page: `(doc page)(relative/path/to/file.md)` — parentheses, not brackets.

## Complete example

Illustrative (not the current text of `dq.unit()`), showing every section in order:

```python
def unit(x: QArrayLike, *, psd: bool = False) -> QArray:
    r"""Normalize a ket, bra, density matrix or Hermitian matrix to unit norm.

    The returned object is divided by its norm $\|x\|$, see [dq.norm()][dynamiqs.norm].

    Args:
        x (qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n)): Ket, bra,
            density matrix, or Hermitian matrix.
        psd: Whether `x` is a positive semi-definite matrix.

    Returns:
        (qarray of shape (..., n, 1) or (..., 1, n) or (..., n, n)): Normalized ket,
            bra, density matrix or Hermitian matrix.

    Warning:
        The norm is computed in complex64 by default, so the result is unit-normalized
        only to ~1e-6.

    See also:
        - [dq.norm()][dynamiqs.norm]: returns the norm of a quantum state.

    Examples:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> dq.norm(dq.unit(psi))
        Array(1., dtype=float32)
    """
```

## Documenting classes and methods

- `merge_init_into_class: true` is set in `mkdocs.yml`: document constructor arguments in
  the **class** docstring's `Args` section, not in `__init__`.
- `members_order: source`: the rendering order follows the source order, so order methods
  in the file the way you want them read.
- Private helpers (leading `_`) are not rendered; a one-line comment is usually enough
  for them, and ruff's `D1xx` rules for missing docstrings are disabled.

## Checklist

- [ ] Raw string `r"""`
- [ ] One-line summary ending with a period
- [ ] Math in `$...$` / `$$...$$`, not `:math:`
- [ ] `Args` / `Returns` / `Raises` / `Examples` / `See also`, in that order
- [ ] Shapes documented with `...` for batching and `n` for the Hilbert dimension
- [ ] No descriptions starting with "The"
- [ ] Types in parentheses only where they add information
- [ ] At least one example, with output copied from an actual run
- [ ] No `import` lines in examples (`dq`, `np`, `jnp`, `jax`, `plt`, `qt` are provided)
- [ ] Cross-references use `[dq.f()][dynamiqs.f]` syntax
- [ ] `uv run pytest <the module you edited> -q` passes
- [ ] `uv run task docserve` renders the page correctly (check math and admonitions)
- [ ] For a *new* public function: added to `__all__`, `mkdocs.yml`, and
      `docs/python_api/index.md`
