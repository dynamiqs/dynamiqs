---
name: docstring
description: Write docstrings for Dynamiqs functions, classes and methods following the Dynamiqs conventions (Google-style with mkdocs quirks, KaTeX math, shape annotations, doctested examples). Use when writing or updating docstrings in Dynamiqs code.
---

# Write a Dynamiqs docstring

Dynamiqs docstrings are rendered by mkdocstrings and **executed as tests** by sybil, so
a docstring is both documentation and a test: a wrong example output fails CI. The style
is Google-style with Dynamiqs-specific quirks. The reference docstrings are in
`dynamiqs/utils/general.py`, `dynamiqs/utils/operators.py`, and
`dynamiqs/integrators/apis/mesolve.py`.

## Workflow

1. Read the docstrings of neighbouring functions and reuse their notation.
2. Write the docstring following the template below.
3. Run the examples of the file you touched, and paste the real output:
   `uv run pytest dynamiqs/<module>.py -q`.
4. For math, admonitions, or a new page, check the rendering with `uv run task
   docserve`.
5. For a *new* public function, also add it to `__all__`, `mkdocs.yml`, and
   `docs/python_api/index.md`.

## Template

Every section, in the required order (illustrative, not the current `dq.unit()`):

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

    Examples:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> dq.norm(dq.unit(psi))
        Array(1., dtype=float32)

    See also:
        - [dq.norm()][dynamiqs.norm]: returns the norm of a quantum state.
    """
```

## Sections

**Summary line.** One line ending with a period, without repeating the signature
(mkdocstrings renders it from the annotations).

**Description and math.** Rendered with KaTeX: `$...$` inline, `$$...$$` for display
math. Project macros (`\dag`, `\dd`, `\dt`, `\tr{}`, `\kett{}`) live in
`docs/javascripts/katex.js`; add a new macro there rather than inlining a one-off
expansion. For long equations with per-symbol commentary, use the `{ .annotate }`
extension as in `dq.mesolve()`.

**Args.**
- Add a type in parentheses only when it carries information the signature does not,
  usually a shape: `x (qarray-like of shape (..., n, n)): ...`. Otherwise omit it.
- Shapes use `...` for batch dimensions and `n` for the Hilbert dimension; types are
  lower case (`qarray-like`, `qarray`, `array`).
- Mention a default only when non-obvious; mkdocstrings renders the signature defaults.
- Indent continuation lines by 4 spaces.

**Returns.** The parenthesized type and shape first, then the description:
`(qarray of shape (n, n)): Identity operator, with _n = prod(dims)_.` Use `_..._` for
inline symbolic notes.

**Admonitions.** `Note:` and `Warning:` render open; `Note-:` and `Warning-:` render
collapsed. Collapse asides most readers can skip (`Note-: Equivalent syntax`). Use
`Warning:` for numerical caveats, differentiability restrictions, and conventions that
surprise a physicist ($\hbar=1$, normalization).

**Examples.** Every public function has at least one.
- The sybil namespace already provides `dq`, `np`, `jnp`, `jax`, `plt`, and `qt`.
- Copy the output from an actual run: the `QArray` repr includes `layout`, `dims` and
  `ndiags`, printing uses `precision=3, suppress=True`, and dtypes are
  `float32`/`complex64`.
- `...` is a doctest ellipsis; use it for volatile output.
- A short prose line before a snippet says what it shows.
- For examples producing a figure, follow the `renderfig` pattern of the plot
  docstrings.

**See also.** mkdocs cross-reference syntax: `[dq.sesolve()][dynamiqs.sesolve]` for a
function (keep the `()` in the label), `[dq.Options][dynamiqs.Options]` for a class, and
`(doc page)(relative/path.md)` for a documentation page.

**Classes.** `merge_init_into_class` is on: document constructor arguments in the class
docstring's `Args`, not in `__init__`. Members render in source order, so order methods
as they should be read. Private helpers (leading `_`) are not rendered and need at most
a one-line comment.

## Gotchas

- Use a raw string `r"""` whenever there is LaTeX, which in practice means every public
  function.
- Section headers are exactly `Args`, `Returns`, `Raises`, `Examples`, and `See also`,
  in that order; admonitions go before `Examples`.
- Use `Examples:`, not Sphinx's `Examples::`; use `$...$`, not `:math:`; use
  `[dq.f()][dynamiqs.f]`, not `:func:`.
- Never write `import dynamiqs as dq` or any other import in an example.
- Never start an argument description with "The": `x: Quantum state.`, not
  `x: The quantum state.`
- Run only the touched module's examples; `task doctest-code` runs the whole suite and
  is CI's job.
- Keep it concise: the website shows it next to the signature, it is not a tutorial.
