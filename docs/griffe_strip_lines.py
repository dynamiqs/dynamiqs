"""Griffe extension that hides doctest-only lines from the rendered docstrings.

Docstring examples call `renderfig`, `rendergif` or `default_mpl_style` to generate
the figures shown in the documentation, and use `[...]` to elide doctest outputs. These
lines are needed by the doctests but are noise for the reader, so they are removed from
the docstrings before mkdocstrings renders them.
"""

import re
from typing import Any

import griffe

# match any line containing 'renderfig', 'rendergif', 'default_mpl_style' or '[...]',
# including its trailing newline
pattern = re.compile(
    r'^.*(renderfig|rendergif|default\_mpl\_style|\[\.\.\.\]).*$\n?', flags=re.MULTILINE
)


class StripDoctestLines(griffe.Extension):
    def on_instance(self, *, obj: griffe.Object, **kwargs: Any) -> None:  # noqa: ARG002
        if obj.docstring is not None:
            obj.docstring.value = pattern.sub('', obj.docstring.value)
