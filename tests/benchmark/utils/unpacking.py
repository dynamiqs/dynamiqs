"""Utilitary methods to extract values from wrappers."""

import jax


# FIXME: typing is use case-specific, restraining generic usage. it should be
#        improved to adopt a more general type structure
def maybe_unpack(value: float | jax.Array) -> float:
    """If the value is stored in a Jax array, unpack it. Return the value afterwards.

    Params:
        value: Any value that may be packed in a array.

    Returns:
        The value, unpacked if it was stored in an array.
    """
    if isinstance(value, jax.Array):
        return value.item()

    return value


__all__ = ['maybe_unpack']
