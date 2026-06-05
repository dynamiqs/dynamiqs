"""Utilitary methods to extract information from Result object's `infos` attribute."""

from dynamiqs.result import Result

from .unpacking import maybe_unpack


def extract_nsteps(result: Result) -> int:
    """Extract the number of steps a solving result has taken.

    If the value is not available, return -1.

    Params
        result: A solving result.

    Returns:
        The number of steps of the solving solution if it is available. -1 otherwise.
    """
    value = getattr(result.infos, 'nsteps', -1)

    nsteps = maybe_unpack(value)

    if not isinstance(nsteps, int):
        return -1

    return nsteps


__all__ = ['extract_nsteps']
