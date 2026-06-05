"""Utilitary methods to extract information from Result object's `infos` attribute."""

from dynamiqs.method import Method, _DEAdaptiveStep
from dynamiqs.result import Result

from .structures import MethodParameters
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


def extract_method_parameters(method: Method) -> MethodParameters:
    """Extract the solver method metadata from a Method object.

    Params:
        method: A Method

    Returns:
        A record containing of the name of the method along with its configuration
        parameters.
    """
    method_name = type(method).__name__
    method_rtol = -1.0
    method_atol = -1.0
    method_factor_safety = -1.0
    method_factor_min = -1.0
    method_factor_max = -1.0

    if isinstance(method, _DEAdaptiveStep):
        method_atol = maybe_unpack(method.atol)
        method_rtol = maybe_unpack(method.rtol)
        method_factor_safety = maybe_unpack(method.safety_factor)
        method_factor_min = maybe_unpack(method.min_factor)
        method_factor_max = maybe_unpack(method.max_factor)

    return {
        'method_name': method_name,
        'method_atol': method_atol,
        'method_rtol': method_rtol,
        'method_factor_min': method_factor_min,
        'method_factor_max': method_factor_max,
        'method_factor_safety': method_factor_safety,
    }


__all__ = ['extract_method_parameters', 'extract_nsteps']
