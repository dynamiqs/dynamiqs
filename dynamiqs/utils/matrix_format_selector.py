import functools as ft
from enum import Enum
from typing import Callable, Literal

__all__ = [
    "dia",
    "dense",
    "set_matrix_format",
]


class MatrixFormatEnum(Enum):
    DENSE = "dense"
    SPARSE_DIA = "sparse_dia"


dia = MatrixFormatEnum.SPARSE_DIA
dense = MatrixFormatEnum.DENSE

MatrixFormat = Literal["dense", "sparse_dia"]

global_matrix_format = None
DEFAULT_MATRIX_FORMAT = MatrixFormatEnum.DENSE

dispatch_dict = {}


def dispatch_matrix_format(func: Callable) -> Callable:
    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        global global_matrix_format  # noqa: PLW0602
        matrix_format = kwargs.pop("matrix_format", None)
        matrix_format = matrix_format or global_matrix_format or DEFAULT_MATRIX_FORMAT

        key = (func.__name__, matrix_format)
        if key not in dispatch_dict:
            handlers_list = "\n- ".join(list(map(str, dispatch_dict.keys())))
            raise RuntimeError(
                f"There is no handler for method '{func.__name__}' "
                f"and matrix format '{matrix_format}'.\nRegistered handlers "
                f"are \n- {handlers_list}\n"
                f"This error should never happen, if you encounter it, please "
                f"open a ticket at https://github.com/dynamiqs/dynamiqs/issues."
            )

        return dispatch_dict[key](*args, **kwargs)

    return wrapper


def register_format_handler(
    function_name: str, matrix_format: MatrixFormatEnum
) -> Callable:
    def wrapper(func: Callable) -> Callable:
        dispatch_dict[(function_name, matrix_format)] = func
        return func

    return wrapper


def set_matrix_format(
    matrix_format: Literal[MatrixFormatEnum.DENSE, MatrixFormatEnum.SPARSE_DIA],
):
    global global_matrix_format  # noqa: PLW0603
    accepted_values = set(MatrixFormatEnum)
    if matrix_format in accepted_values:
        global_matrix_format = matrix_format
    else:
        raise ValueError(
            f"Format should be one of {accepted_values}, got '{matrix_format}'"
        )
