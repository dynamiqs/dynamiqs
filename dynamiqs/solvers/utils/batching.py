from torch import Tensor

from .td_tensor import TDTensor


def batch_H(H: TDTensor) -> TDTensor:
    # handle H batching
    # x: (b_H?, n, n) ->  (b_H, 1, 1, n, n)
    if H.dim() == 2:  # (n, n)
        H = H.unsqueeze(0)  # (1, n, n)
    H = H.unsqueeze(1)
    H = H.unsqueeze(1)
    return H


def batch_jump_ops(jump_ops: Tensor) -> Tensor:
    # handle jump_ops batching
    # jump_ops: (n_jump_ops, b_jump_ops?, m, n) ->  (n_jump_ops, 1, b_jump_ops, 1, n, n)
    if jump_ops.dim() == 3:
        jump_ops = jump_ops.unsqueeze(1)  # (n_jump_ops, b_jump_ops, n, n)
    jump_ops = jump_ops.unsqueeze(-3)  # (n_jump_ops, b_jump_ops, 1, n, n)
    jump_ops = jump_ops.unsqueeze(1)  # (n_jump_ops, 1, b_jump_ops, 1, n, n)

    return jump_ops


def batch_y0(y0: Tensor, H: TDTensor, jump_ops: Tensor = None) -> Tensor:
    # handle y0 batching
    # y0: (b_y0?, m, n) ->  (1, 1, b_y0, n, n)
    if y0.ndim == 2:  # (n, n)
        y0 = y0.unsqueeze(0)  # (1, n, n)
    y0 = y0.unsqueeze(0)  # (1, 1, n, n)
    y0 = y0.unsqueeze(0)  # (1, 1, 1, n, n)

    if jump_ops is None:
        jump_ops_repeat = 1
    else:
        jump_ops_repeat = jump_ops.size(2)

    return y0.repeat(H.size(0), jump_ops_repeat, 1, 1, 1)
