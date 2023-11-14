from typing import List

import torch
from torch import Tensor

from ...utils.tensor_types import ArrayLike, to_tensor
from .td_tensor import TDTensor


def batch_H(H: TDTensor) -> TDTensor:
    # handle H batching
    # x: (b_H?, n, n) ->  (b_H, 1, 1, n, n)
    if H.dim() == 2:  # (n, n)
        H = H.unsqueeze(0)  # (1, n, n)
    H = H.unsqueeze(1)
    H = H.unsqueeze(1)
    return H


def batch_jump_ops(jump_ops: List[ArrayLike], dtype=None, device=None) -> Tensor:
    # check all jump ops are batched in a similar way or not at all
    batched_jump_ops, not_batched_jump_ops = [], []

    for jump_op in jump_ops:
        jump_op = to_tensor(jump_op, dtype=dtype, device=device)
        if jump_op.dim() == 3:
            if jump_op.shape[0] == 1:  # allow a jump operator of shape (1, n, n))
                not_batched_jump_ops.append(jump_op.squeeze(0))
            else:
                batched_jump_ops.append(jump_op)
        elif jump_op.dim() == 2:
            not_batched_jump_ops.append(jump_op)
        else:
            raise ValueError(
                "All jump operators must have 2 dimensions or 3 dimensions if batched,"
                f" but a jump operator had {jump_op.ndim} dimensions with shape"
                f" {jump_op.shape}."
            )

    # check all batching are the same for batched jump_ops
    batched_jump_ops_shapes = set(map(lambda x: x.shape, batched_jump_ops))
    if len(batched_jump_ops_shapes) > 1:
        raise ValueError(
            "All batched jump operators (of dimension 3) must have the same shape, but"
            f" got shapes {batched_jump_ops_shapes}"
        )

    # batch all un-batched jump operators if necessary
    if len(batched_jump_ops) > 0:
        b = batched_jump_ops[0].shape[0]
        for jump_op in not_batched_jump_ops:
            jump_op = jump_op.repeat(b, 1, 1)
            batched_jump_ops.append(jump_op)
        jump_ops = batched_jump_ops

    jump_ops = torch.stack(jump_ops)
    if jump_ops.dim() == 3:  # (n_jump_ops, n, n)
        return jump_ops[:, None, None, None, ...]  # (n_jump_ops, 1, 1, 1, n, n)
    elif jump_ops.dim() == 4:  # (n_jump_ops, b_jump_ops, n, n)
        return jump_ops[:, None, :, None, ...]  # (n_jump_ops, 1, b_jump_ops, 1, n, n)
    else:
        raise ValueError(
            "Expected `jump_ops` of dimension 2 or dimension 3 if batched, got"
            f" {jump_ops.ndim} dimensions with shape {jump_ops.shape}."
        )


def batch_y0(y0: Tensor, H: TDTensor, jump_ops: Tensor = None) -> Tensor:
    # handle y0 batching
    # y0: (b_y0?, m, n) ->  (1, 1, b_y0, n, n)

    if y0.dim() == 2:  # (n, n)
        y0 = y0[None, None, None, ...]  # (1, 1, 1, n, n)
    elif y0.dim() == 3:  # (b_y0, n, n)
        y0 = y0[None, None, ...]  # (1, 1, b_y0, n, n)
    else:
        raise ValueError(
            "Expected `y0` of dimension 2 or dimension 3 if batched, got"
            f" {y0.ndim} dimensions with shape {y0.shape}."
        )

    if jump_ops is None:
        jump_ops_repeat = 1
    else:
        jump_ops_repeat = jump_ops.size(2)

    return y0.repeat(H.size(0), jump_ops_repeat, 1, 1, 1)
