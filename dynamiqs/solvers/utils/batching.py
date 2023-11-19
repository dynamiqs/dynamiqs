from __future__ import annotations

import torch
from torch import Tensor

from ...utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from .td_tensor import TDTensor, to_td_tensor


def batch_H(
    H: TDArrayLike,
    dtype: torch.complex64 | torch.complex128,
    device: torch.device,
) -> TDTensor:
    # H: (b_H?, n, n) ->  (b_H, 1, 1, n, n)

    # convert to TDTensor
    H = to_td_tensor(H, dtype=dtype, device=device)

    # handle H batching
    if H.ndim == 2:  # (n, n)
        H = H.unsqueeze(0)  # (b_H, n, n)
    H = H.unsqueeze(1)  # (b_H, 1, n, n)
    H = H.unsqueeze(1)  # (b_H, 1, 1, n, n)
    return H


def batch_jump_ops(
    jump_ops: list[ArrayLike],
    dtype: torch.complex64 | torch.complex128,
    device: torch.device,
) -> Tensor:
    # L: [(b_L?, n, n)] ->  (len(L), 1, b_L, 1, n, n)

    # check all jump ops are batched in a similar way or not at all
    batched_jump_ops, not_batched_jump_ops = [], []

    for jump_op in jump_ops:
        jump_op = to_tensor(jump_op, dtype=dtype, device=device)
        if jump_op.ndim == 3:
            if jump_op.shape[0] == 1:  # allow a jump operator of shape (1, n, n))
                not_batched_jump_ops.append(jump_op.squeeze(0))
            else:
                batched_jump_ops.append(jump_op)
        elif jump_op.ndim == 2:
            not_batched_jump_ops.append(jump_op)
        else:
            raise ValueError(
                'All jump operators must have 2 dimensions or 3 dimensions if batched,'
                f' but a jump operator had {jump_op.ndim} dimensions with shape'
                f' {jump_op.shape}.'
            )

    # check all batching are the same for batched jump_ops
    batched_jump_ops_shapes = set(map(lambda x: x.shape, batched_jump_ops))
    if len(batched_jump_ops_shapes) > 1:
        raise ValueError(
            'All batched jump operators (of dimension 3) must have the same shape, but'
            f' got shapes {batched_jump_ops_shapes}'
        )

    # batch all un-batched jump operators if necessary
    if len(batched_jump_ops) > 0:
        b = batched_jump_ops[0].shape[0]
        for jump_op in not_batched_jump_ops:
            jump_op = jump_op.repeat(b, 1, 1)
            batched_jump_ops.append(jump_op)
        jump_ops = batched_jump_ops
    else:
        jump_ops = not_batched_jump_ops

    jump_ops = torch.stack(jump_ops)  # (len(L), b_L?, n, n)
    if jump_ops.ndim == 3:  # (len(L), n, n)
        return jump_ops[:, None, None, None, ...]  # (len(L), 1, 1, 1, n, n)
    elif jump_ops.ndim == 4:  # (len(L), b_L, n, n)
        return jump_ops[:, None, :, None, ...]  # (len(L), 1, b_L, 1, n, n)
    else:
        raise ValueError(
            'Expected `jump_ops` of dimension 2 or dimension 3 if batched, got'
            f' {jump_ops.ndim} dimensions with shape {jump_ops.shape}.'
        )


def batch_y0(
    y0: ArrayLike,
    dtype: torch.complex64 | torch.complex128,
    device: torch.device,
    b_H: int = 1,
    b_L: int = 1,
) -> Tensor:
    # y0: (b_y0?, m, n) ->  (b_H, b_L, b_y0, m, n)

    # convert to Tensor
    y0 = to_tensor(y0, dtype=dtype, device=device)

    # handle batching
    if y0.ndim == 2:  # (m, n)
        y0 = y0[None, ...]  # (b_y0, m, n)
    elif y0.ndim == 3:  # (b_y0, m, n)
        pass
    else:
        raise ValueError(
            'Expected `y0` of dimension 2 or dimension 3 if batched, got'
            f' {y0.ndim} dimensions with shape {y0.shape}.'
        )

    return y0.repeat(b_H, b_L, 1, 1, 1)  # (b_H, b_L, b_y0, m, n)
