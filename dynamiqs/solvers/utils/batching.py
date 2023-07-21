from torch import Tensor

from .td_tensor import TDTensor


def batch_H(H: TDTensor) -> TDTensor:
    # handle H batching
    # x: (b_H?, n, n) ->  (b_H, 1, n, n)
    if H.dim() == 2:  # (n, n)
        H = H.unsqueeze(0)  # (1, n, n)
    return H.unsqueeze(1)


def batch_y0(y0: Tensor, H: TDTensor) -> Tensor:
    # handle y0 batching
    # y0: (b_y0?, m, n) ->  (1, b_y0, n, n)
    if y0.ndim == 2:  # (n, n)
        y0 = y0.unsqueeze(0)  # (1, n, n)
    y0 = y0.unsqueeze(0)  # (1, 1, n, n)
    return y0.repeat(H.size(0), 1, 1, 1)
