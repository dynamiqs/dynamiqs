import torch


def construct_dopri5(y_dtype, t_dtype):
    alpha = torch.tensor([1 / 5, 3 / 10, 4 / 5, 8 / 9, 1., 1., 0.], dtype=t_dtype)
    beta = torch.tensor(
        [[1 / 5, 0, 0, 0, 0, 0, 0], [3 / 40, 9 / 40, 0, 0, 0, 0, 0],
         [44 / 45, -56 / 15, 32 / 9, 0, 0, 0, 0],
         [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0, 0],
         [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0, 0],
         [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0]], dtype=y_dtype)
    csol = torch.tensor([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84, 0],
                        dtype=y_dtype)
    cerr_ = torch.tensor([
        1951 / 21600, 0, 22642 / 50085, 451 / 720, -12231 / 42400, 649 / 6300, 1 / 60.
    ], dtype=y_dtype)
    return (alpha, beta, csol, csol - cerr_)
