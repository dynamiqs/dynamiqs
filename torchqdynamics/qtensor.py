from typing import List

import numpy as np
import qutip as qt
import torch


class QTensor(torch.Tensor):
    """
    A torch tensor that represents a quantum object (operator, density matrix, state vector...)

    Subclasses of Tensor can slow torch down significantly
    TODO: check performance impact and implement mitigations
    Ref : https://benjaminwarner.dev/2022/06/14/debugging-pytorch-performance-decrease
    """
    def __init__(self, *args, **kwargs):
        """ Creates a QTensor
            Args:
                - arguments inherited from torch.Tensor
                - dims: the dimension of the Hilbert sub-spaces
         """
        dims = kwargs.pop('dims', None)
        super().__init__(*args, **kwargs)

        if dims is None:
            raise ValueError(
                "Argument 'dims' must be specified at QTensor initialisation."
            )
        self.dims = dims

    def qutip(self) -> qt.Qobj:
        """Convert a PyTorch tensor to a QuTiP quantum object.

        Note:
            The returned object does not share memory with the argument, it is
            simply a copy.
        Returns:
            QuTiP quantum object.
        """
        return qt.Qobj(self.numpy(force=True), dims=self.dims)

    @staticmethod
    def from_qutip(x: qt.Qobj) -> torch.Tensor:
        """Convert a QuTiP quantum object to a PyTorch tensor.

        Note:
            The returned object does not share memory with the argument, it is
            simply a copy.

        Args:
            x: QuTiP quantum object.

        Returns:
            TorchQDynamics QTensor.
        """
        return QTensor(torch.from_numpy(x.full()), dims=x.dims)

    @staticmethod
    def from_numpy(x: np.array, dims: List[int]):
        return QTensor(torch.from_numpy(x.full()), dims=dims)
