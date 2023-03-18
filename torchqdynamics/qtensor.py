from math import prod
from typing import Optional, Tuple

import numpy as np
import qutip as qt
import torch

QTYPE_FULL = {
    'ket': 'Ket quantum state',
    'bra': 'Bra quantum state',
    'dm': 'Density matrix or Operator',
}


class QTensor(object):
    def __init__(
        self, data: torch.Tensor, *, qtype: Optional[str] = None,
        dims: Optional[Tuple[int, ...]] = None, **kwargs
    ):
        """Class derived from torch.Tensor that represents a quantum object.

        If `qtype` is None, it is inferred from the following rules on the
        input tensor shape (highest priority first):
            (n): ket
            (n,n): dm
            (n,1): ket
            (1,n): bra
            (b,n): ket (batched)
            (b,n,n): dm (batched)
            (b,n,1): ket (batched)
            (b,1,n): bra (batched)

        Args:
            data: Quantum object values.
            qtype: Quantum object type. Can be 'ket', 'bra', 'dm'.
                Defaults to None.
            dims: Underlying tensor product dimensions of the Hilbert space.
                Defaults to None.
        Attributes:
            data: Tensor of shape (b,n,m) containing the quantum object values,
                where b >= 1 is the batch size, and (n,m) are the quantum object
                total dimensions, with n==m or n==1 or m==1.
            qtype: Quantum object type. Can be 'ket', 'bra', 'dm'.
            dims: Underlying tensor product dimensions of the Hilbert space.
            hilbert_size: Total size of the Hilbert space.
            batch_size: Size of the batch.
        """
        self.data = torch.as_tensor(data, **kwargs)
        self.qtype = qtype or self.qtype_from_shape(self.data.shape)
        self.dims = dims
        self.data = self.unsqueeze_by_qtype(self.data, self.qtype)
        self.batch_size = self.data.size(0)
        self.hilbert_size = max(self.data.size(1), self.data.size(2))
        if not prod(self.dims) == self.hilbert_size:
            raise ValueError(
                f'Argument dims {self.dims} does not match the '
                f'Hilbert space size {self.hilbert_size}.'
            )

    def qtype_from_shape(self, shape):
        """Infer a qtype from a given tensor shape."""
        if len(shape) == 1:
            qtype = 'ket'
        elif len(shape) == 2:
            if shape[0] == shape[1]:
                qtype = 'dm'
            elif shape[1] == 1:
                qtype = 'ket'
            elif shape[0] == 1:
                qtype = 'bra'
            else:
                qtype = 'ket'
        elif len(shape) == 3:
            if shape[1] == shape[2]:
                qtype = 'dm'
            elif shape[2] == 1:
                qtype = 'ket'
            elif shape[1] == 1:
                qtype = 'bra'
            else:
                raise ValueError(
                    'The input tensor must be of shape (b,n,n) for a density matrix, '
                    '(b,n,1) for a ket or (b,1,n) for a bra.'
                )
        else:
            raise ValueError(
                f'The input tensor has {len(shape)} dimensions, but must have '
                'between 1 and 3.'
            )
        return qtype

    def unsqueeze_by_qtype(self, data, qtype):
        """
        Unsqueeze the input data tensor to a tensor of shape (b,n,m) using
        the proper unsqueezing rule defined by the qtype. Also checks if the
        data shape is compatible with the specified qtype.
        """
        error_msg = (
            f'The input tensor shape {data.shape} is incompatible with '
            f'the specified qtype {qtype}.'
        )
        data = torch.squeeze(data)
        if qtype == 'ket':
            if data.ndim == 1:
                data = data[None, :, None]
            elif data.ndim == 2:
                data = data[:, :, None]
            else:
                raise ValueError(error_msg)
        elif qtype == 'bra':
            if data.ndim == 1:
                data = data[None, None, :]
            elif data.ndim == 2:
                data = data[:, None, :]
            else:
                raise ValueError(error_msg)
        elif qtype == 'dm':
            if data.ndim == 2 and data.size(0) == data.size(1):
                data = data[None, :, :]
            elif data.ndim == 3 and data.size(1) == data.size(2):
                data = data
            else:
                raise ValueError(error_msg)
        else:
            raise TypeError(f'Argument qtype {qtype} is not recognized.')
        return data

    def __repr__(self):
        return (
            f'{QTYPE_FULL[self.qtype]} QTensor of shape {tuple(self.data.shape)}'
            f'and dims {self.dims}.\n'
            f'QTensor data: \n{torch.squeeze(self.data)}'
        )

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        metadatas = tuple(a._metadata for a in args if hasattr(a, '_metadata'))
        args = [a.data if hasattr(a, 'data') else a for a in args]
        assert len(metadatas) > 0
        ret = func(*args, **kwargs)
        return MetadataTensor(ret, metadata=metadatas[0])


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
    def from_qutip(x: qt.Qobj) -> torch.tensor:
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
