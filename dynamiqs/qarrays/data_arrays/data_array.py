from abc import abstractmethod

import equinox as eqx
import jax.numpy as jnp

from .layout import Layout

__all__ = ['DataArray']

class DataArray(eqx.Module):

    def __init__(self):
        super().__init__()

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        pass

    @property
    @abstractmethod
    def layout(self) -> Layout:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def mT(self) -> DataArray:
        pass
