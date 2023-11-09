from abc import ABC, abstractmethod

class System(ABC):
    @abstractmethod
    def __init__(self):
        self.H = None
        self.jump_ops = None
        self.tsave = None

    @property
    @abstractmethod
    def H(self):
        pass

    @property
    @abstractmethod
    def jump_ops(self):
        pass

    @property
    @abstractmethod
    def psi0(self):
        pass

    @property
    @abstractmethod
    def tsave(self):
        pass

    def to(self, dtype, device):
        self.jump_ops = [op.to(dtype, device) for op in self.jump_ops]
        self.psi0 = self.psi0.to(dtype, device)
        self.tsave = self.tsave.to(dtype.real(), device)
        return self
