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
    def tsave(self):
        pass
