from dataclasses import dataclass

from torchqdynamics.solvers._base import FixedStepOption, VariableStepOption


@dataclass
class Rouchon(FixedStepOption):
    """ TODO: write description """
    pass


@dataclass
class RK4(FixedStepOption):
    """ TODO: write description """
    pass


@dataclass
class DOPRI6(VariableStepOption):
    """ TODO: write description """
    pass
