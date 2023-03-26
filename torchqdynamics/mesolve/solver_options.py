from dataclasses import dataclass

from ..solver_options import FixedStep


@dataclass
class Rouchon1(FixedStep):
    pass


Rouchon = Rouchon1


@dataclass
class Rouchon1_5(FixedStep):
    pass


@dataclass
class Rouchon2(FixedStep):
    pass
