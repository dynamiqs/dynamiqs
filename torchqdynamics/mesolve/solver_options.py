from dataclasses import dataclass

from ..solver import FixedStep


@dataclass
class Rouchon1(FixedStep):
    pass


@dataclass
class Rouchon1_5(FixedStep):
    pass


@dataclass
class Rouchon2(FixedStep):
    pass
