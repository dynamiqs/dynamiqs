from dataclasses import dataclass

from ..solver import FixedStep


@dataclass
class Rouchon(FixedStep):
    order: float = 1.0
