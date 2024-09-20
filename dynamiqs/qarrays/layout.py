from enum import Enum


class Layout(Enum):
    DENSE = 'dense'
    DIA = 'dia'

    def __repr__(self) -> str:
        return self.value


dense = Layout.DENSE
dia = Layout.DIA
