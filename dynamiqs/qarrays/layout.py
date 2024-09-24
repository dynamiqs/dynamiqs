from enum import Enum


class Layout(Enum):
    DENSE = 'dense'
    DIA = 'dia'

    def __repr__(self) -> str:
        return self.value

    def __str__(self) -> str:
        return repr(self)


dense = Layout.DENSE
dia = Layout.DIA
