from __future__ import annotations


class System:
    pass


class ClosedSystem(System):
    def __init__(self):
        self.H = None
        self.y0 = None
        self.tsave = None


class OpenSystem(ClosedSystem):
    def __init__(self):
        super().__init__()
        self.jump_ops = None
