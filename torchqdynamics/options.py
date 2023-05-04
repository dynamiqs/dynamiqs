__all__ = [
    'Propagator',
    'Dopri5',
    'Euler',
    'Rouchon',
    'Rouchon1',
    'Rouchon1_5',
    'Rouchon2',
]


class Options:
    def __init__(self, *, verbose: bool = True, save_states: bool = True):
        """...

        Args:
            save_states (bool, optional): If `True`, the state is saved at every
                time value. If `False`, only the final state is stored and returned.
                Defaults to `True`.
        """
        self.verbose = verbose
        self.save_states = save_states


class ODEFixedStep(Options):
    def __init__(self, *, dt: float, **kwargs):
        super().__init__(**kwargs)
        self.dt = dt


class ODEAdaptiveStep(Options):
    def __init__(
        self,
        *,
        atol: float = 1e-8,
        rtol: float = 1e-6,
        max_steps: int = 100_000,
        factor: float = 0.9,
        min_factor: float = 0.2,
        max_factor: float = 5.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.atol = atol
        self.rtol = rtol
        self.max_steps = max_steps
        self.factor = factor
        self.min_factor = min_factor
        self.max_factor = max_factor


class Propagator(Options):
    pass


class Dopri5(ODEAdaptiveStep):
    pass


class Euler(ODEFixedStep):
    pass


class Rouchon1(ODEFixedStep):
    pass


# make alias for Rouchon1
Rouchon = Rouchon1


class Rouchon1_5(ODEFixedStep):
    pass


class Rouchon2(ODEFixedStep):
    pass
