import equinox as eqx


# a decorator that takes a class method f and returns g(f(x))
def pack_dims(method: callable) -> callable:
    """Decorator to return a new QArray with the same dimensions as the original one."""

    def wrapper(self: QArray, *args, **kwargs) -> callable:
        return self.__class__(method(self, *args, **kwargs), dims=self.dims)

    return wrapper


class QArray(eqx.Module):
    pass
