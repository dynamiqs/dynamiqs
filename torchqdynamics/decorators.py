"""
Code freely inspired from functools.cached_property
"""
import inspect
from copy import copy
from typing import Dict

CACHE_KEY = '__cache__'
CACHED_VALUE_KEY = 'cached_value'
PREVIOUS_KEY = 'previous_key'
HAS_CHANGED_KEY = 'has_changed'


def cached_main(func):
    """
    Decorator that implements a size 1 LRU and propagates changes to children,
    defined using `cached_depends_on`. The function must return a boolean to indicate whether the
    value changed or not since last call
    """
    func_name = func.__name__

    def inner(instance, *args, **kwargs):
        instance_dict = instance.__dict__
        if CACHE_KEY not in instance_dict:
            instance_dict[CACHE_KEY] = {}

        cache_dict = instance_dict[CACHE_KEY].setdefault(func_name, {})

        key = _make_key(args, kwargs, False)
        has_changed = False

        if cache_dict.get(CACHED_VALUE_KEY
                         ) is None or cache_dict.get(PREVIOUS_KEY) != key:
            val = func(instance, *args, **kwargs)
            assert isinstance(
                val[-1], bool
            ), 'The last return param of a cached_main function must be a boolean indicating whether the return value ' \
               'should change'
            has_changed = val[-1] or cache_dict.get(CACHED_VALUE_KEY) is None
            cache_dict[PREVIOUS_KEY] = key
            if has_changed:
                cache_dict[CACHED_VALUE_KEY] = val[0:-1]

        cache_dict[HAS_CHANGED_KEY] = has_changed
        return cache_dict[CACHED_VALUE_KEY]

    return inner


def cached_depends_on(dependencies: Dict[str, str]):
    """
    Decorator that implements a size 1 LRU on the function. All the parameters of the functions
    are hashed as the LRU key **except** for the dependencies.
    We call 'source function' a function that computes a dependence.
    Dependencies are results from class method marked with the decorator `@cached_main`.
    With all non dependency parameters equal, the source function will be triggered iff at least one of the
    dependencies has changed.

    Dependencies are declared in a dictionnary as :
    - key: dependent function parameter name
    - value: source function the parameter depends on
    """
    def decorator(func):
        func_name = func.__name__
        argument_names = inspect.getfullargspec(func)[0][1:]  # drop 'self'

        def inner(instance, *args, **kwargs):
            instance_dict = instance.__dict__
            if CACHE_KEY not in instance_dict:
                instance_dict[CACHE_KEY] = {}

            cache_dict = instance_dict[CACHE_KEY].setdefault(func_name, {})

            # Compute a key for all caching argument except for the one that depends on cached values
            key_kwargs = copy(kwargs)
            key_kwargs.update(dict(zip(argument_names, args)))
            for argument_name in dependencies.keys():
                del key_kwargs[argument_name]
            key = _make_key(tuple(), key_kwargs, False)

            no_cached_value = cache_dict.get(CACHED_VALUE_KEY) is None

            dependencies_have_changed = any(
                instance_dict[CACHE_KEY].get(dependence, {}
                                            ).get(HAS_CHANGED_KEY, False)
                for dependence in dependencies.values()
            )

            key_has_changed = cache_dict.get(PREVIOUS_KEY) != key

            if no_cached_value or dependencies_have_changed or key_has_changed:
                val = func(instance, *args, **kwargs)
                cache_dict[CACHED_VALUE_KEY] = val
                cache_dict[PREVIOUS_KEY] = key

            return cache_dict[CACHED_VALUE_KEY]

        return inner

    return decorator


################################################################################
# Code below is forked from functools
################################################################################


class _HashedSeq(list):
    """ This class guarantees that hash() will be called no more than once
        per element.  This is important because the lru_cache() will hash
        the key multiple times on a cache miss.
    """

    __slots__ = 'hashvalue'

    def __init__(self, tup, hash=hash):
        self[:] = tup
        self.hashvalue = hash(tup)

    def __hash__(self):
        return self.hashvalue


def _make_key(
    args, kwds, typed, kwd_mark=(object(), ), fasttypes={int, str}, tuple=tuple,
    type=type, len=len
):
    """Make a cache key from optionally typed positional and keyword arguments

    The key is constructed in a way that is flat as possible rather than
    as a nested structure that would take more memory.

    If there is only a single argument and its data type is known to cache
    its hash value, then that argument is returned without a wrapper.  This
    saves space and improves lookup speed.

    """
    # All of code below relies on kwds preserving the order input by the user.
    # Formerly, we sorted() the kwds before looping.  The new way is *much*
    # faster; however, it means that f(x=1, y=2) will now be treated as a
    # distinct call from f(y=2, x=1) which will be cached separately.
    key = args
    if kwds:
        key += kwd_mark
        for item in kwds.items():
            key += item
    if typed:
        key += tuple(type(v) for v in args)
        if kwds:
            key += tuple(type(v) for v in kwds.values())
    elif len(key) == 1 and type(key[0]) in fasttypes:
        return key[0]
    return _HashedSeq(key)
