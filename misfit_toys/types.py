from typing import Any, Callable, Protocol, TypeVar, Union

T1 = TypeVar('T1')
T2 = TypeVar('T2')

UnaryFunction = Callable[[T1], T2]


class HigherOrderUnary(Protocol[T1, T2]):
    def __call__(self, **kwargs) -> UnaryFunction[T1, T2]: ...


ConfigurableUnaryFunction = Union[UnaryFunction, HigherOrderUnary]


class PickleUnaryFunction:
    def __init__(self, *, callback: Callable[[Any, dict], Any], **kwargs):
        self.kwargs = kwargs
        self.callback = callback

    def __call__(self, x: Any):
        return self.callback(x, **self.kwargs)


class SingleArgPlusKwEnforcer:
    def __init__(
        self, *, callback: Callable[[Any, dict], Any], expected_keys, **kwargs
    ):
        self.kwargs = kwargs
        self.callback = callback
        self.expected_keys = expected_keys

    def __call__(self, x, **kwargs):
        for key in self.expected_keys:
            assert key in kwargs, f'Expected key {key} in kwargs'
        return self.callback(x, **self.kwargs, **kwargs)


class HardKwEnforcer:
    def __init__(
        self, *, callback: Callable[[dict], Any], expected_keys, **kwargs
    ):
        self.kwargs = kwargs
        self.callback = callback
        self.expected_keys = expected_keys

    def __call__(self, **kwargs):
        for key in self.expected_keys:
            assert key in kwargs, f'Expected key {key} in kwargs'
        return self.callback(**{**self.kwargs, **kwargs})


# I think I may have been errant in the need of this class in the past
# Reconcile this decision later...move on for now, not a huge deal
# Do not think this subclass is necessary OR wanted actually
class SoftKwEnforcer(HardKwEnforcer):
    def __init__(
        self, *, callback: Callable[[dict], Any], expected_keys, **kwargs
    ):
        self.kwargs = kwargs
        self.callback = callback
        self.expected_keys = expected_keys + list(kwargs.keys())


class Plotter(HardKwEnforcer):
    def __init__(self, *, callback: Callable[[dict], Any], **kwargs):
        super().__init__(
            callback=callback,
            expected_keys=['data', 'idx', 'fig', 'axes'],
            **kwargs,
        )


class SoftPlotter(SoftKwEnforcer):
    def __init__(self, *, callback: Callable[[dict], Any], **kwargs):
        super().__init__(
            callback=callback,
            expected_keys=['data', 'idx', 'fig', 'axes'],
            **kwargs,
        )


class FlatPlotter(HardKwEnforcer):
    def __init__(self, *, callback: Callable[[dict], Any], **kwargs):
        super().__init__(callback=callback, expected_keys=['data'], **kwargs)
