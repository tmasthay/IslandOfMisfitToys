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
