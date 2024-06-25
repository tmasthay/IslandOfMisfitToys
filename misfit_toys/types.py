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
