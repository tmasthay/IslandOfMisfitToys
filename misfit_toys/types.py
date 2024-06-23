from typing import Callable, Protocol, TypeVar, Union

T1 = TypeVar('T1')
T2 = TypeVar('T2')

UnaryFunction = Callable[[T1], T2]


class HigherOrderUnary(Protocol[T1, T2]):
    def __call__(self, **kwargs) -> UnaryFunction[T1, T2]: ...


ConfigurableUnaryFunction = Union[UnaryFunction, HigherOrderUnary]
