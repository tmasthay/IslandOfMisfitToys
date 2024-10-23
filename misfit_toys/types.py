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
    
class PicklePositional:
    def __init__(self, *, callback: Callable[[Any], Any], update_kw:Callable[[Any], dict]=None, **kwargs):
        self.kwargs = kwargs
        self.callback = callback
        self.update_kw = update_kw    
    
    def __call__(self, *args: Any):
        if self.update_kw is not None:
            self.kwargs = self.update_kw(*args, **self.kwargs)
        return self.callback(*args, **self.kwargs)


class SingleArgPlusKwEnforcer:
    def __init__(
        self, *, callback: Callable[[Any, dict], Any], required_keys, **kwargs
    ):
        self.kwargs = kwargs
        self.callback = callback
        self.required_keys = required_keys

    def __call__(self, x, **kwargs):
        for key in self.required_keys:
            assert key in kwargs, f'Expected key {key} in kwargs'
        return self.callback(x, **self.kwargs, **kwargs)


class KwEnforcer:
    def __init__(
        self, *, callback: Callable[[dict], Any], required_keys, **kwargs
    ):
        self.kwargs = kwargs
        self.callback = callback
        self.required_keys = required_keys

    def __call__(self, **kwargs):
        for key in self.required_keys:
            assert key in kwargs, f'Expected key {key} in kwargs'
        return self.callback(**{**self.kwargs, **kwargs})


class Plotter(KwEnforcer):
    def __init__(self, *, callback: Callable[[dict], Any], **kwargs):
        super().__init__(
            callback=callback,
            required_keys=['data', 'idx', 'fig', 'axes'],
            **kwargs,
        )


class FlatPlotter(KwEnforcer):
    def __init__(self, *, callback: Callable[[dict], Any], **kwargs):
        super().__init__(callback=callback, required_keys=['data'], **kwargs)
        

