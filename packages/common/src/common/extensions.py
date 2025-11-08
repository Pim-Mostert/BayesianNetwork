# %%

from functools import wraps
from typing import Any, Callable, Generic, Type, TypeVar

T = TypeVar("T")


class ExtensionDecorator(Generic[T]):
    def __init__(self, func: Callable[..., Any], args, kwargs, to: Type[T]):
        self._func = func
        self._to = to
        self._args = args
        self._kwargs = kwargs

    def __call__(self, _: Any) -> Any:
        raise NotImplementedError("Don't call an extension method directly.")

    def __ror__(self, other: T) -> Any:
        if not isinstance(other, self._to):
            raise TypeError(
                f"Extension '{self._func.__name__}' can only be used on '{self._to.__name__}', "
                f"not '{type(other).__name__}'"
            )

        return self._func(other, *self._args, **self._kwargs)


class ExtensionDecoratorFactory(Generic[T]):
    def __init__(self, func: Callable[..., Any], to: Type[T]):
        self._func = func
        self._to = to

        wraps(func)(self)

    def __call__(self, *args, **kwargs) -> ExtensionDecorator[T]:
        return ExtensionDecorator(self._func, args, kwargs, self._to)


def extension(*, to: Type[T]) -> Callable[[Callable[..., Any]], ExtensionDecoratorFactory[T]]:
    def wrapper(func: Callable[..., Any]) -> ExtensionDecoratorFactory[T]:
        return ExtensionDecoratorFactory(func, to)

    return wrapper
