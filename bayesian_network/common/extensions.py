# %%


from functools import wraps


class ExtensionDecorator:
    def __init__(self, func, args, kwargs, for_type):
        self._func = func
        self._for_type = for_type
        self._args = args
        self._kwargs = kwargs

    def __call__(self, _):
        raise NotImplementedError("Don't call an extension method directly.")

    def __rrshift__(self, other):
        if not isinstance(other, self._for_type):
            raise TypeError(
                f"Extension '{self._func.__name__}' can only be used on '{self._for_type.__name__}', "
                f"not '{type(other).__name__}'"
            )

        return self._func(other, *self._args, **self._kwargs)


class ExtensionDecoratorFactory:
    def __init__(self, func, for_type):
        self._func = func
        self._for_type = for_type

        wraps(func)(self)

    def __call__(self, *args, **kwargs):
        return ExtensionDecorator(self._func, args, kwargs, self._for_type)


def extension_to(for_type: type):
    def wrapper(func):
        return ExtensionDecoratorFactory(func, for_type)

    return wrapper
