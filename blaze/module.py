from typing import Any

from .context import get_current_frame
from .utils import camel_to_snake


class Module:
    """Base class for user-defined blaze modules.

    Subclasses override ``__call__`` to define the forward pass using inline
    ``blaze`` layer wrappers.  The ``__call__`` is automatically wrapped to
    push/pop a naming scope so that inner layers get hierarchical paths.

    Example::

        class Block(blaze.Module):
            def __call__(self, x):
                x = blaze.Linear(32, 64)(x)
                return blaze.ReLU()(x)
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Wrap __call__ if the subclass defines its own
        if "__call__" in cls.__dict__:
            original = cls.__dict__["__call__"]
            cls.__call__ = _wrap_call(original)

    def __init__(self, name: str | None = None):
        frame = get_current_frame()
        base = name or camel_to_snake(type(self).__name__)
        self._blaze_name = frame.current_counter.next_name(base)

    def __call__(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            f"{type(self).__name__} must implement __call__"
        )


def _wrap_call(fn):
    """Wrap a user-defined __call__ with scope push/pop."""

    def wrapper(self, *args, **kwargs):
        frame = get_current_frame()
        frame.push_scope(self._blaze_name)
        try:
            return fn(self, *args, **kwargs)
        finally:
            frame.pop_scope()

    wrapper.__name__ = fn.__name__
    wrapper.__qualname__ = fn.__qualname__
    wrapper._blaze_wrapped = True
    return wrapper
