import functools
import linecache
import textwrap
import warnings
from typing import Any, Callable

import torch
import torch.nn as nn

from .context import Frame, Mode, pop_frame, push_frame

class _TraceWrapper(nn.Module):
    """Temporary module used by ``compile()`` to produce a traced forward."""

    def __init__(self, fn, registry):
        super().__init__()
        self._fn_ref = fn
        self._reg = registry
        for key, mod in registry.items():
            self.add_module(key.replace("/", "__"), mod)

    def forward(self, *args):
        frame = Frame(mode=Mode.APPLY)
        frame.registry = dict(self._reg)
        push_frame(frame)
        try:
            return self._fn_ref(*args)
        finally:
            pop_frame()

class BlazeModule(nn.Module):
    """A PyTorch ``nn.Module`` produced by :func:`transform`.

    Call :meth:`init` once with a sample input to initialise all sub-modules,
    then use the model normally (``model(x)``).

    After ``init``, the model supports ``torch.jit.script`` for any number of
    tensor inputs or outputs.
    """

    def __init__(self):
        super().__init__()
        self._compiled = False

    @torch.jit.ignore
    def init(self, *args: Any, **kwargs: Any) -> "BlazeModule":
        """Run the wrapped function in INIT mode to discover and create all modules."""
        frame = Frame(mode=Mode.INIT)
        push_frame(frame)
        try:
            self._fn(*args, **kwargs)
        finally:
            pop_frame()

        # Build a traced module that owns all parameters and supports JIT.
        # Trace in eval mode to avoid non-determinism warnings from
        # Dropout / BatchNorm, then restore original mode.
        fn_for_trace = self._fn
        if kwargs:
            fn_for_trace = functools.partial(self._fn, **kwargs)
        wrapper = _TraceWrapper(fn_for_trace, frame.registry)
        wrapper.eval()
        example = args[0] if len(args) == 1 else args
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
            self._traced = torch.jit.trace(wrapper, example)
        wrapper.train()

        # Plain dict for the dynamic (non-JIT) forward path
        self._registry = frame.registry
        self._compiled = True

        # Swap to a concrete-typed subclass so that torch.jit.script() can see
        # explicit parameter annotations. Return type is inferred from _traced.
        self.__class__ = _get_bm_class(len(args))
        return self

    @torch.jit.unused
    def _forward_dynamic(self, *args: Any, **kwargs: Any) -> Any:
        """Full dynamic dispatch — used in eager (non-JIT) mode."""
        if not self._compiled:
            raise RuntimeError(
                "Model not compiled. Call .init(sample_input) first."
            )
        frame = Frame(mode=Mode.APPLY)
        frame.registry = dict(self._registry)
        push_frame(frame)
        try:
            return self._fn(*args, **kwargs)
        finally:
            pop_frame()

    @torch.jit.ignore
    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "_registry"):
            for mod in self._registry.values():
                mod.train(mode)
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Default for uncompiled models; overridden by __class__ swap
        return self._forward_dynamic(x)

    @torch.jit.ignore
    def __repr__(self) -> str:
        keys = list(self._registry.keys()) if hasattr(self, "_registry") else []
        return f"BlazeModule(modules={keys})"

# ---------------------------------------------------------------------------
# Dynamic subclass generation
#
# TorchScript requires explicit parameter and return-type annotations on
# forward().  We generate a concrete subclass for each (n_inputs).
# The source is injected into linecache so that
# inspect.getsource() — which TorchScript calls internally — can find it.
# ---------------------------------------------------------------------------

_BM_CLASS_CACHE: dict = {}


def _make_bm_class(n_inputs: int) -> type:
    params_str = ", ".join(f"x{i}" for i in range(n_inputs))
    args_str   = ", ".join(f"x{i}" for i in range(n_inputs))
    cls_name   = f"_TM{n_inputs}"

    src = textwrap.dedent(f"""\
        class {cls_name}(BlazeModule):
            def forward(self, {params_str}):
                if torch.jit.is_scripting():
                    return self._traced({args_str})
                return self._forward_dynamic({args_str})
    """)

    # Inject source into linecache so inspect.getsource() resolves correctly
    # for TorchScript's internal source-parsing step.
    filename = f"<blaze:{cls_name}>"
    lines = src.splitlines(keepends=True)
    linecache.cache[filename] = (len(src), None, lines, filename)

    code = compile(src, filename, "exec")
    namespace: dict = {"BlazeModule": BlazeModule, "torch": torch}
    exec(code, namespace)  # noqa: S102
    return namespace[cls_name]


def _get_bm_class(n_inputs: int) -> type:
    if n_inputs not in _BM_CLASS_CACHE:
        _BM_CLASS_CACHE[n_inputs] = _make_bm_class(n_inputs)
    return _BM_CLASS_CACHE[n_inputs]


@torch.jit.ignore
def transform(fn: Callable, *args: Any, **kwargs: Any) -> BlazeModule:
    """Wrap a forward-pass function into a :class:`BlazeModule`.

    Any extra ``*args`` / ``**kwargs`` are partially applied to *fn* so that
    :meth:`~BlazeModule.init` and the forward call only need the
    tensor inputs.

    Example::

        model = blaze.transform(forward, in_size=10, out_size=20)
        model.init(torch.randn(5, 10))
        output = model(torch.randn(5, 10))
    """
    if args or kwargs:
        fn = functools.partial(fn, *args, **kwargs)
    model = BlazeModule()
    model._fn = fn  # stored outside __init__ to stay invisible to TorchScript
    return model
