from typing import Callable, Optional

import torch
import torch.nn as nn

from .context import Mode, get_current_frame


class _ParameterHolder(nn.Module):
    def __init__(self, param: nn.Parameter):
        super().__init__()
        self.param = param


class _BufferHolder(nn.Module):
    def __init__(self, tensor: torch.Tensor):
        super().__init__()
        self.register_buffer("buf", tensor)


def get_state(
    name: str,
    shape: tuple[int, ...],
    init_fn: Optional[Callable] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Create or retrieve a buffer tensor, analogous to ``hk.get_state``.

    Buffers are non-trainable state tensors registered via ``register_buffer``.
    In INIT mode, creates and registers the buffer.
    In APPLY mode, retrieves the existing buffer.
    """
    frame = get_current_frame()

    prefix = frame.current_path()
    full_path = prefix + "." + name if prefix else name

    if frame.mode == Mode.INIT:
        if init_fn is None:
            init_fn = torch.zeros
        data = init_fn(shape, dtype=dtype)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype)
        holder = _BufferHolder(data)
        frame.registry[full_path] = holder
        return holder.buf
    else:
        if full_path not in frame.registry:
            raise RuntimeError(
                f"State '{full_path}' not found in registry. "
                f"Function structure changed between compile() and forward()."
            )
        return frame.registry[full_path].buf


def get_parameter(
    name: str,
    shape: tuple[int, ...],
    init_fn: Optional[Callable] = None,
    dtype: torch.dtype = torch.float32,
) -> nn.Parameter:
    """Create or retrieve a raw ``nn.Parameter``, analogous to ``hk.get_parameter``.

    In INIT mode, creates a new parameter and registers it.
    In APPLY mode, retrieves the existing parameter.
    """
    frame = get_current_frame()

    prefix = frame.current_path()
    full_path = prefix + "." + name if prefix else name

    if frame.mode == Mode.INIT:
        if init_fn is None:
            init_fn = torch.zeros
        data = init_fn(shape, dtype=dtype)
        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data, dtype=dtype)
        param = nn.Parameter(data)
        frame.registry[full_path] = _ParameterHolder(param)
        return param
    else:
        if full_path not in frame.registry:
            raise RuntimeError(
                f"Parameter '{full_path}' not found in registry. "
                f"Function structure changed between compile() and forward()."
            )
        return frame.registry[full_path].param
