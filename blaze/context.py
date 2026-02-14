import threading
from collections import defaultdict
from enum import Enum, auto
from typing import Optional

import torch.nn as nn


class Mode(Enum):
    INIT = auto()
    APPLY = auto()


class NameCounter:
    def __init__(self):
        self._counts: dict[str, int] = defaultdict(int)

    def next_name(self, base: str) -> str:
        count = self._counts[base]
        self._counts[base] += 1
        if count == 0:
            return base
        return f"{base}_{count}"


class Frame:
    def __init__(self, mode: Mode = Mode.INIT):
        self.mode = mode
        self.registry: dict[str, nn.Module] = {}
        self.name_stack: list[str] = []
        self.counter_stack: list[NameCounter] = [NameCounter()]
        self.call_order: list[str] = []

    @property
    def current_counter(self) -> NameCounter:
        return self.counter_stack[-1]

    def push_scope(self, scope_name: str):
        self.name_stack.append(scope_name)
        self.counter_stack.append(NameCounter())

    def pop_scope(self):
        self.name_stack.pop()
        self.counter_stack.pop()

    def current_path(self) -> str:
        return ".".join(self.name_stack)


class _FrameStack(threading.local):
    def __init__(self):
        super().__init__()
        self.stack: list[Frame] = []

    def push(self, frame: Frame):
        self.stack.append(frame)

    def pop(self) -> Frame:
        return self.stack.pop()

    @property
    def current(self) -> Optional[Frame]:
        return self.stack[-1] if self.stack else None


_frame_stack = _FrameStack()


def get_current_frame() -> Frame:
    frame = _frame_stack.current
    if frame is None:
        raise RuntimeError(
            "No active blaze frame. All blaze module usage must happen inside "
            "a function wrapped with blaze.transform()."
        )
    return frame


def push_frame(frame: Frame):
    _frame_stack.push(frame)


def pop_frame() -> Frame:
    return _frame_stack.pop()
