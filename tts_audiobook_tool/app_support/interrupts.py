import signal
from typing import Protocol

from tts_audiobook_tool.app_types import SingletonBase


class InterruptEvent(Protocol):
    def is_set(self) -> bool: ...
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *


class Interrupts(SingletonBase):
    """
    Raises a flag which can be checked for when control-c is pressed.

    Must call `init()` first.
    """

    _mode = ""
    _flag = False
    _press_count = 0
    _external_event: InterruptEvent | None = None

    def __init__(self):
        ...

    def init(self) -> None:
        signal.signal(signal.SIGINT, self.signal_handler)

    def set(self, mode: str) -> None:
        """
        Any value for `mode`
        """
        self._mode = mode
        self._flag = False
        self._press_count = 0

    def set_external_event(self, event: InterruptEvent | None) -> None:
        """Use a process-safe event as an additional cancellation source."""
        self._external_event = event

    @property
    def did_interrupt(self) -> bool:
        event = self._external_event
        return self._flag or (event is not None and event.is_set())

    @property
    def press_count(self) -> int:
        """Number of control-c presses since `set()`."""
        return self._press_count

    def clear(self) -> bool:
        """
        Returns did_interrupt as a convenience
        """
        result = self.did_interrupt
        self._mode = ""
        self._flag = False
        self._press_count = 0
        return result

    def signal_handler(self, _, __):
        if not self._mode:
            # Eats control-c
            return

        self._flag = True
        self._press_count += 1

        feedback = ""
        match self._mode:
            case "model init":
                feedback = "Control-C pressed, will cancel"
            case "generating":
                feedback = "Control-C pressed, will stop after current generation is complete"
            case "concatenating":
                feedback = "Control-C pressed, will stop"
            case "llm":
                # Silent: the caller reports what a press does (see LlmRequestUtil.send_cancellable)
                feedback = ""
            case _:
                feedback = "Control-C pressed, will stop"
        if feedback:
            printt()
            printt(COL_ERROR + "*" * len(feedback))
            printt(feedback)
            printt(COL_ERROR + "*" * len(feedback))
