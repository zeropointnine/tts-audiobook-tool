import signal
from tts_audiobook_tool.app_types import SingletonBase
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *


class SigIntHandler(SingletonBase):
    """
    Raises a flag which can be checked for when control-c is pressed.

    Must call `init()` first.
    """

    _mode = ""
    _flag = False

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

    @property
    def did_interrupt(self) -> bool:
        return self._flag

    def clear(self) -> bool:
        """
        Returns did_interrupt as a convenience
        """
        result = self._flag
        self._mode = ""
        self._flag = False
        return result

    def signal_handler(self, _, __):
        if not self._mode:
            # Eats control-c
            return

        self._flag = True # TODO: rly shd just send an event or even just callback

        feedback = ""
        match self._mode:
            case "generating":
                feedback = "Control-C pressed, will stop after current generation job is complete"
            case "concatenating":
                feedback = "Control-C pressed, will stop"
        if feedback:
            printt() # Clear any 3p lib's use of "\r" in their print() statements
            printt(COL_ERROR + "*" * len(feedback))
            printt(feedback)
            printt(COL_ERROR + "*" * len(feedback))
