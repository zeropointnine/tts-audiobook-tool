"""Helpers shared by the speaker assignment (auto-assign character voices) menus."""
import os
import tempfile

from tts_audiobook_tool.constants import COL_ACCENT, COL_DEFAULT, COL_DIM
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestProgress
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import print_feedback, printt


def check_llm_configured(state: State) -> bool:
    """Prints feedback and returns False if the LLM is not configured."""
    # Same criterion as voice chat readiness: API key and model name
    # may or may not be required depending on the endpoint
    if not state.prefs.llm_url.strip():
        print_feedback("LLM not configured. See: Options > LLM settings", is_error=True)
        return False
    return True


def print_llm_info(prefs: Prefs) -> None:
    printt(f"LLM: {COL_ACCENT}{prefs.llm_model or '(no model name)'}{COL_DEFAULT} @ {prefs.llm_url}")
    if prefs.llm_extra_params:
        printt(f"{COL_DIM}Extra params: {prefs.llm_extra_params}")


class StatusLine:
    """A single console line, rewritten in place with the request's progress."""

    def __init__(self) -> None:
        self._text = ""

    def update_progress(self, progress: LlmRequestProgress) -> None:
        if progress.answer_chars:
            text = f"Receiving... {progress.answer_chars:,} chars"
        elif progress.reasoning_chars:
            text = f"Thinking... {progress.reasoning_chars:,} chars"
        else:
            text = f"Waiting for response... {format_elapsed(progress.elapsed)}"
            if progress.cancel_pending:
                text += " (cancel pending)"
        if text == self._text:
            return
        padding = " " * max(0, len(self._text) - len(text))
        print(f"\r{text}{padding}", end="", flush=True)
        self._text = text

    def end(self) -> None:
        """Moves past the status line, if one was printed."""
        if self._text:
            print(flush=True)
            self._text = ""

    def clear(self) -> None:
        """Erases the status line, if one was printed (the cursor stays at its start)."""
        if self._text:
            print(f"\r{' ' * len(self._text)}\r", end="", flush=True)
            self._text = ""


def save_temp_text(file_name: str, text: str) -> str:
    """Writes text to a file in the temp directory for inspection; returns its
    path, or "" on failure (not worth interrupting the flow for)."""
    path = os.path.join(tempfile.gettempdir(), file_name)
    try:
        with open(path, "w", encoding="utf-8") as file:
            file.write(text)
    except OSError:
        return ""
    return path


# Rough English-text heuristic; actual counts depend on the model's tokenizer
CHARS_PER_TOKEN = 4


def format_token_count(tokens: int) -> str:
    """eg "151k" (or the exact number below 10,000)"""
    if tokens < 10_000:
        return f"{tokens:,}"
    return f"{round(tokens / 1000):,}k"


def format_elapsed(seconds: float) -> str:
    """eg "48s", "12m 05s", "1h 02m 09s"."""
    minutes, secs = divmod(int(round(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


# Shared by the steps that send book text to the LLM
LLM_DATA_NOTICE = \
"""The full book text is sent to the configured LLM provider, so the usual
considerations about privacy and token costs apply."""
