"""Helpers shared by the speaker assignment (Auto-identify characters) menus."""
import os
import tempfile

from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestProgress
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.util import duration_string, print_feedback, printt


def check_llm_configured(state: State) -> bool:
    """Prints feedback and returns False if the LLM is not configured."""
    # Same criterion as voice chat readiness: API key and model name
    # may or may not be required depending on the endpoint
    if not state.prefs.llm_url.strip():
        print_feedback("LLM not configured. See: Options > LLM settings", is_error=True)
        return False
    return True


def print_llm_info(prefs: Prefs) -> None:
    printt(f"{COL_DEFAULT}LLM: {COL_DIM}{prefs.llm_url}, {prefs.llm_model or '(no model name)'}")
    if prefs.llm_speaker_assign_extra_params:
        printt(f"{COL_DEFAULT}Extra params: {COL_DIM}{prefs.llm_speaker_assign_extra_params}")


class StatusLine:
    """A single console line, rewritten in place with the request's progress."""

    def __init__(self) -> None:
        self._text = ""

    def update_progress(self, progress: LlmRequestProgress) -> None:
        if progress.answer_chars:
            text = f"{COL_DEFAULT}Receiving: {COL_DIM}{progress.answer_chars:,} chars{COL_DEFAULT}"
        elif progress.reasoning_chars:
            text = f"{COL_DEFAULT}Thinking: {COL_DIM}{progress.reasoning_chars:,} chars{COL_DEFAULT}"
        else:
            text = f"{COL_DEFAULT}Waiting for response: {COL_DIM}{duration_string(progress.elapsed)}"
            if progress.cancel_pending:
                text += " (cancel pending)"
            text += COL_DEFAULT
        if text == self._text:
            return
        # Padding erases the previous text's tail; widths exclude color codes
        padding = " " * max(0, visible_len(self._text) - visible_len(text))
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
            print(f"\r{' ' * visible_len(self._text)}\r", end="", flush=True)
            self._text = ""


def visible_len(text: str) -> int:
    return len(strip_ansi_codes(text))


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
