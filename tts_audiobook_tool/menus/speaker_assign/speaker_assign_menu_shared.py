"""Helpers shared by the speaker assignment (Auto-identify characters) menus."""
from dataclasses import dataclass
import os
import tempfile

from tts_audiobook_tool.constants import COL_DEFAULT, COL_DIM
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestProgress, LlmRequestResult
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
            text = f"{COL_DEFAULT}Receiving: {COL_DIM}~{format_token_count(estimate_tokens(progress.answer_chars), realtime=True)} tokens{COL_DEFAULT}"
        elif progress.reasoning_chars:
            text = f"{COL_DEFAULT}Thinking: {COL_DIM}~{format_token_count(estimate_tokens(progress.reasoning_chars), realtime=True)} tokens{COL_DEFAULT}"
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


def estimate_tokens(chars: int) -> int:
    """Rough token count, rounded up so a nonempty stream never shows zero."""
    return (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


@dataclass
class OutputUsage:
    """Reported usage plus estimates for missing metadata, including partial streams.

    Output includes reasoning. Estimates cover only visible text; hidden reasoning
    cannot be reconstructed unless the provider reports its token count.
    """
    output_tokens: int = 0
    reasoning_tokens: int = 0
    estimated_output: bool = False
    estimated_reasoning: bool = False
    has_reasoning: bool = False

    def add(self, result: LlmRequestResult, *, answer_chars: int = 0, reasoning_chars: int = 0) -> None:
        answer_chars = max(answer_chars, len(result.reply))
        reasoning_chars = max(reasoning_chars, len(result.reasoning))
        self.has_reasoning |= bool(reasoning_chars or result.reasoning_tokens)
        reasoning = (result.reasoning_tokens if result.reasoning_tokens is not None
                     else estimate_tokens(reasoning_chars))
        self.reasoning_tokens += reasoning
        self.estimated_reasoning |= bool(reasoning_chars and result.reasoning_tokens is None)
        if result.output_tokens is not None:
            self.output_tokens += result.output_tokens
        else:
            # Use reported reasoning even when its text is hidden. Never add it
            # to a reported output count, which already includes reasoning.
            self.output_tokens += (estimate_tokens(answer_chars) + reasoning
                                   if result.reasoning_tokens is not None
                                   else estimate_tokens(answer_chars + reasoning_chars))
            self.estimated_output = True

    def format(self) -> str:
        output = f"{'~' if self.estimated_output else ''}{format_token_count(self.output_tokens)} output tokens"
        if self.has_reasoning:
            output += f" ({'~' if self.estimated_reasoning else ''}{format_token_count(self.reasoning_tokens)} reasoning)"
        return output


def format_token_count(tokens: int, *, realtime: bool = False) -> str:
    """Always use K; round summaries over 10K to whole K (halfway rounds up)."""
    if not realtime and tokens > 10_000:
        return f"{(tokens + 500) // 1000:,}K"
    tenths = (tokens + 50) // 100
    return f"{tenths // 10:,}.{tenths % 10}K"
