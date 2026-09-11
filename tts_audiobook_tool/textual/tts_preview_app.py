from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar, cast

from textual.app import ComposeResult

from tts_audiobook_tool import util
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationTerminalStatus,
    GenerationUpdate,
    ModelWorkerEvent,
    TtsPreviewFinished,
    WorkerExited,
)
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual.generation_header import GenerationHeader, PromptMode
from tts_audiobook_tool.textual.worker_app import (
    FINAL_OUTPUT_SETTLE_SECONDS,
    WorkerTextualApp,
    session_failure_result,
    worker_app_css,
)
from tts_audiobook_tool.worker_reset import HardResetCause


@dataclass(frozen=True)
class TtsPreviewResult:
    """Terminal result for one non-persistent diagnostic TTS prompt."""

    status: GenerationTerminalStatus
    sound: Sound | None = None
    message: str = ""
    hard_reset_cause: HardResetCause | None = None

    @property
    def completed(self) -> bool:
        return self.status is GenerationTerminalStatus.COMPLETED


_TERMINAL_LABELS: dict[GenerationTerminalStatus, str] = {
    GenerationTerminalStatus.COMPLETED: "Preview generated.",
    GenerationTerminalStatus.CANCELLED: "Preview generation cancelled.",
    GenerationTerminalStatus.ABORTED: "Preview generation stopped.",
    GenerationTerminalStatus.FAILED: "Preview generation failed.",
    GenerationTerminalStatus.WORKER_RESET: "Preview generation stopped; model worker was reset.",
}


class TtsPreviewApp(WorkerTextualApp[TtsPreviewResult]):
    """Full-screen worker session for one in-memory TTS preview."""

    CSS = worker_app_css("tts-preview-divider")

    DIVIDER_ID: ClassVar[str] = "tts-preview-divider"
    OUTPUT_SHELL_ID: ClassVar[str] = "tts-preview-output-shell"
    # A preview emits no progress and its header shows only the elapsed
    # counter, which renders whole seconds, so the generation session's rate
    # is already faster than this header can change.
    HEADER_UPDATE_SECONDS: ClassVar[float] = 1.0

    def __init__(self, state: State, prompt: str) -> None:
        super().__init__(state)
        self.prompt = prompt

    def compose_header(self) -> ComposeResult:
        yield GenerationHeader(title="Generating preview", id="tts-preview-header")

    def submit_worker_job(self) -> str:
        return ModelWorker.submit_tts_preview(
            state=self.state,
            prompt=self.prompt,
            apply_word_substitutions=False,
        )

    def _on_submit_failure(self, message: str) -> None:
        self._show_terminal_summary(
            TtsPreviewResult(GenerationTerminalStatus.FAILED, message=message)
        )

    def _handle_session_event(self, event: ModelWorkerEvent) -> None:
        if isinstance(event, GenerationUpdate):
            self._handle_update(event.update)
            return
        if not isinstance(event, TtsPreviewFinished):
            return
        if self.finishing or self.terminal_result is not None:
            return
        self.finishing = True
        self.phase = event.status.value.replace("_", " ").title()
        self._update_header()
        self.set_timer(
            FINAL_OUTPUT_SETTLE_SECONDS,
            lambda: self._finish_from_worker_event(event),
        )

    def _finish_from_worker_event(self, event: TtsPreviewFinished) -> None:
        self._finalize_console()
        sound = cast(Sound | None, event.sound)
        status = event.status
        message = event.message
        if status is GenerationTerminalStatus.COMPLETED and sound is None:
            status = GenerationTerminalStatus.FAILED
            message = "Preview generation returned no audio"
        self._show_terminal_summary(
            TtsPreviewResult(status=status, sound=sound, message=message)
        )

    def _handle_update(self, update: object) -> None:
        """Route worker-health updates; a preview emits no progress events.

        The preview's single inference runs under the GEN_TIMEOUT watchdog, so
        the only structured updates it can produce are the ones that ask for a
        worker reset (`GenerationTimedOut`, `ModelUnhealthy`). Those take the
        same path as in the generation and realtime sessions.
        """
        # Once a reset or terminal flow owns the session, ignore updates that
        # the old worker had already queued.
        if self.reset_in_progress or self.terminal_result is not None:
            return
        if self._begin_hard_reset_for_update(update):
            self._update_header()

    def _on_worker_command_failed(self, message: str) -> None:
        self._finalize_failure(message)

    def _on_worker_exit(self, event: WorkerExited) -> None:
        self._finalize_failure(
            event.message or "Model worker exited during preview generation"
        )

    def _finalize_failure(self, message: str) -> None:
        if self.terminal_result is not None:
            return
        self._finalize_console()
        self._show_terminal_summary(
            TtsPreviewResult(GenerationTerminalStatus.FAILED, message=message)
        )

    def terminal_label(self, result: TtsPreviewResult) -> str:
        return "" if result.completed else _TERMINAL_LABELS[result.status]

    def terminal_summary_extra_lines(self, result: TtsPreviewResult) -> list[str]:
        if self.auto_exit:
            return []
        return ["", f"Press {util.make_hotkey_string('ENTER')} to continue"]

    def should_auto_exit(self, result: TtsPreviewResult) -> bool:
        """A completed preview returns to the word-substitutions editor."""
        return result.completed

    @property
    def prompt_mode(self) -> PromptMode:
        if self.terminal_result is not None:
            return "auto_return" if self.auto_exit else "finished"
        if self.cancel_pending:
            return "cancel_pending"
        return "default"

    def _update_header(self) -> None:
        if not self.is_mounted:
            return
        now = self.finished_at if self.finished_at is not None else time.monotonic()
        elapsed = max(0.0, now - self.started_at)
        header = self.query_one(GenerationHeader)
        header.update_memory_text()
        header.update_status(self.phase)
        header.update_stats(
            1 if self.terminal_result is not None and self.terminal_result.completed else 0,
            1,
            elapsed,
        )
        header.update_hotkey(self.prompt_mode)

    def make_worker_reset_result(
        self,
        message: str,
        cause: HardResetCause,
    ) -> TtsPreviewResult:
        return TtsPreviewResult(
            GenerationTerminalStatus.WORKER_RESET,
            message=message,
            hard_reset_cause=cause,
        )


def _make_preview_failure(
    message: str, reset_cause: HardResetCause | None
) -> TtsPreviewResult:
    return TtsPreviewResult(
        GenerationTerminalStatus.FAILED,
        message=message,
        hard_reset_cause=reset_cause,
    )


def run_tts_preview_app(state: State, prompt: str) -> TtsPreviewResult:
    """Run one preview screen and return its generated in-memory sound."""
    app = TtsPreviewApp(state, prompt)
    try:
        result = app.run(inline=False)
    except Exception as exception:
        return session_failure_result(
            app,
            _make_preview_failure,
            f"{type(exception).__name__}: {exception}",
        )
    if result is None:
        return session_failure_result(
            app,
            _make_preview_failure,
            "Preview screen stopped without returning a result",
        )
    return result
