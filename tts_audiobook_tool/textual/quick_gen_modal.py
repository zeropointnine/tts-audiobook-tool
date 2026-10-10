"""In-place "quick" TTS generation as a modal over a live editor.

``QuickGenModal`` hosts one model-worker job (a single-line regeneration or a
literal diagnostic preview) on a ``ModalScreen``, so the editor underneath is
never torn down. The worker-job lifecycle comes from ``WorkerSessionMixin``;
this module supplies the modal's chrome (a two-row header, a divider, the
live worker log, and the review panel), the job specs, and the mapping from
worker events to a ``QuickGenResult``.

Lifecycle of the modal:

- A job with pre-flight problems submits nothing; the problems are listed in
  red and the modal waits for ESC.
- A line regeneration is staged (see ``segment_staging_util``). When the job
  leaves a staged candidate, the review panel appears and the new sound plays;
  ENTER keeps it, ESC discards it. The host applies the decision.
- A cleanly completed preview dismisses itself immediately.
- Anything else (cancelled, failed, reset, or completed with word errors and no
  candidate) stays open until ESC/ENTER so the output can be reviewed.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, cast

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Rule, Static

from tts_audiobook_tool import app_support, readiness
from tts_audiobook_tool.app_support.system_sleep import SystemSleepLock
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import (
    COL_ACCENT,
    COL_DEFAULT,
    COL_DIM,
    COL_DIM_ITALICS,
    COL_ERROR,
)
from tts_audiobook_tool.generation_events import (
    GenerationPhase,
    GenerationProgress,
    GenerationRunEnded,
)
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.model_worker_protocol import (
    GenerationFinished,
    GenerationTerminalStatus,
    GenerationUpdate,
    ModelWorkerEvent,
    TtsPreviewFinished,
    WorkerExited,
)
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.segment_staging_util import (
    describe_segment_word_errors,
    get_best_staged_segment,
    make_staging_dir_path,
)
from tts_audiobook_tool.sound.play_sound_util import PlaySoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.textual import worker_app
from tts_audiobook_tool.textual.generation_header import make_cancel_pending_prompt
from tts_audiobook_tool.textual.generation_app import (
    GenerationTranscript,
    make_generation_transcript_path,
)
from tts_audiobook_tool.textual.worker_content import WorkerLogContentArea
from tts_audiobook_tool.textual.worker_session import (
    FINAL_OUTPUT_SETTLE_SECONDS,
    WorkerSessionMixin,
)
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.worker_reset import HardResetCause, HardResetRequest


# A preview's worker prints only a heading and a few status lines.
PREVIEW_DIALOG_HEIGHT = "16"

# How often the review panel re-checks whether its sound is still playing.
REVIEW_PLAYBACK_POLL_SECONDS = 0.1

ReviewSound = Literal["original", "new"]


@dataclass(frozen=True)
class QuickGenResult:
    """Terminal result of one quick-generation modal session."""

    status: GenerationTerminalStatus
    message: str = ""
    # In-memory audio of a preview job (None for a line regeneration).
    sound: Sound | None = None
    # The worker's updated generation range (regeneration only; may be empty).
    remaining_range_string: str = ""
    failed_items: int = 0
    errored_items: int = 0
    hard_reset_cause: HardResetCause | None = None
    # False when pre-flight problems stopped the job before submission.
    submitted: bool = True
    # The staged candidate segment of a reviewable job ("" when none).
    staged_path: str = ""
    # Whether the user chose to keep the staged candidate.
    accepted: bool = False

    @property
    def completed(self) -> bool:
        return self.status is GenerationTerminalStatus.COMPLETED

    @property
    def completed_cleanly(self) -> bool:
        """Whether the job completed without any item needing attention."""
        return self.completed and not (self.failed_items or self.errored_items)


@dataclass(frozen=True)
class QuickGenReview:
    """Where a reviewable job stages its output, and what it would replace."""

    # Per-job directory the worker saves into (created by the worker).
    staging_dir: str
    # The line being regenerated.
    phrase_index: int
    # The line's current best segment, left in place during the job ("" if none).
    original_path: str = ""


@dataclass(frozen=True)
class QuickGenJob:
    """Specification of one job for the modal to host."""

    # Row-1 title, e.g. "Quick generate - line 42".
    title: str
    # Submits the worker job and returns its operation id.
    submit: Callable[[], str]
    # What the job is called in terminal labels ("Generation", "Preview").
    noun: str = "Generation"
    # Pre-flight problems; when non-empty nothing is submitted.
    problems: tuple[str, ...] = ()
    # Whether the console output is also saved to the project's gen log.
    use_transcript: bool = False
    # Dialog height as a CSS value. A preview prints little, so it is compact.
    height: str = "80%"
    # Reconcile persisted project state if the host closes without receiving
    # a dismissal result. Preview and staged jobs leave the project untouched.
    reconcile: Callable[[str], str] | None = None
    # Set for a staged line regeneration whose output awaits user review.
    review: QuickGenReview | None = None


def collect_preflight_problems(state: State, *, for_preview: bool) -> tuple[str, ...]:
    """Everything that should stop a quick job from starting, in one list.

    Checks the run (or preview) readiness blockers, the voice samples
    (including missing transcripts, which the modal cannot auto-transcribe),
    and whether the shared model worker is already busy.
    """
    # Main-process UI flow only; lazy import keeps this module importable
    # without the menu chain.
    from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared

    problems: list[str] = []
    blocker_text = (
        readiness.get_tts_preview_blocker_text(state, verbose=True)
        if for_preview
        else readiness.get_run_blocker_text(state, verbose=True)
    )
    if blocker_text:
        problems.append(blocker_text)
    problems.extend(VoiceMenuShared.get_voice_problems(state))
    if ModelWorker.is_busy():
        problems.append("The model worker is busy with another job")
    return tuple(problems)


def make_generation_job(state: State, phrase_index: int) -> QuickGenJob:
    """Job that regenerates exactly one project line into a staging directory.

    Nothing in the project changes until the host accepts the staged result.
    """
    project = state.project
    best_segment = project.sound_segments.get_best_item_for(phrase_index)
    review = QuickGenReview(
        staging_dir=make_staging_dir_path(project.dir_path),
        phrase_index=phrase_index,
        original_path=(
            os.path.join(project.sound_segments_path, best_segment.file_name)
            if best_segment is not None
            else ""
        ),
    )
    return QuickGenJob(
        title=f"Quick generate - line {phrase_index + 1}",
        noun="Generation",
        problems=collect_preflight_problems(state, for_preview=False),
        use_transcript=True,
        review=review,
        submit=lambda: ModelWorker.submit_generation(
            state=state,
            indices={phrase_index},
            batch_size=ProjectVoiceUtil.get_batch_size(project),
            is_regen=True,
            staging_dir=review.staging_dir,
        ),
    )


def make_preview_job(state: State, title: str, prompt: str) -> QuickGenJob:
    """Job that speaks one literal prompt without persisting anything."""
    return QuickGenJob(
        title=title,
        noun="Preview generation",
        problems=collect_preflight_problems(state, for_preview=True),
        height=PREVIEW_DIALOG_HEIGHT,
        submit=lambda: ModelWorker.submit_tts_preview(
            state=state,
            prompt=prompt,
            apply_word_substitutions=False,
        ),
    )


class QuickGenHeader(Vertical):
    """Two-row modal header.

    Row 1 is the title (left) and the status (right-justified); row 2 is the
    key prompt.
    """

    DEFAULT_CSS = """
    QuickGenHeader {
        height: auto;
        padding: 0 1;
    }
    QuickGenHeader .quick-gen-row {
        height: 1;
        width: 100%;
    }
    QuickGenHeader #quick-gen-title {
        width: 1fr;
        height: 1;
    }
    QuickGenHeader #quick-gen-status {
        width: auto;
        height: 1;
    }
    QuickGenHeader #quick-gen-prompt-row {
        height: 1;
        width: 100%;
    }
    """

    def __init__(self, title: str, *, id: str | None = None) -> None:
        super().__init__(id=id)
        self.title_text = title

    def compose(self) -> ComposeResult:
        with Horizontal(classes="quick-gen-row"):
            yield Static(
                Text.from_ansi(f"{COL_ACCENT}{self.title_text}"),
                id="quick-gen-title",
            )
            yield Static("", id="quick-gen-status")
        yield Static("", id="quick-gen-prompt-row", markup=False)

    def update_status(self, status: str) -> None:
        """Render "Status: <status>" right-justified on row 1."""
        text = f"Status: {COL_DIM_ITALICS}{status}{COL_DEFAULT}" if status else ""
        self.query_one("#quick-gen-status", Static).update(Text.from_ansi(text))

    def update_prompt(self, text: str) -> None:
        """Render the key prompt (ANSI allowed) on row 2."""
        self.query_one("#quick-gen-prompt-row", Static).update(Text.from_ansi(text))


class QuickGenModal(WorkerSessionMixin[QuickGenResult], ModalScreen[QuickGenResult]):
    """Modal that runs one quick job over the editor and dismisses with its result."""

    AUTO_FOCUS = None

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("escape", "cancel_or_continue", show=False, priority=True),
        Binding("enter", "continue", show=False, priority=True),
        Binding("1", "play_review_sound('original')", show=False),
        Binding("2", "play_review_sound('new')", show=False),
    ]

    CSS = """
    QuickGenModal {
        align: center middle;
        background: transparent;
    }

    #quick-gen-dialog {
        width: 90%;
        max-width: 120;
        height: 80%;
        border: round #888888;
        background: ansi_default;
    }

    #quick-gen-divider, #quick-gen-review-divider {
        color: #888888;
        margin: 0;
    }

    #quick-gen-review {
        height: auto;
        display: none;
    }

    #quick-gen-review-text {
        height: auto;
        padding: 0 1;
    }
    """

    def __init__(self, state: State, job: QuickGenJob) -> None:
        super().__init__()
        self._init_session(state)
        self.job = job
        self.failed_items = 0
        self.errored_items = 0
        self.transcript: GenerationTranscript | None = None
        self.sleep_lock: SystemSleepLock | None = None
        self._result_delivered = False
        # Set while the review panel awaits the keep/discard decision.
        self.review_pending = False
        self.review_sound_id = ""
        self.review_playing: ReviewSound | None = None

    # ----------------------------------------------------------------
    # composition and startup
    # ----------------------------------------------------------------

    def compose(self) -> ComposeResult:
        with Vertical(id="quick-gen-dialog"):
            yield QuickGenHeader(self.job.title, id="quick-gen-header")
            yield Rule(id="quick-gen-divider")
            yield WorkerLogContentArea(
                output_filters=Tts.get_info(self.state.project).output_filters,
                id="quick-gen-output-shell",
            )
            with Vertical(id="quick-gen-review"):
                yield Rule(id="quick-gen-review-divider")
                yield Static("", id="quick-gen-review-text")

    def on_mount(self) -> None:
        self.query_one("#quick-gen-dialog").styles.height = self.job.height
        self.query_one(WorkerLogContentArea).worker_log.focus()
        if self.job.problems:
            self._show_preflight_problems()
            return
        if self.job.use_transcript:
            self.transcript = GenerationTranscript(
                make_generation_transcript_path(self.state.project.dir_path),
                enabled=self.state.prefs.save_gen_log,
            )
        self.sleep_lock = SystemSleepLock()
        self._start_session(worker_app.EVENT_POLL_SECONDS, 1.0)

    async def on_unmount(self) -> None:
        """Do not strand an owned job when the editor exits or its UI fails."""
        self._session_closed = True
        self._stop_review_playback()
        try:
            if self._result_delivered or self.operation_id is None:
                return
            result = self.terminal_result
            if result is None:
                message = "Quick-generation interface closed before returning a result"
                cause = None
                # request_cancel is operation-id guarded: an old screen must
                # never terminate an unrelated successor job. A reset already
                # in progress still needs to finish before reconciling files.
                if self.reset_in_progress or ModelWorker.request_cancel(self.operation_id):
                    request = self.reset_request or HardResetRequest(
                        HardResetCause.INTERFACE_FAILURE, message
                    )
                    outcome = await asyncio.to_thread(self._perform_hard_reset_once, request)
                    message = outcome.message
                    cause = outcome.request.cause
                result = QuickGenResult(
                    GenerationTerminalStatus.FAILED,
                    message=message,
                    hard_reset_cause=cause,
                )
                self.terminal_result = result
            if self.job.reconcile is not None:
                save_error = self.job.reconcile(result.remaining_range_string)
                if save_error:
                    raise RuntimeError(save_error)
        finally:
            self._release_resources()

    def _release_resources(self) -> None:
        if self.sleep_lock is not None:
            self.sleep_lock.release()
        if self.transcript is not None:
            self.transcript.close()

    def _show_preflight_problems(self) -> None:
        """List every problem in red; nothing is submitted."""
        message = "\n".join(
            f"{COL_ERROR}{problem}{COL_DEFAULT}" for problem in self.job.problems
        )
        self._show_terminal_summary(
            QuickGenResult(
                GenerationTerminalStatus.FAILED,
                message=message,
                submitted=False,
            )
        )

    def submit_worker_job(self) -> str:
        return self.job.submit()

    def _on_submit_failure(self, message: str) -> None:
        self._finalize_failure(message)

    # ----------------------------------------------------------------
    # worker events
    # ----------------------------------------------------------------

    def _handle_session_event(self, event: ModelWorkerEvent) -> None:
        if isinstance(event, GenerationUpdate):
            self._handle_update(event.update)
        elif isinstance(event, (GenerationFinished, TtsPreviewFinished)):
            self._begin_finish(event)

    def _handle_update(self, update: object) -> None:
        # Program-requested resets override a pending cancellation.
        if self._begin_hard_reset_for_update(update):
            self._update_header()
            return
        if isinstance(update, GenerationPhase):
            self.phase = update.label
        elif isinstance(update, GenerationProgress):
            if update.current_indices:
                self._expect_separator()
            self.failed_items = update.failed
            self.errored_items = update.errored
        elif isinstance(update, GenerationRunEnded):
            self._arm_trailing_divider()
        self._update_header()

    def _begin_finish(self, event: GenerationFinished | TtsPreviewFinished) -> None:
        if self.reset_in_progress or self.finishing or self.terminal_result is not None:
            return
        self.finishing = True
        self.phase = event.status.value.replace("_", " ").title()
        self._update_header()
        self.set_timer(
            FINAL_OUTPUT_SETTLE_SECONDS,
            lambda: self._finish_from_worker_event(event),
        )

    def _finish_from_worker_event(
        self, event: GenerationFinished | TtsPreviewFinished
    ) -> None:
        self._finalize_console()
        if isinstance(event, GenerationFinished):
            result = QuickGenResult(
                status=event.status,
                message=event.message,
                remaining_range_string=event.remaining_range_string,
                failed_items=self.failed_items,
                errored_items=self.errored_items,
                staged_path=self._find_staged_candidate(event.status),
            )
        else:
            sound = cast(Sound | None, event.sound)
            status = event.status
            message = event.message
            if status is GenerationTerminalStatus.COMPLETED and sound is None:
                status = GenerationTerminalStatus.FAILED
                message = "Preview generation returned no audio"
            result = QuickGenResult(status=status, message=message, sound=sound)
        self._show_terminal_summary(result)

    def _on_worker_command_failed(self, message: str) -> None:
        self._finalize_failure(message)

    def _on_worker_exit(self, event: WorkerExited) -> None:
        # The synthesized worker-death event; a reset in progress owns the
        # terminal summary instead (the drain loop already skips it then).
        self._finalize_failure(event.message or "Model worker exited unexpectedly")

    def _finalize_failure(self, message: str) -> None:
        if self.terminal_result is not None:
            return
        self.finishing = True
        if self.is_mounted:
            self._finalize_console()
        self._show_terminal_summary(
            QuickGenResult(GenerationTerminalStatus.FAILED, message=message)
        )

    def make_worker_reset_result(
        self, message: str, cause: HardResetCause
    ) -> QuickGenResult:
        return QuickGenResult(
            GenerationTerminalStatus.WORKER_RESET,
            message=message,
            hard_reset_cause=cause,
        )

    # ----------------------------------------------------------------
    # console / transcript
    # ----------------------------------------------------------------

    def _record_console_output(self, text: str) -> None:
        if self.transcript is not None:
            self.transcript.write_chunk(text)

    def _append_application_lines(self, lines: list[str]) -> None:
        self._append_lines(lines)
        if self.transcript is not None:
            self.transcript.write_lines(lines)

    # ----------------------------------------------------------------
    # terminal summary
    # ----------------------------------------------------------------

    def _terminal_labels(self) -> dict[GenerationTerminalStatus, str]:
        noun = self.job.noun
        return {
            GenerationTerminalStatus.COMPLETED: f"{noun} completed.",
            GenerationTerminalStatus.CANCELLED: f"{noun} cancelled.",
            GenerationTerminalStatus.ABORTED: f"{noun} stopped.",
            GenerationTerminalStatus.FAILED: f"{noun} failed.",
            GenerationTerminalStatus.WORKER_RESET: f"{noun} stopped; model worker was reset.",
        }

    def should_auto_exit(self, result: QuickGenResult) -> bool:
        """A cleanly completed preview returns to the editor at once; a staged
        regeneration always waits for the user."""
        return result.completed_cleanly and self.job.review is None

    def _suppress_terminal_summary_ui(self) -> bool:
        # Nothing worth flashing for the instant before the auto-dismissal.
        return self.auto_exit

    def terminal_label(self, result: QuickGenResult) -> str:
        if not result.submitted:
            return "Cannot generate audio"
        return self._terminal_labels()[result.status]

    def terminal_display_label(self, result: QuickGenResult) -> str:
        label = self.terminal_label(result)
        if not result.submitted:
            return f"{COL_ERROR}{label}{COL_DEFAULT}"
        if result.status is GenerationTerminalStatus.CANCELLED:
            return f"{COL_DIM_ITALICS}{label}{COL_DEFAULT}"
        return label

    def terminal_summary_extra_lines(self, result: QuickGenResult) -> list[str]:
        if self.transcript is not None and self.transcript.path:
            from tts_audiobook_tool import text_util

            link = text_util.make_terminal_hyperlink(self.transcript.path, is_file=True)
            return [f"Transcript: {link}"]
        return []

    def _pre_terminal_summary(self, result: QuickGenResult) -> None:
        # The worker job is over: release the sleep lock now so an idle machine
        # is not held awake while the user reads the summary.
        if self.sleep_lock is not None:
            self.sleep_lock.release()
        super()._pre_terminal_summary(result)

    def _post_terminal_summary(self, result: QuickGenResult) -> None:
        if (
            result.status is GenerationTerminalStatus.WORKER_RESET
            and result.hard_reset_cause is not None
            and result.hard_reset_cause.should_alert
        ):
            app_support.play_fatal_gen_sound()
        super()._post_terminal_summary(result)
        if result.staged_path:
            self._show_review()

    # ----------------------------------------------------------------
    # review
    # ----------------------------------------------------------------

    def _find_staged_candidate(self, status: GenerationTerminalStatus) -> str:
        """The staged segment to review, for a completed staged job only."""
        if self.job.review is None or status is not GenerationTerminalStatus.COMPLETED:
            return ""
        review = self.job.review
        best = get_best_staged_segment(review.staging_dir, review.phrase_index)
        return str(best[0]) if best is not None else ""

    def _review_path(self, which: ReviewSound) -> str:
        result = self.terminal_result
        if which == "new":
            return result.staged_path if result is not None else ""
        return self.job.review.original_path if self.job.review is not None else ""

    def _show_review(self) -> None:
        self.review_pending = True
        self.query_one("#quick-gen-review").display = True
        self._update_header()
        self._update_review_text()
        self.set_interval(REVIEW_PLAYBACK_POLL_SECONDS, self._poll_review_playback)
        self.action_play_review_sound("new")

    def _make_review_line(self, key: str, which: ReviewSound, label: str) -> str:
        path = self._review_path(which)
        if not path:
            return f"{COL_DIM}[{key}] {label} (none){COL_DEFAULT}"
        details = describe_segment_word_errors(path)
        details_text = f"  {COL_DIM}({details}){COL_DEFAULT}" if details else ""
        playing_text = (
            f"  {COL_DIM_ITALICS}playing{COL_DEFAULT}"
            if self.review_playing == which
            else ""
        )
        return (
            f"[{COL_ACCENT}{key}{COL_DEFAULT}] {label}{details_text}{playing_text}"
        )

    def _update_review_text(self) -> None:
        lines = [
            self._make_review_line("1", "original", "Play original sound"),
            self._make_review_line("2", "new", "Play new sound"),
            f"Press [{COL_ACCENT}ENTER{COL_DEFAULT}] to keep new sound, "
            f"[{COL_ACCENT}ESC{COL_DEFAULT}] to discard",
        ]
        self.query_one("#quick-gen-review-text", Static).update(
            Text.from_ansi("\n".join(lines))
        )

    def action_play_review_sound(self, which: ReviewSound) -> None:
        """Play the original or new sound; the playing one toggles off."""
        if not self.review_pending:
            return
        was_playing = self.review_playing
        self._stop_review_playback()
        path = self._review_path(which)
        if was_playing != which and path:
            sound_id, error = PlaySoundUtil.play_sound_file_async(path)
            if error:
                self.notify(f"Couldn't play sound: {error}", severity="error")
            else:
                self.review_sound_id = sound_id
                self.review_playing = which
        self._update_review_text()

    def _poll_review_playback(self) -> None:
        if self.review_sound_id and PlaySoundUtil.current_sound_id() != self.review_sound_id:
            self.review_sound_id = ""
            self.review_playing = None
            self._update_review_text()

    def _stop_review_playback(self) -> None:
        if self.review_sound_id and PlaySoundUtil.current_sound_id() == self.review_sound_id:
            PlaySoundUtil.stop_sound_async()
        self.review_sound_id = ""
        self.review_playing = None

    def _finish_review(self, accepted: bool) -> None:
        result = self.terminal_result
        assert result is not None
        self.review_pending = False
        self._stop_review_playback()
        self._leave_session(replace(result, accepted=accepted))

    def action_continue(self) -> None:
        if self.review_pending:
            self._finish_review(accepted=True)
            return
        super().action_continue()

    def action_cancel_or_continue(self) -> None:
        if self.review_pending:
            self._finish_review(accepted=False)
            return
        super().action_cancel_or_continue()

    # ----------------------------------------------------------------
    # header and exit
    # ----------------------------------------------------------------

    def _prompt_text(self) -> str:
        if self.review_pending:
            # The review panel carries the prompt.
            return ""
        if self.terminal_result is not None:
            return "" if self.auto_exit else f"Press [{COL_ACCENT}ESC{COL_DEFAULT}] to close"
        if self.cancel_pending:
            return make_cancel_pending_prompt()
        return f"Press [{COL_ACCENT}ESC{COL_DEFAULT}] to cancel"

    def _update_header(self) -> None:
        headers = self.query(QuickGenHeader)
        if not headers:
            return
        header = headers.first()
        header.update_status(self.phase)
        header.update_prompt(self._prompt_text())

    def _leave_session(self, result: QuickGenResult) -> None:
        self.dismiss(result)
        self._result_delivered = True
        self._release_resources()
