from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import datetime
from enum import Enum
import time
from typing import TYPE_CHECKING, Any

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan, PlannedChunk
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile, RangeAttribution
from tts_audiobook_tool.speaker_assign.attribution_prompt_util import (
    ATTRIBUTION_SYSTEM_PROMPT,
    AttributionPromptUtil,
)
from tts_audiobook_tool.speaker_assign.attribution_reply_util import AttributionReplyUtil, ParsedReply
from tts_audiobook_tool.speaker_assign.chunk_plan_util import DEFAULT_CHUNK_WORDS
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult, LlmRequestUtil
from tts_audiobook_tool.speaker_assign.roster import Roster
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil
from tts_audiobook_tool.util import make_error_string

if TYPE_CHECKING:
    from tts_audiobook_tool.prefs import Prefs


# Seconds to wait before the 2nd and 3rd attempt at a failed chunk
CHUNK_RETRY_DELAYS = (10.0, 30.0)

# This many chunks failing in a row (each after all its attempts) stops the
# run: the cause is likely not chunk-specific (eg the server is down).
MAX_CONSECUTIVE_FAILED_CHUNKS = 2

# How often a retry wait checks for a stop request
STOP_POLL_SECONDS = 0.1


class AttributionError(Exception):
    """
    A failed attribution request. `retryable`: repeating it may succeed.
    `fatal`: the error would affect every chunk (eg a bad API key), so the run
    stops; otherwise only the chunk fails (by default, errors that aren't
    retryable are fatal).
    """

    def __init__(self, message: str, retryable: bool, fatal: bool | None = None) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.fatal = (not retryable) if fatal is None else fatal


class AttributionCancelled(Exception):
    """A request was cancelled (see LlmRequestUtil.send_cancellable).
    `abandoned`: given up before its response started."""

    def __init__(self, abandoned: bool) -> None:
        super().__init__("Cancelled")
        self.abandoned = abandoned


class _StopRequested(Exception):
    pass


class RunOutcome(Enum):
    FINISHED = "finished"     # every remaining chunk was attempted
    STOPPED = "stopped"       # stop requested; stopped between requests
    CANCELLED = "cancelled"   # a request was cancelled mid-chunk
    FAILED = "failed"         # a non-retryable error, repeated failures or a save error


@dataclass
class AttributionRunResult:
    outcome: RunOutcome
    attributions: AttributionsFile
    saved_count: int = 0
    failures: list[tuple[PlannedChunk, str]] = field(default_factory=list)
    error: str = ""            # outcome FAILED
    abandoned: bool = False    # outcome CANCELLED


class AttributionRunListener:
    """Progress hooks for AttributionRunner.run (no-ops by default)."""

    def chunk_started(self, chunk: PlannedChunk, plan: AssignPlan) -> None: ...

    def note(self, text: str) -> None: ...

    def retrying(self, attempt: int, delay: float, error: str) -> None: ...

    def chunk_saved(self, chunk: PlannedChunk, attribution: RangeAttribution, elapsed: float) -> None: ...

    def chunk_failed(self, chunk: PlannedChunk, error: str) -> None: ...


# Sends one message in a session and returns the result (normally
# LlmRequestUtil.send_cancellable with the caller's progress display)
SendFunction = Callable[[LlmSession, str], LlmRequestResult]


class AttributionRunner:
    """
    Step 2, speaker attribution of the whole book: the remaining chunks of the
    AssignPlan, in reading order, one fresh LLM session per chunk. Results are
    saved after every chunk, so a stopped or partly failed run continues where
    it left off when run again; sequential processing lets each chunk's
    context carry the previous chunks' speakers.
    """

    def __init__(
            self,
            book: Book,
            roster: Roster,
            attributions: AttributionsFile,
            project_dir: str,
            prefs: Prefs,
            send: SendFunction,
            stop_requested: Callable[[], bool] = lambda: False,
            listener: AttributionRunListener | None = None,
            make_session: Callable[[], LlmSession] | None = None,
            retry_delays: tuple[float, ...] = CHUNK_RETRY_DELAYS,
            chunk_words: int = DEFAULT_CHUNK_WORDS,
    ) -> None:
        self.book = book
        self.roster = roster
        self.attributions = attributions
        self.project_dir = project_dir
        self.prefs = prefs
        self.send = send
        self.stop_requested = stop_requested
        self.listener = listener or AttributionRunListener()
        self.make_session = make_session or (
            lambda: LlmRequestUtil.make_session(prefs, ATTRIBUTION_SYSTEM_PROMPT))
        self.retry_delays = retry_delays
        self.chunk_words = chunk_words

    def run(self) -> AttributionRunResult:
        """Attributes and saves each remaining chunk. Stops between chunks
        when stop_requested() turns true. Never raises for request errors."""
        plan = AssignPlan.make(self.book, self.attributions, self.chunk_words)
        result = AttributionRunResult(RunOutcome.FINISHED, self.attributions)
        consecutive_failures = 0
        for chunk in plan.remaining:
            if self.stop_requested():
                result.outcome = RunOutcome.STOPPED
                break
            self.listener.chunk_started(chunk, plan)
            started = time.monotonic()
            try:
                attribution = self._attribute_with_retries(chunk.start, chunk.end)
            except _StopRequested:
                result.outcome = RunOutcome.STOPPED
                break
            except AttributionCancelled as e:
                result.outcome, result.abandoned = RunOutcome.CANCELLED, e.abandoned
                break
            except AttributionError as e:
                if e.fatal:
                    result.outcome, result.error = RunOutcome.FAILED, str(e)
                    break
                self.listener.chunk_failed(chunk, str(e))
                result.failures.append((chunk, str(e)))
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILED_CHUNKS:
                    result.outcome = RunOutcome.FAILED
                    result.error = f"Stopped after {consecutive_failures} chunks in a row failed. Last error:\n{e}"
                    break
                continue
            consecutive_failures = 0
            self.attributions = self.attributions.with_range(attribution)
            error = SpeakerAssignFileUtil.save_attributions_file(self.project_dir, self.attributions)
            if error:
                result.outcome, result.error = RunOutcome.FAILED, error
                break
            result.saved_count += 1
            self.listener.chunk_saved(chunk, attribution, time.monotonic() - started)
        result.attributions = self.attributions
        return result

    def attribute_range(self, start: int, end: int) -> RangeAttribution:
        """
        Attributes [start, end) in one fresh session: one request, plus one
        follow-up request for ids missing from the reply; ids still missing
        are UNKNOWN. Context comes from the runner's current attributions.
        Raises AttributionError or AttributionCancelled.
        """
        utterances = UtteranceUtil.find_utterances(self.book, start, end)
        message = AttributionPromptUtil.build_message(
            self.book, start, end, self.roster, self.attributions.speakers_by_line(self.book))
        try:
            llm = self.make_session()
        except ValueError as e:
            raise AttributionError(f"Can't use the LLM settings: {e}", retryable=False)

        requested = set(range(1, len(utterances) + 1))
        parsed = self._parse(self._request(llm, message), requested)
        if parsed.missing:
            self.listener.note(f"{len(parsed.missing)} id(s) missing from the reply; asking again for those")
            try:
                follow_up = self._request(llm, AttributionPromptUtil.missing_ids_message(parsed.missing))
                parsed = AttributionReplyUtil.merge(parsed, self._parse(follow_up, set(parsed.missing)))
            except AttributionError as e:
                if e.fatal:
                    raise
                self.listener.note(f"Follow-up request failed: {e}")
        if parsed.missing:
            self.listener.note(f"{len(parsed.missing)} id(s) still missing; marked UNKNOWN")
        return AttributionReplyUtil.make_range_attribution(
            self.book, start, end, utterances, parsed,
            created=datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            llm=LlmRequestUtil.describe_llm(self.prefs),
        )

    def _attribute_with_retries(self, start: int, end: int) -> RangeAttribution:
        """attribute_range, repeated after retryable errors (with a fresh
        session, after a wait that a stop request ends)."""
        for attempt, delay in enumerate((*self.retry_delays, None), 1):
            try:
                return self.attribute_range(start, end)
            except AttributionError as e:
                if not e.retryable or delay is None:
                    raise
                self.listener.retrying(attempt, delay, str(e))
                if not self._wait(delay):
                    raise _StopRequested()
        raise AssertionError("unreachable")

    def _request(self, llm: LlmSession, message: str) -> str:
        result = self.send(llm, message)
        if result.cancelled:
            raise AttributionCancelled(result.abandoned)
        if result.error is not None:
            raise AttributionError(
                LlmRequestUtil.describe_error(result.error, self.prefs),
                retryable=LlmRequestUtil.is_retryable(result.error),
            )
        if result.hit_output_limit:
            # Would most likely happen again: not retried, but only this chunk fails
            raise AttributionError(LlmRequestUtil.describe_output_limit(result), retryable=False, fatal=False)
        if not result.reply.strip():
            # The provider didn't report why (otherwise caught above)
            hint = (" The model reasoned but never answered: it most likely hit its output-token limit."
                    if result.reasoning else "")
            raise AttributionError(f"Empty reply from the LLM.{hint}", retryable=True)
        return result.reply

    def _parse(self, reply: str, requested: set[int]) -> ParsedReply:
        try:
            value: Any = AttributionReplyUtil.extract_json(reply)
            return AttributionReplyUtil.parse_reply(value, requested, self.roster.ids)
        except Exception as e:
            raise AttributionError(f"Couldn't read the LLM's reply: {make_error_string(e)}", retryable=True)

    def _wait(self, seconds: float) -> bool:
        """Waits, unless a stop is requested; returns False if one was."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stop_requested():
                return False
            time.sleep(min(STOP_POLL_SECONDS, max(0.0, deadline - time.monotonic())))
        return not self.stop_requested()
