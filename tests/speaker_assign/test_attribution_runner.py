import json
import re

import requests

from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.speaker_assign.assign_plan import AssignPlan
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.attribution_runner import (
    AttributionRunListener,
    AttributionRunner,
    RunOutcome,
)
from tts_audiobook_tool.speaker_assign.llm_request_util import LlmRequestResult
from tts_audiobook_tool.speaker_assign.roster import Roster, RosterCharacter
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil
from tts_audiobook_tool.speaker_assign.speaker_assign_types import UNKNOWN

from tests.speaker_assign.speaker_assign_test_support import DIALOG, NARRATION, make_book
from types import SimpleNamespace

# Small chunks, so each section of the test book is its own chunk
CHUNK_WORDS = 30

ROSTER = Roster(characters=(RosterCharacter(id="ANNA", name="Anna"), RosterCharacter(id="BEN", name="Ben")))
PREFS = SimpleNamespace(llm_url="https://example.test/v1", llm_model="m", llm_speaker_assign_extra_params={})


def make_section(n: int):
    narration = " ".join(["word"] * 22)
    return [
        (f"{narration} {n}.\n\n", Reason.PARAGRAPH, NARRATION),
        (f"“Hello {n}.” ", Reason.PHRASE, DIALOG),
        ("she said. ", Reason.PHRASE, NARRATION),
        (f"“Bye {n}.”\n\n", Reason.PARAGRAPH, DIALOG),
    ]


BOOK = make_book(make_section(1), make_section(2), make_section(3))


def ids_in(message: str) -> list[str]:
    """The utterance ids requested in an attribution message (or follow-up)."""
    if "omitted these ids:" in message:
        return re.search(r"omitted these ids: ([\d, ]+)\.", message)[1].split(", ")  # type: ignore[index]
    return re.findall(r'<q id="(\d+)">', message.split("TEXT TO ATTRIBUTE")[1])


def reply(message: str, speaker: str = "ANNA", skip: tuple[str, ...] = ()) -> LlmRequestResult:
    speakers = {i: speaker for i in ids_in(message) if i not in skip}
    return LlmRequestResult(reply=json.dumps({"speakers": speakers, "uncertain": []}), reasoning="", elapsed=0)


class ScriptedSend:
    """A send function answering each request with the next scripted result
    (a callable taking the message); records the messages."""

    def __init__(self, *script):
        self.script = list(script)
        self.messages: list[str] = []

    def __call__(self, _llm, message):
        self.messages.append(message)
        return self.script.pop(0)(message)


def make_runner(tmp_path, send, attributions=AttributionsFile(), **kwargs):
    return AttributionRunner(
        book=BOOK, roster=ROSTER, attributions=attributions, project_dir=str(tmp_path),
        prefs=PREFS, send=send, make_session=lambda: None, retry_delays=(0.0, 0.0),  # type: ignore
        chunk_words=CHUNK_WORDS, **kwargs)


def error_result(error: Exception) -> LlmRequestResult:
    return LlmRequestResult(reply="", reasoning="", elapsed=0, error=error)


def http_error(status: int) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = b"error"
    return requests.HTTPError(str(status), response=response)


def test_test_book_has_one_chunk_per_section() -> None:
    plan = AssignPlan.make(BOOK, AttributionsFile(), CHUNK_WORDS)
    assert [(c.number, c.utterance_count, c.done) for c in plan.chunks] == [(1, 2, False), (2, 2, False), (3, 2, False)]


def test_run_saves_every_chunk_and_carries_context(tmp_path) -> None:
    # Each chunk is saved as it completes; a later request's context tags the
    # previous chunk's dialog with the speaker it was given.
    send = ScriptedSend(reply, lambda m: reply(m, "BEN"), reply)
    result = make_runner(tmp_path, send).run()
    assert result.outcome == RunOutcome.FINISHED and result.saved_count == 3
    assert AssignPlan.make(BOOK, result.attributions, CHUNK_WORDS).is_complete
    saved = SpeakerAssignFileUtil.load_attributions_file(str(tmp_path))
    assert isinstance(saved, AttributionsFile) and len(saved.ranges) == 3
    assert '<q speaker="ANNA">“Bye 1.”</q>' in send.messages[1]
    assert '<q speaker="BEN">“Bye 2.”</q>' in send.messages[2]


def test_run_resumes_after_done_chunks(tmp_path) -> None:
    first = make_runner(tmp_path, ScriptedSend(reply), stop_requested=lambda: False)
    # Stop once the first chunk is saved
    first.stop_requested = lambda: first.attributions.ranges != ()
    partial = first.run()
    assert partial.outcome == RunOutcome.STOPPED and partial.saved_count == 1

    send = ScriptedSend(reply, reply)
    result = make_runner(tmp_path, send, attributions=partial.attributions).run()
    assert result.saved_count == 2 and len(send.messages) == 2
    assert AssignPlan.make(BOOK, result.attributions, CHUNK_WORDS).is_complete


def test_starting_over_replaces_saved_results_only_at_the_first_new_save(tmp_path) -> None:
    # "Start over" runs from empty attributions over a complete saved run: the
    # saved results survive a run that saves nothing, and the first new save
    # replaces all of them.
    make_runner(tmp_path, ScriptedSend(reply, reply, reply)).run()

    failed = make_runner(tmp_path, ScriptedSend(lambda _: error_result(http_error(401)))).run()
    assert failed.outcome == RunOutcome.FAILED and failed.saved_count == 0
    saved = SpeakerAssignFileUtil.load_attributions_file(str(tmp_path))
    assert isinstance(saved, AttributionsFile) and len(saved.ranges) == 3

    runner = make_runner(tmp_path, ScriptedSend(lambda m: reply(m, "BEN")))
    runner.stop_requested = lambda: runner.attributions.ranges != ()
    partial = runner.run()
    assert partial.outcome == RunOutcome.STOPPED and partial.saved_count == 1
    saved = SpeakerAssignFileUtil.load_attributions_file(str(tmp_path))
    assert isinstance(saved, AttributionsFile)
    [only] = saved.ranges
    assert {u.speaker for u in only.utterances} == {"BEN"}


def test_missing_ids_get_one_follow_up_then_become_unknown(tmp_path) -> None:
    send = ScriptedSend(
        lambda m: reply(m, skip=("2",)),   # chunk 1: id 2 missing...
        lambda m: reply(m, skip=("2",)),   # ...and still missing after the follow-up
        reply, reply)
    notes = []

    class Listener(AttributionRunListener):
        def note(self, text):
            notes.append(text)

    result = make_runner(tmp_path, send, listener=Listener()).run()
    assert "omitted these ids: 2." in send.messages[1]
    first = result.attributions.ranges[0]
    assert [u.speaker for u in first.utterances] == ["ANNA", UNKNOWN]
    assert len(notes) == 2


def test_retryable_errors_are_retried_with_a_fresh_request(tmp_path) -> None:
    # A dropped connection, then an unparseable reply, then success
    send = ScriptedSend(
        lambda _: error_result(requests.ConnectionError("reset")),
        lambda _: LlmRequestResult(reply="not json", reasoning="", elapsed=0),
        reply, reply, reply)
    result = make_runner(tmp_path, send).run()
    assert result.outcome == RunOutcome.FINISHED and result.saved_count == 3
    assert len(send.messages) == 5


def test_non_retryable_error_stops_the_run_at_once(tmp_path) -> None:
    # A bad API key would fail the same way for every chunk: no retries.
    send = ScriptedSend(lambda _: error_result(http_error(401)))
    result = make_runner(tmp_path, send).run()
    assert result.outcome == RunOutcome.FAILED and "HTTP 401" in result.error
    assert len(send.messages) == 1


def test_a_failed_chunk_is_skipped_but_repeated_failures_stop_the_run(tmp_path) -> None:
    server_error = lambda _: error_result(http_error(503))  # noqa: E731
    # Chunk 1 fails all 3 attempts, chunk 2 succeeds, chunk 3 fails: kept going
    send = ScriptedSend(server_error, server_error, server_error, reply, server_error, server_error, server_error)
    result = make_runner(tmp_path, send).run()
    assert result.outcome == RunOutcome.FINISHED
    assert [chunk.number for chunk, _ in result.failures] == [1, 3]
    assert result.saved_count == 1

    # Two chunks in a row failing stops the run (the cause is likely not chunk-specific)
    send = ScriptedSend(*[server_error] * 6)
    result = make_runner(tmp_path / "b", send).run()
    assert result.outcome == RunOutcome.FAILED and "2 chunks in a row" in result.error
    assert len(send.messages) == 6


def test_cancelled_request_stops_without_saving_the_chunk(tmp_path) -> None:
    cancelled = lambda _: LlmRequestResult(reply="{", reasoning="", elapsed=0, cancelled=True, abandoned=True)  # noqa: E731
    result = make_runner(tmp_path, ScriptedSend(reply, cancelled)).run()
    assert result.outcome == RunOutcome.CANCELLED and result.abandoned
    assert result.saved_count == 1 and len(result.attributions.ranges) == 1


def test_stop_during_retry_wait_ends_the_run(tmp_path) -> None:
    # A stop request is honored during the wait before a retry, not only
    # between chunks.
    stop = {"now": False}

    class Listener(AttributionRunListener):
        def retrying(self, attempt, delay, error):
            stop["now"] = True

    send = ScriptedSend(lambda _: error_result(requests.ConnectionError("reset")))
    runner = make_runner(tmp_path, send, listener=Listener(), stop_requested=lambda: stop["now"])
    runner.retry_delays = (5.0, 5.0)
    result = runner.run()
    assert result.outcome == RunOutcome.STOPPED and len(send.messages) == 1


def test_output_limit_hit_fails_the_chunk_without_retries(tmp_path) -> None:
    # Hitting the output-token limit would most likely repeat, so the chunk
    # isn't retried; the run continues with the next chunk.
    limit = lambda _: LlmRequestResult(reply="", reasoning="...", elapsed=0, finish_reason="length")  # noqa: E731
    send = ScriptedSend(limit, reply, reply)
    result = make_runner(tmp_path, send).run()
    assert result.outcome == RunOutcome.FINISHED and result.saved_count == 2
    assert len(send.messages) == 3
    [(chunk, error)] = result.failures
    assert chunk.number == 1 and "output-token limit while still thinking" in error

    # A cut-off answer (limit hit partway through it) is reported the same way
    cut_off = lambda _: LlmRequestResult(reply='{"speakers": {"1": "AN', reasoning="", elapsed=0,  # noqa: E731
                                         finish_reason="length")
    result = make_runner(tmp_path / "b", ScriptedSend(cut_off, reply, reply)).run()
    assert "partway through its answer" in result.failures[0][1]
