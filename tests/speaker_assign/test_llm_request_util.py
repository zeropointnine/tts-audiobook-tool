import threading
import time
from types import SimpleNamespace

import requests

from tts_audiobook_tool.speaker_assign.llm_request_util import CancelEvent, LlmRequestUtil


class FakeLlm:
    """
    Stands in for LlmSession. Stays silent until `streaming_starts` is set (like
    a provider queueing or prefilling), then streams reasoning chunks, then the
    answer. As with a real HTTP stream, a cancel during the silent phase can't
    take effect (cancel_active_request blocks) until the first chunk arrives.
    """

    def __init__(self, reasoning=("r",) * 3, answer=("{", "}"), error=None, streaming_starts=None):
        self.reasoning = reasoning
        self.answer = answer
        self.error = error
        self.streaming_starts = streaming_starts or threading.Event()
        if streaming_starts is None:
            self.streaming_starts.set()
        self.cancel_requested = threading.Event()
        self.chunks_sent = 0
        self.chunks_sent_at_cancel: int | None = None

    def send(self, _message, on_chunk, on_reasoning):
        if not self.streaming_starts.wait(3):
            return ""
        if self.error is not None:
            raise self.error
        reply = ""
        for callback, chunk in [(on_reasoning, c) for c in self.reasoning] + [(on_chunk, c) for c in self.answer]:
            self.chunks_sent += 1
            callback(chunk)
            if callback is on_chunk:
                reply += chunk
            time.sleep(0.03)
            if self.cancel_requested.is_set():
                return reply
        return reply

    def cancel_active_request(self):
        self.chunks_sent_at_cancel = self.chunks_sent
        self.cancel_requested.set()
        self.streaming_starts.wait(3)


def send(llm, presses, **kwargs):
    """send_cancellable with control-c presses read from presses["n"], and the
    cancel events it reported."""
    events = []
    result = LlmRequestUtil.send_cancellable(
        llm, "msg", cancel_requests=lambda: presses["n"], on_cancel_event=events.append,
        poll_interval=0.01, **kwargs)
    return result, events


def test_send_returns_reply_and_reasoning() -> None:
    result, events = send(FakeLlm(reasoning=["hm", "m"]), {"n": 0})
    assert (result.reply, result.reasoning, result.cancelled, result.error) == ("{}", "hmm", False, None)
    assert events == []


def test_request_errors_are_returned_not_raised() -> None:
    error = requests.ConnectionError("refused")
    result, _ = send(FakeLlm(error=error), {"n": 0})
    assert result.error is error
    assert not result.cancelled


def test_cancel_while_streaming_closes_right_away() -> None:
    llm = FakeLlm(reasoning=("r",) * 50)
    presses = {"n": 0}

    def on_poll(progress):
        if progress.reasoning_chars:
            presses["n"] = 1

    result, events = send(llm, presses, on_poll=on_poll)
    assert events == [CancelEvent.CANCELLING]
    assert result.cancelled and not result.abandoned
    assert llm.cancel_requested.is_set()
    assert llm.chunks_sent < 10


def test_cancel_before_streaming_is_deferred_until_the_first_chunk() -> None:
    # A press during the silent phase only marks the cancel as pending (the
    # connection can't be closed then anyway); once streaming starts, the
    # request is closed right away, so the provider stops generating.
    streaming_starts = threading.Event()
    llm = FakeLlm(reasoning=("r",) * 50, streaming_starts=streaming_starts)
    presses = {"n": 1}
    polls = []

    def on_poll(progress):
        polls.append(progress)
        if len(polls) == 5:
            streaming_starts.set()

    result, events = send(llm, presses, on_poll=on_poll)
    assert events == [CancelEvent.PENDING, CancelEvent.CANCELLING]
    assert all(p.cancel_pending for p in polls)
    assert llm.chunks_sent_at_cancel is not None and llm.chunks_sent_at_cancel >= 1
    assert llm.chunks_sent < 10
    assert result.cancelled and not result.abandoned


def test_second_press_before_streaming_abandons_without_blocking() -> None:
    # The provider stays silent and closing the connection would block until
    # its next byte; the caller still returns promptly.
    llm = FakeLlm(streaming_starts=threading.Event())
    presses = {"n": 1}
    polls = []

    def on_poll(progress):
        polls.append(progress)
        if len(polls) == 3:
            presses["n"] = 2

    started = time.monotonic()
    result, events = send(llm, presses, on_poll=on_poll)
    assert events == [CancelEvent.PENDING, CancelEvent.ABANDONING]
    assert result.cancelled and result.abandoned
    assert result.error is None
    assert time.monotonic() - started < 2


def make_http_error(status: int, body: str) -> requests.HTTPError:
    response = requests.Response()
    response.status_code = status
    response._content = body.encode()
    return requests.HTTPError(f"{status} error", response=response)


def test_describe_http_error_includes_body_and_hint() -> None:
    # A provider's context-length rejection: the body is shown along with a
    # hint that the book may not fit the model's context window.
    prefs = SimpleNamespace(llm_url="https://example.test/v1")
    text = LlmRequestUtil.describe_error(
        make_http_error(400, '{"error": "maximum context length is 131072 tokens"}'), prefs)  # type: ignore[arg-type]
    assert "HTTP 400" in text
    assert "maximum context length" in text
    assert "context window" in text

    text = LlmRequestUtil.describe_error(make_http_error(401, "bad key"), prefs)  # type: ignore[arg-type]
    assert "API key" in text


def test_without_deferral_a_cancel_before_streaming_abandons_at_once() -> None:
    # Step 2 uses its first control-c for "stop after this chunk", so the
    # next press must stop the request immediately, even before streaming.
    llm = FakeLlm(streaming_starts=threading.Event())
    presses = {"n": 0}
    polls = []

    def on_poll(progress):
        polls.append(progress)
        if len(polls) == 3:
            presses["n"] = 1

    started = time.monotonic()
    result, events = send(llm, presses, on_poll=on_poll, defer_until_streaming=False)
    assert events == [CancelEvent.ABANDONING]
    assert result.cancelled and result.abandoned
    assert time.monotonic() - started < 2


def test_is_retryable_separates_transient_from_permanent_errors() -> None:
    assert LlmRequestUtil.is_retryable(requests.ConnectionError("reset"))
    assert LlmRequestUtil.is_retryable(requests.Timeout("slow"))
    for status in (429, 500, 503):
        assert LlmRequestUtil.is_retryable(make_http_error(status, ""))
    for status in (400, 401, 403, 404):
        assert not LlmRequestUtil.is_retryable(make_http_error(status, ""))
    assert not LlmRequestUtil.is_retryable(ValueError("bug"))


def test_result_carries_the_sessions_finish_reason() -> None:
    llm = FakeLlm()
    llm.last_finish_reason = "length"
    result, _ = send(llm, {"n": 0})
    assert result.finish_reason == "length" and result.hit_output_limit
