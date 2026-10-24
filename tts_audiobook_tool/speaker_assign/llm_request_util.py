from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

import requests

from tts_audiobook_tool.conversation.llm_session import OUTPUT_LIMIT_FINISH_REASONS, LlmSession
from tts_audiobook_tool.util import ellipsize, make_error_string

if TYPE_CHECKING:
    from tts_audiobook_tool.prefs import Prefs


# Connect timeout, and the read timeout between streamed chunks. A full-book
# request can take minutes (including time to the first byte), so the read
# timeout is much longer than voice chat's.
SPEAKER_ASSIGN_LLM_TIMEOUT = (15.0, 900.0)

# Output-token limit for the Anthropic API (which requires one) when the
# prefs' extra params don't set max_tokens. Generous, since reasoning counts
# toward it and the roster reply for a whole book can be long.
SPEAKER_ASSIGN_ANTHROPIC_MAX_TOKENS = 64_000

# How long to wait for the request thread to finish after closing a streaming
# response (it's a daemon thread; see send_cancellable).
CANCEL_GRACE_SECONDS = 1.0

HTTP_ERROR_BODY_MAX_CHARS = 600


class CancelEvent(Enum):
    """What send_cancellable did about a cancel request (see its docstring)."""
    PENDING = "pending"          # deferred until the response starts streaming
    CANCELLING = "cancelling"    # closing the streaming response now
    ABANDONING = "abandoning"    # second request before streaming: giving up on it now


@dataclass(frozen=True)
class LlmRequestProgress:
    """Characters received so far, seconds since the request started, and
    whether a cancel request is waiting for the response to start."""
    answer_chars: int
    reasoning_chars: int
    elapsed: float
    cancel_pending: bool = False


@dataclass(frozen=True)
class LlmRequestResult:
    """
    The outcome of send_cancellable. When `cancelled`, `reply` is whatever had
    arrived (a partial reply) and should be discarded; `abandoned` means the
    request was given up before its response started, so the provider may
    still process it. `error` is the exception raised by the request, if any
    (not set when cancelled). `finish_reason` is why the provider ended the
    reply, if reported (see LlmSession.last_finish_reason).
    """
    reply: str
    reasoning: str
    elapsed: float
    cancelled: bool = False
    abandoned: bool = False
    error: Exception | None = None
    finish_reason: str | None = None

    @property
    def hit_output_limit(self) -> bool:
        """The reply was cut off by the output-token limit (with no answer
        at all if the model was still thinking)."""
        return self.finish_reason in OUTPUT_LIMIT_FINISH_REASONS


class LlmRequestUtil:
    """
    LLM requests for speaker assignment: sessions configured from the app's
    LLM prefs, and a request that can be cancelled (eg on control-c) while the
    calling thread stays responsive (see send_cancellable).
    """

    @staticmethod
    def make_session(prefs: Prefs, system_prompt: str) -> LlmSession:
        """
        A session configured from the app prefs (url, key, model, extra params),
        the same way voice chat builds its session; reasoning, effort etc are
        therefore controlled by the prefs' extra params. Only the transport
        timeout and the Anthropic default output limit differ (see
        SPEAKER_ASSIGN_LLM_TIMEOUT, SPEAKER_ASSIGN_ANTHROPIC_MAX_TOKENS).
        Raises ValueError for an unusable configuration.
        """
        return LlmSession(
            api_endpoint_url=prefs.llm_url,
            token=prefs.llm_api_key,
            model=prefs.llm_model,
            system_prompt=system_prompt,
            extra_params=prefs.llm_extra_params,
            timeout=SPEAKER_ASSIGN_LLM_TIMEOUT,
            anthropic_default_max_tokens=SPEAKER_ASSIGN_ANTHROPIC_MAX_TOKENS,
        )

    @staticmethod
    def describe_llm(prefs: Prefs) -> dict[str, Any]:
        """The LLM settings, as recorded in saved results."""
        return {"url": prefs.llm_url, "model": prefs.llm_model, "extra_params": prefs.llm_extra_params}

    @staticmethod
    def send_cancellable(
            llm: LlmSession,
            message: str,
            cancel_requests: Callable[[], int],
            on_poll: Callable[[LlmRequestProgress], None] | None = None,
            on_cancel_event: Callable[[CancelEvent], None] | None = None,
            poll_interval: float = 0.1,
            defer_until_streaming: bool = True,
    ) -> LlmRequestResult:
        """
        Sends `message` (streaming) on a background thread while the calling
        thread polls every `poll_interval` seconds: `cancel_requests` (the
        number of cancel requests so far, eg control-c presses) and `on_poll`
        (progress). Never raises for request errors (see LlmRequestResult.error).

        Closing the response stops the provider only once it is streaming
        (reasoning or answer chunks arriving): before that, the connection
        can't be closed until the server sends its next byte (response headers
        or the first token), so cancelling then wouldn't stop anything sooner.
        So a cancel request:
        - while streaming: closes the response right away (CANCELLING);
        - before streaming: is deferred (PENDING) and carried out as soon as
          the first chunk arrives (CANCELLING), stopping the provider then;
        - a second one before streaming: gives up on the request at once
          (ABANDONING). The request thread (a daemon) is left to close the
          connection when the server's next byte arrives.
        With `defer_until_streaming` False, the first request before streaming
        already abandons (for callers whose own first cancel request means
        something else, eg "stop after this request").
        The close itself runs on a helper thread, since it can block until the
        server's next byte; the calling thread never waits on it.
        """
        answer_chars = 0
        reasoning_parts: list[str] = []
        reasoning_chars = 0
        outcome: dict[str, Any] = {}

        def on_chunk(chunk: str) -> None:
            nonlocal answer_chars
            answer_chars += len(chunk)

        def on_reasoning(chunk: str) -> None:
            nonlocal reasoning_chars
            reasoning_parts.append(chunk)
            reasoning_chars += len(chunk)

        def run() -> None:
            try:
                outcome["reply"] = llm.send(message, on_chunk=on_chunk, on_reasoning=on_reasoning)
                outcome["finish_reason"] = getattr(llm, "last_finish_reason", None)
            except Exception as e:
                outcome["error"] = e

        def notify(event: CancelEvent) -> None:
            if on_cancel_event is not None:
                on_cancel_event(event)

        def close(wait: bool) -> None:
            threading.Thread(target=llm.cancel_active_request, name="speaker-assign-llm-cancel", daemon=True).start()
            if wait:  # streaming: the request thread finishes almost immediately
                thread.join(CANCEL_GRACE_SECONDS)

        started = time.monotonic()
        thread = threading.Thread(target=run, name="speaker-assign-llm", daemon=True)
        thread.start()

        handled_requests = 0
        pending = False
        cancelled = False
        abandoned = False
        while thread.is_alive():
            thread.join(poll_interval)
            streaming = answer_chars > 0 or reasoning_chars > 0
            requests_so_far = cancel_requests()
            is_new_request = requests_so_far > handled_requests
            handled_requests = requests_so_far

            if (is_new_request or pending) and streaming:
                notify(CancelEvent.CANCELLING)
                cancelled = True
                close(wait=True)
                break
            if is_new_request and (pending or requests_so_far >= 2 or not defer_until_streaming):
                notify(CancelEvent.ABANDONING)
                cancelled = abandoned = True
                close(wait=False)
                break
            if is_new_request:
                notify(CancelEvent.PENDING)
                pending = True

            if on_poll is not None:
                on_poll(LlmRequestProgress(
                    answer_chars, reasoning_chars, time.monotonic() - started, cancel_pending=pending
                ))

        # A cancel request that coincided with completion still counts
        cancelled = cancelled or pending or cancel_requests() > handled_requests

        error = outcome.get("error")
        return LlmRequestResult(
            reply=outcome.get("reply", "") or "",
            reasoning="".join(reasoning_parts),
            elapsed=time.monotonic() - started,
            cancelled=cancelled,
            abandoned=abandoned,
            error=None if cancelled else error,
            finish_reason=outcome.get("finish_reason"),
        )

    @staticmethod
    def describe_error(error: Exception, prefs: Prefs) -> str:
        """User-facing description of a failed LLM request, with a hint where
        the cause is recognizable."""
        if isinstance(error, requests.HTTPError):
            response = error.response
            status = response.status_code if response is not None else None
            body = ""
            if response is not None:
                try:
                    body = response.text.strip()
                except Exception:
                    pass
            message = f"LLM request failed: HTTP {status if status is not None else '?'}"
            if body:
                message += f"\n{ellipsize(body, HTTP_ERROR_BODY_MAX_CHARS)}"
            hint = LlmRequestUtil._http_error_hint(status, body)
            if hint:
                message += f"\n\n{hint}"
            return message
        if isinstance(error, requests.Timeout):
            return (
                f"LLM request timed out: {make_error_string(error)}\n\n"
                f"The server didn't respond or stopped sending data."
            )
        if isinstance(error, requests.ConnectionError):
            return (
                f"Couldn't connect to the LLM endpoint ({prefs.llm_url}): {make_error_string(error)}"
            )
        return f"LLM request failed: {make_error_string(error)}"

    @staticmethod
    def describe_output_limit(result: LlmRequestResult) -> str:
        """User-facing description of a reply cut off by the output-token limit."""
        when = "while still thinking, before answering" if not result.reply.strip() else "partway through its answer"
        return (
            f"The LLM hit its output-token limit {when} (finish reason: {result.finish_reason}).\n"
            f"Reasoning counts toward the limit. A lower reasoning effort, or a higher max tokens\n"
            f"value if the provider allows it, may help (Options > LLM settings > Extra params)."
        )

    @staticmethod
    def is_retryable(error: Exception) -> bool:
        """Whether a failed request may succeed when repeated: timeouts,
        dropped connections, rate limiting and server errors, but not other
        client errors (bad key, URL or model; request too large), which would
        just fail again."""
        if isinstance(error, requests.HTTPError):
            status = error.response.status_code if error.response is not None else None
            return status is None or status in (408, 409, 425, 429) or status >= 500
        return isinstance(error, requests.RequestException)

    @staticmethod
    def _http_error_hint(status: int | None, body: str) -> str:
        lower = body.lower()
        # Checked before the context hint: these bodies also mention "tokens"
        if status == 400 and "max_tokens" in lower:
            if "unsupported parameter" in lower:
                # eg OpenAI's reasoning models, which want max_completion_tokens
                return ("This model doesn't accept max_tokens; the error may name the parameter to use instead\n"
                        "(Options > LLM settings > Extra params).")
            # eg Anthropic: "max_tokens: 64000 > 32000, which is the maximum allowed number of output tokens..."
            return ("The output-token limit (max_tokens) may be too high for this model. Set a lower one in\n"
                    'Options > LLM settings > Extra params, eg {"max_tokens": 32000}.')
        if status in (400, 413) and any(
            word in lower for word in ("context", "token", "too long", "too large", "length")
        ):
            return "The text may be too long for the model's context window."
        if status in (401, 403):
            return "Check the API key environment variable (Options > LLM settings)."
        if status == 404:
            return "Check the endpoint URL and model name (Options > LLM settings)."
        if status == 429:
            return "The provider is rate-limiting requests, or the account is out of credits."
        return ""
