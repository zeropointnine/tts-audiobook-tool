from __future__ import annotations

import json
import threading
import time
from typing import Callable

import requests

from tts_audiobook_tool.l import L


class LlmSession:
    """
    Manages a multi-turn LLM conversation and makes the HTTP API calls.

    Compatible with OpenAI chat completions and Anthropic messages APIs.
    The API format is auto-detected from the endpoint URL (override with api_type).

    Thread safety: send() serializes complete request/history transactions.
    cancel_active_request() may be called concurrently to unblock the active stream.

    The system prompt is immutable session configuration. It is captured when
    the session is constructed and remains stable for the lifetime of its chat
    history.
    """

    def __init__(
        self,
        api_endpoint_url: str,
        token: str,
        model: str,
        system_prompt: str = "",
        extra_params: dict | None = None,
        max_tokens: int | None = None,
        temperature: float = 0.7,
        api_type: str | None = None,
        timeout: float | tuple[float, float] = (15.0, 120.0),
        verbose: bool = False,
    ) -> None:
        """
        Args:
            api_endpoint_url: Full URL to the chat completions / messages endpoint.
            token: API key / bearer token.
            model: Model identifier string.
            system_prompt: System message injected on every request (not stored in history).
            temperature: Sampling temperature. Overridden to 1.0 by Anthropic when thinking is enabled.
            max_tokens: Maximum tokens in the response; is optional for OpenAI but required for Anthropic.
            extra_params: Provider-specific params merged into the request payload as-is (e.g.
                {"reasoning_effort": "high"} for OpenAI, {"thinking": {"type": "enabled"}} for Anthropic).
            api_type: Force "openai" or "anthropic" format. None = auto-detect from URL.
            timeout: HTTP request timeout. A single float sets both the connection and read
                timeout in seconds. A (connect, read) tuple sets them independently
                (default: (15.0, 120.0) — a fast connection failure with a lenient read window).
        """
        self.api_endpoint_url = api_endpoint_url
        self.token = token
        self.model = model
        self._system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = extra_params or {}
        self.timeout = timeout
        self.verbose = verbose
        self.history: list[dict[str, str]] = []
        self.history_lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._active_response_lock = threading.Lock()
        self._active_response: requests.Response | None = None
        self._cancel_requested = threading.Event()
        self.resolved_api_type = (
            api_type if api_type is not None else self.detect_api_type()
        )
        if self.resolved_api_type == "anthropic" and max_tokens is None:
            raise ValueError("max_tokens is required for Anthropic API")

    @property
    def system_prompt(self) -> str:
        """The system prompt captured for this chat session and its history."""
        return self._system_prompt

    def vlog(self, label: str, data: object, extra_line: str = "") -> None:
        if not self.verbose:
            return
        sep = "─" * 60
        print(f"\n{sep}\n[LlmUtil {label}]")
        if isinstance(data, (dict, list)):
            print(json.dumps(data, indent=2, ensure_ascii=False))
        else:
            print(data)
        if extra_line:
            print(extra_line)
        print(sep)

    def detect_api_type(self) -> str:
        if "anthropic.com" in self.api_endpoint_url:
            return "anthropic"
        return "openai"

    def send(
        self,
        message: str,
        on_chunk: Callable[[str], None] | None = None,
        interrupt_event: threading.Event | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> str:
        """Serialize one history transaction and provider request."""
        with self._send_lock:
            return self._send(message, on_chunk, interrupt_event, on_reasoning)

    def _send(
        self,
        message: str,
        on_chunk: Callable[[str], None] | None = None,
        interrupt_event: threading.Event | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> str:
        """Send a user message and return the assistant reply.

        Blocks until the full reply is received. If on_chunk is provided, streams
        the response and calls on_chunk(text) for each incremental chunk; the full
        concatenated string is still returned at the end.

        If interrupt_event is set mid-stream, streaming stops after the next chunk
        and the partial reply is returned. If no content was received before the
        interrupt, the user message is rolled back from history. An empty reply
        with no interrupt (provider returned nothing) is rolled back the same way.

        On HTTP or timeout errors, raises requests.HTTPError or requests.exceptions.Timeout.
        The user message is removed from history if the call fails.
        """
        if interrupt_event is not None and interrupt_event.is_set():
            L.i("[llm] send skipped: interrupt already set")
            return ""
        self._cancel_requested.clear()
        with self.history_lock:
            self.history.append({"role": "user", "content": message})
            history_snapshot = list(self.history)

        send_t0 = time.monotonic()
        L.i(
            f"[llm] request start: api={self.resolved_api_type} model={self.model}"
            f" url={self.api_endpoint_url} history={len(history_snapshot)}"
            f" streaming={on_chunk is not None}"
        )
        try:
            if self.resolved_api_type == "anthropic":
                payload = self.build_anthropic_payload(history_snapshot)
                reply = self.call_anthropic(payload, on_chunk, interrupt_event, on_reasoning)
            else:
                payload = self.build_openai_payload(history_snapshot)
                reply = self.call_openai(payload, on_chunk, interrupt_event, on_reasoning)
        except Exception as e:
            L.e(f"[llm] request failed after {time.monotonic() - send_t0:.1f}s: {type(e).__name__}: {e}")
            with self.history_lock:
                if self.history and self.history[-1] == {
                    "role": "user",
                    "content": message,
                }:
                    self.history.pop()
            raise

        if reply:
            L.i(f"[llm] request complete in {time.monotonic() - send_t0:.1f}s: {len(reply)} chars")
            with self.history_lock:
                self.history.append({"role": "assistant", "content": reply})
        else:
            if (interrupt_event is not None and interrupt_event.is_set()) or self._cancel_requested.is_set():
                L.i("[llm] interrupted with no content; rolling back user message")
            else:
                L.w("[llm] empty reply from provider; rolling back user message")
            # No assistant content arrived (interrupted before any content,
            # or the provider returned nothing) — roll back the user message
            # so history stays consistent. An appended empty assistant
            # message would be rejected by the Anthropic API on the next
            # request, failing that turn with a confusing provider error.
            with self.history_lock:
                if self.history and self.history[-1] == {
                    "role": "user",
                    "content": message,
                }:
                    self.history.pop()

        return reply

    def cancel_active_request(self) -> None:
        """Close the active HTTP stream so a blocked response worker unblocks."""
        L.i("[llm] cancel_active_request: closing active HTTP stream")
        self._cancel_requested.set()
        with self._active_response_lock:
            response = self._active_response
        if response is not None:
            try:
                response.close()
            except Exception:
                pass

    def _register_active_response(self, response: requests.Response) -> None:
        with self._active_response_lock:
            self._active_response = response
            should_close = self._cancel_requested.is_set()
        if should_close:
            response.close()

    def _clear_active_response(self, response: requests.Response) -> None:
        with self._active_response_lock:
            if self._active_response is response:
                self._active_response = None

    def clear(self) -> None:
        self.cancel_active_request()
        with self.history_lock:
            self.history = []

    def build_openai_payload(self, history_snapshot: list[dict]) -> dict:
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(history_snapshot)

        payload: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            **self.extra_params,
        }
        if self.max_tokens is not None:
            payload["max_tokens"] = self.max_tokens
        return payload

    def build_anthropic_payload(self, history_snapshot: list[dict]) -> dict:
        for i in range(len(history_snapshot) - 1):
            if history_snapshot[i]["role"] == history_snapshot[i + 1]["role"]:
                raise ValueError(
                    f"Anthropic requires alternating roles but found consecutive "
                    f"'{history_snapshot[i]['role']}' messages at positions {i} and {i + 1}. "
                    "Conversation history must alternate user and assistant turns."
                )

        payload: dict = {
            "model": self.model,
            "messages": history_snapshot,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **self.extra_params,
        }
        if self.system_prompt:
            payload["system"] = self.system_prompt

        thinking = self.extra_params.get("thinking")
        if isinstance(thinking, dict) and thinking.get("type") == "enabled":
            payload["temperature"] = 1.0

        return payload

    def call_openai(
        self,
        payload: dict,
        on_chunk: Callable[[str], None] | None,
        interrupt_event: threading.Event | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> str:
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }
        self.vlog("REQUEST (openai)", payload)
        if on_chunk is not None:
            import time

            payload = {**payload, "stream": True}
            t0 = time.monotonic()
            response = requests.post(
                self.api_endpoint_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            L.i(f"[llm] openai: headers ok in {time.monotonic() - t0:.1f}s, reading stream")
            full_text = ""
            reasoning_text = ""
            t_first: float | None = None
            first_line_logged = False
            lines_seen = 0
            heartbeat_at = time.monotonic()
            self._register_active_response(response)
            try:
                # chunk_size=1 avoids urllib3 read-ahead buffering so SSE lines
                # are delivered as soon as the server emits them.
                for line in response.iter_lines(chunk_size=1):
                    if not line:
                        continue
                    if not first_line_logged:
                        first_line_logged = True
                        L.i(f"[llm] openai: first stream line in {time.monotonic() - t0:.1f}s")
                    lines_seen += 1
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data: "):
                        L.i(f"[llm] openai: non-data stream line: {line_str[:80]!r}")
                        continue
                    data_str = line_str[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_data = json.loads(data_str)
                    except json.JSONDecodeError:
                        L.w(
                            f"[llm] openai: unparseable stream line: {data_str[:120]!r}"
                        )
                        continue
                    choices = chunk_data.get("choices") or []
                    if not choices:
                        # Usage chunks (e.g. stream_options.include_usage)
                        # and some providers' keep-alive/error events arrive
                        # with an empty choices list.
                        continue
                    delta = choices[0].get("delta") or {}
                    reasoning = delta.get("reasoning_content") or ""
                    if reasoning:
                        if not reasoning_text:
                            L.i(f"[llm] openai: reasoning tokens arriving (model is thinking before content)")
                        reasoning_text += reasoning
                        if on_reasoning is not None:
                            on_reasoning(reasoning)
                    text = delta.get("content") or ""
                    if text:
                        if t_first is None:
                            t_first = time.monotonic() - t0
                            L.i(f"[llm] openai: first content token in {t_first:.1f}s")
                        full_text += text
                        on_chunk(text)
                    now = time.monotonic()
                    if t_first is None and now - heartbeat_at > 10.0:
                        L.w(
                            f"[llm] openai: no content yet after {now - t0:.0f}s"
                            f" (lines={lines_seen}, reasoning chars={len(reasoning_text)})"
                        )
                        heartbeat_at = now
                    if (
                        interrupt_event is not None and interrupt_event.is_set()
                    ) or self._cancel_requested.is_set():
                        break
            except Exception:
                if not self._cancel_requested.is_set():
                    raise
            finally:
                self._clear_active_response(response)
                response.close()
            if self.verbose:
                t_total = time.monotonic() - t0
                timing_info = (
                    f"first chunk: {t_first:.2f}s  " if t_first is not None else ""
                )
                timing_info += f"total: {t_total:.2f}s"
                self.vlog(
                    "RESPONSE (openai streaming)",
                    {
                        "reasoning_content": reasoning_text or None,
                        "content": full_text,
                    },
                    extra_line=timing_info,
                )
            return full_text
        else:
            response = requests.post(
                self.api_endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.vlog("RESPONSE (openai)", data)
            return data["choices"][0]["message"]["content"]

    def call_anthropic(
        self,
        payload: dict,
        on_chunk: Callable[[str], None] | None,
        interrupt_event: threading.Event | None = None,
        on_reasoning: Callable[[str], None] | None = None,
    ) -> str:
        headers = {
            "x-api-key": self.token,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        self.vlog("REQUEST (anthropic)", payload)
        if on_chunk is not None:
            import time

            payload = {**payload, "stream": True}
            t0 = time.monotonic()
            response = requests.post(
                self.api_endpoint_url,
                headers=headers,
                json=payload,
                stream=True,
                timeout=self.timeout,
            )
            response.raise_for_status()
            L.i(f"[llm] anthropic: headers ok in {time.monotonic() - t0:.1f}s, reading stream")
            full_text = ""
            thinking_text = ""
            t_first: float | None = None
            first_line_logged = False
            self._register_active_response(response)
            try:
                # chunk_size=1 avoids urllib3 read-ahead buffering so SSE lines
                # are delivered as soon as the server emits them.
                for line in response.iter_lines(chunk_size=1):
                    if not line:
                        continue
                    if not first_line_logged:
                        first_line_logged = True
                        L.i(f"[llm] anthropic: first stream line in {time.monotonic() - t0:.1f}s")
                    line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                    if not line_str.startswith("data: "):
                        continue
                    event_data = json.loads(line_str[6:])
                    if event_data.get("type") == "content_block_delta":
                        delta = event_data.get("delta", {})
                        if delta.get("type") == "thinking_delta":
                            if not thinking_text:
                                L.i("[llm] anthropic: thinking tokens arriving before content")
                            thinking_text += delta.get("thinking", "")
                            if on_reasoning is not None:
                                on_reasoning(delta.get("thinking", ""))
                        elif delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            if text:
                                if t_first is None:
                                    # When thinking is enabled, this fires after the thinking
                                    # phase finishes, so it reflects thinking time too.
                                    t_first = time.monotonic() - t0
                                full_text += text
                                on_chunk(text)
                    if (
                        interrupt_event is not None and interrupt_event.is_set()
                    ) or self._cancel_requested.is_set():
                        break
            except Exception:
                if not self._cancel_requested.is_set():
                    raise
            finally:
                self._clear_active_response(response)
                response.close()
            if self.verbose:
                t_total = time.monotonic() - t0
                self.vlog(
                    "RESPONSE (anthropic streaming)",
                    {
                        "thinking": thinking_text or None,
                        "content": full_text,
                    },
                )
                first_str = (
                    f"first chunk: {t_first:.2f}s  " if t_first is not None else ""
                )
                print(f"{first_str}total: {t_total:.2f}s")
            return full_text
        else:
            response = requests.post(
                self.api_endpoint_url,
                headers=headers,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
            self.vlog("RESPONSE (anthropic)", data)
            return self.extract_anthropic_text(data["content"])

    def extract_anthropic_text(self, content: list[dict]) -> str:
        return "".join(
            block.get("text", "") for block in content if block.get("type") == "text"
        )
