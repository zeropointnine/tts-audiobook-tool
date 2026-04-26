from __future__ import annotations

import json
import threading
from typing import Callable

import requests


class LlmUtil:
    """
    Manages a multi-turn LLM conversation and makes the HTTP API calls.

    Compatible with OpenAI chat completions and Anthropic messages APIs.
    The API format is auto-detected from the endpoint URL (override with api_type).

    Thread safety: send() is safe to call from multiple threads, but concurrent
    calls on a shared instance interleave user messages in non-deterministic order.
    For strict turn ordering, serialize send() calls externally or use separate instances.
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
        timeout: float = 120.0,
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
            timeout: HTTP request timeout in seconds.
        """
        self.api_endpoint_url = api_endpoint_url
        self.token = token
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.extra_params = extra_params or {}
        self.timeout = timeout
        self.verbose = verbose
        self.history: list[dict[str, str]] = []
        self.history_lock = threading.Lock()
        self.resolved_api_type = api_type if api_type is not None else self.detect_api_type()
        if self.resolved_api_type == "anthropic" and max_tokens is None:
            raise ValueError("max_tokens is required for Anthropic API")

    def vlog(self, label: str, data: object, extra_line: str="") -> None:
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
    ) -> str:
        """Send a user message and return the assistant reply.

        Blocks until the full reply is received. If on_chunk is provided, streams
        the response and calls on_chunk(text) for each incremental chunk; the full
        concatenated string is still returned at the end.

        If interrupt_event is set mid-stream, streaming stops after the next chunk
        and the partial reply is returned. If no content was received before the
        interrupt, the user message is rolled back from history.

        On HTTP or timeout errors, raises requests.HTTPError or requests.exceptions.Timeout.
        The user message is removed from history if the call fails.
        """
        with self.history_lock:
            self.history.append({"role": "user", "content": message})
            history_snapshot = list(self.history)

        try:
            if self.resolved_api_type == "anthropic":
                payload = self.build_anthropic_payload(history_snapshot)
                reply = self.call_anthropic(payload, on_chunk, interrupt_event)
            else:
                payload = self.build_openai_payload(history_snapshot)
                reply = self.call_openai(payload, on_chunk, interrupt_event)
        except Exception:
            with self.history_lock:
                if self.history and self.history[-1] == {"role": "user", "content": message}:
                    self.history.pop()
            raise

        if reply:
            with self.history_lock:
                self.history.append({"role": "assistant", "content": reply})
        elif interrupt_event is not None and interrupt_event.is_set():
            # Interrupted before any content arrived — roll back user message so
            # history stays consistent (no orphaned user turn with no assistant reply).
            with self.history_lock:
                if self.history and self.history[-1] == {"role": "user", "content": message}:
                    self.history.pop()
        else:
            with self.history_lock:
                self.history.append({"role": "assistant", "content": reply})

        return reply

    def clear(self) -> None:
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
                    f"This can happen when concurrent send() calls interleave — serialize externally."
                )

        payload: dict = {
            "model": self.model,
            "messages": history_snapshot,
            "max_tokens": self.max_tokens,  # validated non-None at __init__ for anthropic
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
            full_text = ""
            reasoning_text = ""
            t_first: float | None = None
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                chunk_data = json.loads(data_str)
                delta = chunk_data["choices"][0]["delta"]
                reasoning = delta.get("reasoning_content") or ""
                if reasoning:
                    reasoning_text += reasoning
                text = delta.get("content") or ""
                if text:
                    if t_first is None:
                        t_first = time.monotonic() - t0
                    full_text += text
                    on_chunk(text)
                if interrupt_event is not None and interrupt_event.is_set():
                    break
            if self.verbose:
                t_total = time.monotonic() - t0
                timing_info = f"first chunk: {t_first:.2f}s  " if t_first is not None else ""
                timing_info += f"total: {t_total:.2f}s"
                self.vlog("RESPONSE (openai streaming)", {
                    "reasoning_content": reasoning_text or None,
                    "content": full_text,
                }, extra_line=timing_info)
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
            full_text = ""
            thinking_text = ""
            t_first: float | None = None
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if not line_str.startswith("data: "):
                    continue
                event_data = json.loads(line_str[6:])
                if event_data.get("type") == "content_block_delta":
                    delta = event_data.get("delta", {})
                    if delta.get("type") == "thinking_delta":
                        thinking_text += delta.get("thinking", "")
                    elif delta.get("type") == "text_delta":
                        text = delta.get("text", "")
                        if text:
                            if t_first is None:
                                # When thinking is enabled, this fires after the thinking
                                # phase finishes, so it reflects thinking time too.
                                t_first = time.monotonic() - t0
                            full_text += text
                            on_chunk(text)
                if interrupt_event is not None and interrupt_event.is_set():
                    break
            if self.verbose:
                t_total = time.monotonic() - t0
                self.vlog("RESPONSE (anthropic streaming)", {
                    "thinking": thinking_text or None,
                    "content": full_text,
                })
                first_str = f"first chunk: {t_first:.2f}s  " if t_first is not None else ""
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
            block.get("text", "")
            for block in content
            if block.get("type") == "text"
        )
