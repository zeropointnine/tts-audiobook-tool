import threading
from unittest.mock import MagicMock

import pytest

from tts_audiobook_tool.conversation.llm_session import LlmSession


def make_response(json_data=None, lines=None, status_error=None) -> MagicMock:
    """A fake requests.Response for non-streaming or SSE-streaming calls."""
    response = MagicMock()
    if status_error is not None:
        response.raise_for_status.side_effect = status_error
    else:
        response.raise_for_status.return_value = None
    if lines is not None:
        response.iter_lines.return_value = iter(lines)
    else:
        response.json.return_value = json_data
    return response


def sse(*data_objects: str, done: bool = False) -> list[str]:
    lines = [f"data: {obj}" for obj in data_objects]
    if done:
        lines.append("data: [DONE]")
    return lines


def make_openai_llm(**kwargs) -> LlmSession:
    params = {
        "api_endpoint_url": "https://api.openai.com/v1/chat/completions",
        "token": "test-token",
        "model": "test-model",
    }
    params.update(kwargs)
    return LlmSession(**params)


def make_anthropic_llm(**kwargs) -> LlmSession:
    params = {
        "api_endpoint_url": "https://api.anthropic.com/v1/messages",
        "token": "test-token",
        "model": "test-model",
        "max_tokens": 100,
    }
    params.update(kwargs)
    return LlmSession(**params)


def test_detect_api_type_by_url() -> None:
    assert make_openai_llm().resolved_api_type == "openai"
    assert make_anthropic_llm().resolved_api_type == "anthropic"


def test_explicit_api_type_overrides_url_detection() -> None:
    llm = make_anthropic_llm(api_type="openai")
    assert llm.resolved_api_type == "openai"


def test_anthropic_requires_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        LlmSession(
            api_endpoint_url="https://api.anthropic.com/v1/messages",
            token="t",
            model="m",
            max_tokens=None,
        )


def test_build_openai_payload_includes_system_prompt_and_params() -> None:
    llm = make_openai_llm(system_prompt="sys", temperature=0.2, max_tokens=77,
                          extra_params={"reasoning_effort": "high"})

    payload = llm.build_openai_payload([{"role": "user", "content": "Hi"}])

    assert payload == {
        "model": "test-model",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
        ],
        "temperature": 0.2,
        "reasoning_effort": "high",
        "max_tokens": 77,
    }


def test_build_openai_payload_omits_system_and_max_tokens_when_unset() -> None:
    llm = make_openai_llm()

    payload = llm.build_openai_payload([{"role": "user", "content": "Hi"}])

    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert "max_tokens" not in payload


def test_build_anthropic_payload_puts_system_separately_and_merges_params() -> None:
    llm = make_anthropic_llm(system_prompt="sys", temperature=0.5,
                             extra_params={"top_k": 3})

    payload = llm.build_anthropic_payload([{"role": "user", "content": "Hi"}])

    assert payload["system"] == "sys"
    assert payload["messages"] == [{"role": "user", "content": "Hi"}]
    assert payload["temperature"] == 0.5
    assert payload["max_tokens"] == 100
    assert payload["top_k"] == 3


def test_build_anthropic_payload_forces_temperature_one_for_thinking() -> None:
    llm = make_anthropic_llm(temperature=0.5,
                             extra_params={"thinking": {"type": "enabled"}})

    payload = llm.build_anthropic_payload([{"role": "user", "content": "Hi"}])

    assert payload["temperature"] == 1.0


def test_build_anthropic_payload_rejects_consecutive_roles() -> None:
    llm = make_anthropic_llm()

    with pytest.raises(ValueError, match="alternating"):
        llm.build_anthropic_payload([
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ])


def test_send_openai_non_stream_records_history(monkeypatch) -> None:
    llm = make_openai_llm(system_prompt="sys")
    posted: list[dict] = []

    def fake_post(url, headers=None, json=None, **kwargs):
        posted.append({"url": url, "headers": headers, "json": json})
        return make_response({"choices": [{"message": {"content": "Hello back"}}]})

    monkeypatch.setattr("tts_audiobook_tool.conversation.llm_session.requests.post", fake_post)

    reply = llm.send("Hi")

    assert reply == "Hello back"
    assert llm.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello back"},
    ]
    assert posted[0]["url"] == "https://api.openai.com/v1/chat/completions"
    assert posted[0]["headers"]["Authorization"] == "Bearer test-token"
    assert posted[0]["json"]["messages"][0] == {"role": "system", "content": "sys"}


def test_send_openai_streaming_streams_content_and_reasoning_chunks(monkeypatch) -> None:
    llm = make_openai_llm()
    lines = sse(
        '{"choices": [{"delta": {"reasoning_content": "hmm"}}]}',
        '{"choices": [{"delta": {"content": "Hello"}}]}',
        '{"choices": [{"delta": {"content": " world"}}]}',
        done=True,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.llm_session.requests.post",
        lambda *args, **kwargs: make_response(lines=lines),
    )

    chunks: list[str] = []
    reply = llm.send("Hi", on_chunk=chunks.append)

    assert reply == "Hello world"
    assert chunks == ["Hello", " world"]
    assert llm.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello world"},
    ]


def test_send_openai_error_rolls_back_user_message(monkeypatch) -> None:
    llm = make_openai_llm()
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.llm_session.requests.post",
        lambda *args, **kwargs: make_response(status_error=RuntimeError("HTTP 500")),
    )

    with pytest.raises(RuntimeError):
        llm.send("Hi")

    assert llm.history == []


def test_send_interrupt_before_any_content_rolls_back_user_message(monkeypatch) -> None:
    llm = make_openai_llm()
    lines = sse('{"choices": [{"delta": {"reasoning_content": "hmm"}}]}', done=True)
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.llm_session.requests.post",
        lambda *args, **kwargs: make_response(lines=lines),
    )
    interrupt = threading.Event()
    interrupt.set()

    reply = llm.send("Hi", on_chunk=lambda _chunk: None, interrupt_event=interrupt)

    assert reply == ""
    assert llm.history == []


def test_send_interrupt_after_content_returns_partial_reply(monkeypatch) -> None:
    llm = make_openai_llm()
    lines = sse(
        '{"choices": [{"delta": {"content": "Hel"}}]}',
        '{"choices": [{"delta": {"content": "lo world"}}]}',
        done=True,
    )
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.llm_session.requests.post",
        lambda *args, **kwargs: make_response(lines=lines),
    )
    interrupt = threading.Event()

    def on_chunk(chunk: str) -> None:
        interrupt.set()  # interrupt after the first chunk

    reply = llm.send("Hi", on_chunk=on_chunk, interrupt_event=interrupt)

    assert reply == "Hel"
    assert llm.history == [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hel"},
    ]


def test_send_anthropic_non_stream_extracts_text_blocks(monkeypatch) -> None:
    llm = make_anthropic_llm()
    content = [
        {"type": "thinking", "thinking": "let me think"},
        {"type": "text", "text": "Hi "},
        {"type": "text", "text": "there"},
    ]
    posted: list[dict] = []

    def fake_post(url, headers=None, json=None, **kwargs):
        posted.append({"headers": headers, "json": json})
        return make_response({"content": content})

    monkeypatch.setattr("tts_audiobook_tool.conversation.llm_session.requests.post", fake_post)

    reply = llm.send("Hi")

    assert reply == "Hi there"
    assert posted[0]["headers"]["x-api-key"] == "test-token"
    assert posted[0]["headers"]["anthropic-version"] == "2023-06-01"


def test_send_anthropic_streaming_collects_text_deltas(monkeypatch) -> None:
    llm = make_anthropic_llm()
    lines = [
        "data: " + '{"type": "content_block_delta", "delta": {"type": "thinking_delta", "thinking": "hm"}}',
        "data: " + '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": "Hello"}}',
        "data: " + '{"type": "content_block_delta", "delta": {"type": "text_delta", "text": " world"}}',
    ]
    monkeypatch.setattr(
        "tts_audiobook_tool.conversation.llm_session.requests.post",
        lambda *args, **kwargs: make_response(lines=lines),
    )

    chunks: list[str] = []
    reply = llm.send("Hi", on_chunk=chunks.append)

    assert reply == "Hello world"
    assert chunks == ["Hello", " world"]


def test_extract_anthropic_text_skips_non_text_blocks() -> None:
    llm = make_anthropic_llm()

    text = llm.extract_anthropic_text([
        {"type": "thinking", "thinking": "nope"},
        {"type": "text", "text": "a"},
        {"type": "tool_use", "id": "x"},
        {"type": "text", "text": "b"},
    ])

    assert text == "ab"


def test_clear_empties_history() -> None:
    llm = make_openai_llm()
    llm.history = [{"role": "user", "content": "x"}]

    llm.clear()

    assert llm.history == []