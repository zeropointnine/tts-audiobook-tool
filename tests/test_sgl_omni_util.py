from io import BytesIO
import base64
import json

import httpx
import numpy as np
import soundfile

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_support.sgl_omni_util import SglOmniUtil
from tts_audiobook_tool.l import L


def encode_wav(data: np.ndarray, sample_rate: int = 24000) -> bytes:
    out = BytesIO()
    soundfile.write(out, data, sample_rate, format="WAV")
    return out.getvalue()


def sse_line(event: dict) -> bytes:
    return b"data: " + json.dumps(event).encode("utf-8") + b"\n"


class ByteStream(httpx.SyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


def test_generate_streaming_decodes_sse_audio_chunks_and_callbacks():
    L.init("test_sgl_omni_util")
    chunk_1 = np.asarray([0.0, 0.25, -0.25], dtype=np.float32)
    chunk_2 = np.asarray([0.5, -0.5], dtype=np.float32)
    request_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content.decode("utf-8")))
        events = b"".join([
            sse_line({
                "object": "audio.speech.chunk",
                "index": 0,
                "audio": {"data": base64.b64encode(encode_wav(chunk_1)).decode("ascii")},
                "finish_reason": None,
            }),
            sse_line({
                "object": "audio.speech.chunk",
                "index": 1,
                "audio": {"data": base64.b64encode(encode_wav(chunk_2)).decode("ascii")},
                "finish_reason": None,
            }),
            sse_line({
                "object": "audio.speech.chunk",
                "index": 2,
                "audio": None,
                "finish_reason": "stop",
            }),
            b"data: [DONE]\n",
        ])
        return httpx.Response(200, content=events)

    old_transport = httpx.Client
    streamed_chunks = []
    end_calls = []

    def make_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return old_transport(*args, **kwargs)

    httpx.Client = make_client
    try:
        result = SglOmniUtil.generate_streaming(
            "http://example.test",
            {"input": "hello", "stream": False},
            on_stream_chunk=lambda data: streamed_chunks.append(data.copy()),
            on_stream_end=lambda: end_calls.append(True),
        )
    finally:
        httpx.Client = old_transport

    assert isinstance(result, Sound)
    assert request_payloads == [{"input": "hello", "stream": True}]
    assert len(streamed_chunks) == 2
    assert len(end_calls) == 1
    np.testing.assert_allclose(streamed_chunks[0], chunk_1, atol=1e-4)
    np.testing.assert_allclose(streamed_chunks[1], chunk_2, atol=1e-4)
    np.testing.assert_allclose(result.data, np.concatenate([chunk_1, chunk_2]), atol=1e-4)
    assert result.sr == 24000


def test_generate_streaming_decodes_raw_pcm_audio_chunks_and_callbacks():
    L.init("test_sgl_omni_util")
    chunk_1_i16 = np.asarray([0, 8192, -8192], dtype=np.int16)
    chunk_2_i16 = np.asarray([16384, -16384], dtype=np.int16)
    request_payloads = []

    def handler(request: httpx.Request) -> httpx.Response:
        request_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-sample-rate": "24000",
                "x-channels": "1",
                "x-bit-depth": "16",
            },
            stream=ByteStream([chunk_1_i16.tobytes() + chunk_2_i16.tobytes()]),
        )

    old_transport = httpx.Client
    streamed_chunks = []
    end_calls = []

    def make_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return old_transport(*args, **kwargs)

    httpx.Client = make_client
    try:
        result = SglOmniUtil.generate_streaming(
            "http://example.test",
            {"input": "hello", "stream": False, "response_format": "pcm"},
            on_stream_chunk=lambda data: streamed_chunks.append(data.copy()),
            on_stream_end=lambda: end_calls.append(True),
        )
    finally:
        httpx.Client = old_transport

    assert isinstance(result, Sound)
    assert request_payloads == [{"input": "hello", "stream": True, "response_format": "pcm"}]
    assert len(streamed_chunks) == 1
    assert len(end_calls) == 1
    expected = np.concatenate([chunk_1_i16, chunk_2_i16]).astype(np.float32) / 32768.0
    np.testing.assert_allclose(streamed_chunks[0], expected)
    np.testing.assert_allclose(result.data, expected)
    assert result.sr == 24000


def test_generate_streaming_invokes_pcm_callback_per_raw_stream_chunk():
    L.init("test_sgl_omni_util")
    chunk_1_i16 = np.asarray([0, 8192], dtype=np.int16)
    chunk_2_i16 = np.asarray([-8192, 16384], dtype=np.int16)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={
                "content-type": "audio/pcm",
                "x-sample-rate": "24000",
                "x-channels": "1",
                "x-bit-depth": "16",
            },
            stream=ByteStream([chunk_1_i16.tobytes(), chunk_2_i16.tobytes()]),
        )

    old_transport = httpx.Client
    streamed_chunks = []

    def make_client(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return old_transport(*args, **kwargs)

    httpx.Client = make_client
    try:
        result = SglOmniUtil.generate_streaming(
            "http://example.test",
            {"input": "hello", "stream": False, "response_format": "pcm"},
            on_stream_chunk=lambda data: streamed_chunks.append(data.copy()),
        )
    finally:
        httpx.Client = old_transport

    assert isinstance(result, Sound)
    assert len(streamed_chunks) == 2
    np.testing.assert_allclose(streamed_chunks[0], chunk_1_i16.astype(np.float32) / 32768.0)
    np.testing.assert_allclose(streamed_chunks[1], chunk_2_i16.astype(np.float32) / 32768.0)
