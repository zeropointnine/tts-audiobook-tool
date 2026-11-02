"""STT input is always an in-memory waveform, never a file path.

Both Whisper backends accept a path and decode it themselves; mlx-whisper does
so by launching the `ffmpeg` executable, which the app must not depend on.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.model_runtime import ModelRuntimeRole
from tts_audiobook_tool.stt import FasterWhisperAdapter, MlxWhisperAdapter, Stt

pytestmark = pytest.mark.no_ffmpeg_exe


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[object] = []

    def transcribe(self, audio, **_kwargs):
        self.calls.append(audio)
        return {"segments": []} if isinstance(self, RecordingMlx) else ([], None)


class RecordingMlx(RecordingBackend):
    pass


def make_adapters():
    faster = object.__new__(FasterWhisperAdapter)
    faster._model = RecordingBackend()  # type: ignore[attr-defined]
    mlx = object.__new__(MlxWhisperAdapter)
    mlx._mlx_whisper = RecordingMlx()  # type: ignore[attr-defined]
    mlx._model = "model"  # type: ignore[attr-defined]
    return [(faster, faster._model), (mlx, mlx._mlx_whisper)]  # type: ignore[attr-defined]


@pytest.mark.parametrize("audio", ["book.mp3", Path("book.mp3"), b"book.mp3", [0.0, 0.1]])
def test_adapters_reject_non_waveform_input_before_reaching_the_backend(audio) -> None:
    for adapter, backend in make_adapters():
        with pytest.raises(TypeError, match="waveform array"):
            adapter.transcribe(audio)
        assert backend.calls == []


def test_adapters_pass_waveforms_through() -> None:
    audio = np.zeros(1600, dtype=np.float32)
    for adapter, backend in make_adapters():
        adapter.transcribe(audio, word_timestamps=True)
        assert backend.calls == [audio]


def test_in_process_transcriber_hands_the_backend_an_array(monkeypatch) -> None:
    # The in-process route (worker roles) resamples to 16 kHz and transcribes
    # the samples; a stand-in backend records what it was given
    from tts_audiobook_tool import transcriber as transcriber_module

    backend = SimpleNamespace(supported_languages=["en"], calls=[])

    def transcribe(audio, **_kwargs):
        backend.calls.append(audio)
        return [], None

    backend.transcribe = transcribe
    monkeypatch.setattr(transcriber_module, "current_role", lambda: ModelRuntimeRole.MODEL_WORKER)
    monkeypatch.setattr(Stt, "get_whisper", staticmethod(lambda: backend))
    monkeypatch.setattr(Stt, "set_variant", staticmethod(lambda _value: None))
    monkeypatch.setattr(Stt, "set_config", staticmethod(lambda _value: None))

    sound = Sound(np.zeros(4800, dtype=np.float32), 48000)
    assert transcriber_module.Transcriber.transcribe_to_words(sound, "en") == []
    assert len(backend.calls) == 1
    assert isinstance(backend.calls[0], np.ndarray)
    assert len(backend.calls[0]) == 1600
