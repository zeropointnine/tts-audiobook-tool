import importlib.metadata
import os
from types import SimpleNamespace

import pytest

# The `moss-tts` distribution installs no importable module (the model itself
# runs via transformers remote code); its dist-info is the venv marker.
try:
    importlib.metadata.distribution("moss-tts")
except importlib.metadata.PackageNotFoundError:
    pytest.skip("moss-tts distribution not present in this environment", allow_module_level=True)

import torch

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.moss_model import MossModel


class FakeProcessor:
    # No `audio_tokenizer` attribute: prepare_audio_tokenizer must no-op
    model_config = SimpleNamespace(sampling_rate=24000)

    def __init__(self):
        self.encode_calls: list[str] = []
        self.build_user_calls: list[dict] = []
        self.decode_calls: list = []

    def encode_audios_from_path(self, paths: list) -> list:
        self.encode_calls.append(paths[0])
        return [torch.randint(0, 1000, (10, 4), dtype=torch.int64)]

    def build_user_message(self, text=None, language=None, reference=None):
        self.build_user_calls.append({"text": text, "language": language, "reference": reference})
        return object()

    def build_assistant_message(self, audio_codes_list=None, content=None):
        return object()

    def __call__(self, conversations, mode=None):
        return {
            "input_ids": torch.zeros(1, 8, dtype=torch.long),
            "attention_mask": torch.ones(1, 8, dtype=torch.long),
        }

    def decode(self, outputs):
        self.decode_calls.append(outputs)
        return [SimpleNamespace(audio_codes_list=[torch.zeros(100, dtype=torch.float32)])]


class FakeModel:
    def __init__(self):
        self.generate_calls = 0
        self.cpu_calls = 0

    def generate(self, **kwargs):
        self.generate_calls += 1
        return object()

    def eval(self):
        return self

    def cpu(self):
        self.cpu_calls += 1
        return self


def make_model() -> MossModel:
    model = MossModel.__new__(MossModel)
    model._device_type = DeviceType.CPU
    model.model_target = "fake"
    model.dtype = torch.float32
    model._voice_info = None
    model.cached_continuation_history = []
    model.audio_tokenizer_is_on_device = False
    model.processor = FakeProcessor()
    model.model = FakeModel()
    return model


def generate(model: MossModel, voice_path: str):
    return model.generate(
        prompts=["Test sentence."],
        voice_path=voice_path,
        rolling_continuation_max_segments=0,
        language="English",
        temperature=1.0,
        audio_top_p=0.9,
        audio_top_k=10,
        seed=1,
    )


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_moss_reuses_audio_codes_for_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert not isinstance(generate(model, str(voice_a)), str)
    assert not isinstance(generate(model, str(voice_b)), str)
    assert not isinstance(generate(model, str(voice_a)), str)

    assert model.processor.encode_calls == [str(voice_a), str(voice_b)]
    assert model.model.generate_calls == 3
    assert len(model._voice_clone_cache) == 2

    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    cached_a = model._voice_clone_cache[key_a]
    assert cached_a.device.type == "cpu"
    assert not cached_a.requires_grad

    # The message builder received the cached tensor directly (identity)
    last_user_msg = model.processor.build_user_calls[-1]
    assert last_user_msg["reference"] == [cached_a]
    assert last_user_msg["reference"][0] is cached_a


def test_moss_clears_continuation_on_voice_change(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    # First call establishes the active voice (clearing pre-seeded history)
    model.cached_continuation_history.append(("pre-seeded", torch.zeros(2, 3)))
    assert not isinstance(generate(model, str(voice_a)), str)
    assert model.cached_continuation_history == []

    # Same voice: history survives
    model.cached_continuation_history.append(("ctx", torch.zeros(2, 3)))
    assert not isinstance(generate(model, str(voice_a)), str)
    assert len(model.cached_continuation_history) == 1

    # Voice switch: history cleared, clone for B prepared
    assert not isinstance(generate(model, str(voice_b)), str)
    assert model.cached_continuation_history == []
    assert len(model._voice_clone_cache) == 2

    # Back to A: reused, history stays cleared
    assert not isinstance(generate(model, str(voice_a)), str)
    assert model.cached_continuation_history == []
    assert len(model.processor.encode_calls) == 2

    fake_model = model.model
    model.kill()
    assert model.cached_continuation_history == []
    assert model._voice_info is None
    assert model.processor is None
    assert model.model is None
    assert fake_model.cpu_calls == 1  # cpu() called during kill


def test_moss_rebuilds_changed_voice_and_clears_cache(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert not isinstance(generate(model, str(voice_path)), str)
    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path)), str)

    assert len(model.processor.encode_calls) == 2
    assert len(model._voice_clone_cache) == 1

    # A no-voice call resets the active voice (and any continuation) but
    # leaves the prepared-clone cache in place, like Higgs/Pocket: it is an
    # optimization keyed per source file, not a "current voice" slot
    assert not isinstance(generate(model, ""), str)
    assert model._voice_info is None
    assert len(model._voice_clone_cache) == 1
    assert model.processor.encode_calls == [str(voice_path), str(voice_path)]

    # Coming back to the same file reuses the retained clone
    assert not isinstance(generate(model, str(voice_path)), str)
    assert len(model.processor.encode_calls) == 2

    model.kill()
    assert getattr(model, "_voice_clone_cache", None) in (None, {})


def test_moss_error_string_on_encode_failure(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def boom(paths):
        raise ValueError("boom")

    model.processor.encode_audios_from_path = boom

    assert generate(model, str(voice_path)) == (
        f"Couldn't create voice clone for {os.path.abspath(str(voice_path))} - ValueError: boom"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.model.generate_calls == 0  # generation never ran


def test_moss_no_voice_bypasses_cache(tmp_path):
    model = make_model()

    assert not isinstance(generate(model, ""), str)

    assert getattr(model, "_voice_clone_cache", None) in (None, {})
    assert model.processor.encode_calls == []
    last_user_msg = model.processor.build_user_calls[0]
    assert last_user_msg["reference"] is None
