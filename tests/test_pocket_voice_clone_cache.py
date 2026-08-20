import os

import pytest

pytest.importorskip("pocket_tts")

import torch
from pocket_tts.utils.config import CONFIGS_DIR  # type: ignore
from pocket_tts.utils.utils import _ORIGINS_OF_PREDEFINED_VOICES  # type: ignore

from tts_audiobook_tool.app_types import DeviceType
from tts_audiobook_tool.tts_models.pocket_model import PocketModel


class FakePocketModel:
    def __init__(self):
        self.device = torch.device("cpu")
        self.origin = None  # as when loaded without a language config
        self.temp = 1.0
        self.lsd_decode_steps = 1
        self.get_state_calls: list[str] = []
        self.created_states: list[dict] = []

    def get_state_for_audio_prompt(self, voice_path: str) -> dict:
        self.get_state_calls.append(voice_path)
        state = {"module_a": {"w": torch.zeros(4)}}
        self.created_states.append(state)
        return state

    def generate_audio_stream(self, voice_state, text, max_tokens=None):
        yield torch.zeros(64, dtype=torch.float32)

    def to(self, device):
        return self


def make_model() -> PocketModel:
    model = PocketModel.__new__(PocketModel)
    model._device_type = DeviceType.CPU
    model.model = FakePocketModel()
    return model


def generate(model: PocketModel, voice_path: str):
    return model.generate(["Test sentence."], voice_path, 1.0, 1)


def cache_key_prefix(path: str) -> str:
    return os.path.normcase(os.path.realpath(os.path.abspath(path)))


def test_pocket_single_voice_cache_evicts_on_switch(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = make_model()

    assert not isinstance(generate(model, str(voice_a)), str)
    assert not isinstance(generate(model, str(voice_b)), str)
    assert not isinstance(generate(model, str(voice_b)), str)  # reused
    assert not isinstance(generate(model, str(voice_a)), str)  # re-prepared

    assert model.model.get_state_calls == [str(voice_a), str(voice_b), str(voice_a)]
    assert len(model._voice_clone_cache) == 1

    # The cache now holds only voice A, built by the third (most recent) call
    key_a = next(k for k in model._voice_clone_cache if k[0] == cache_key_prefix(str(voice_a)))
    assert key_a[1] == ""
    assert model._voice_clone_cache[key_a] is model.model.created_states[2]


def test_pocket_rebuilds_changed_voice_and_clears_cache(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"first")
    model = make_model()

    assert not isinstance(generate(model, str(voice_path)), str)
    voice_path.write_bytes(b"different file contents")
    assert not isinstance(generate(model, str(voice_path)), str)

    assert model.model.get_state_calls == [str(voice_path), str(voice_path)]
    assert len(model._voice_clone_cache) == 1

    model.kill()
    assert model.model is None
    assert getattr(model, "_voice_clone_cache", None) in (None, {})


def test_pocket_error_string_on_state_failure(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def boom(voice_path):
        raise ValueError("boom")

    model.model.get_state_for_audio_prompt = boom

    assert generate(model, str(voice_path)) == (
        f"Couldn't create voice clone for {voice_path} - ValueError: boom"
    )
    assert getattr(model, "_voice_clone_cache", None) in (None, {})


def test_pocket_no_voice_path_is_uncached(tmp_path):
    model = make_model()

    assert not isinstance(generate(model, ""), str)

    assert model.model.get_state_calls == [""]
    assert getattr(model, "_voice_clone_cache", None) in (None, {})


def test_pocket_no_audio_output(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"x")
    model = make_model()

    def no_chunks(voice_state, text, max_tokens=None):
        if False:
            yield

    model.model.generate_audio_stream = no_chunks

    assert generate(model, str(voice_path)) == "No audio output"


def test_pocket_predefined_voice_origin_guard(tmp_path):
    model = make_model()
    name = next(iter(_ORIGINS_OF_PREDEFINED_VOICES))

    # No language config: cannot resolve a predefined voice
    model.model.origin = None
    with pytest.raises(ValueError):
        model._resolve_voice_source_path(name)

    # Origin outside the library's configs dir: same
    model.model.origin = tmp_path / "elsewhere" / "config.json"
    with pytest.raises(ValueError):
        model._resolve_voice_source_path(name)


def test_pocket_predefined_voice_resolves_under_configs(tmp_path):
    """With a valid origin, resolution goes through the library's own path."""
    model = make_model()
    name = next(iter(_ORIGINS_OF_PREDEFINED_VOICES))
    model.model.origin = os.path.join(CONFIGS_DIR, "en", "config.json")

    # get_predefined_voice may return an hf:// or URL reference, which would
    # then be downloaded; to stay offline, skip when the library says the
    # resolved reference is remote.
    from pocket_tts.utils.utils import get_predefined_voice  # type: ignore

    ref = get_predefined_voice(language="en", name=name)
    if str(ref).startswith(("http://", "https://", "hf://")):
        pytest.skip("predefined voice reference is remote; offline test")

    resolved = model._resolve_voice_source_path(name)
    assert os.path.isfile(resolved)
