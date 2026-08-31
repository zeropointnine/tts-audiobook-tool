import os

import pytest

from tts_audiobook_tool.tts_models.tts_base_model import TtsBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


class FakeTtsModel(TtsBaseModel):
    INFO = TtsModelType.NONE.value

    def kill(self) -> None:
        self.clear_voice_clone_cache()

    def generate_using_project(self, *args, **kwargs):
        return []

    def get_or_create(self, source_path: str, transcript: str, factory):
        return self._get_or_create_voice_clone(source_path, transcript, factory)


class MultiVoiceFakeTtsModel(FakeTtsModel):
    RETAINS_MULTIPLE_VOICE_CLONES = True


def make_factory(calls: list[str], value: str):
    def factory():
        calls.append(value)
        return value

    return factory


def test_voice_clone_cache_key_is_path_transcript_mtime_and_size(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice data")
    source_stat = voice_path.stat()

    key = FakeTtsModel._make_voice_clone_cache_key(str(voice_path), "reference words")

    assert key == (
        os.path.normcase(os.path.realpath(os.path.abspath(voice_path))),
        "reference words",
        source_stat.st_mtime_ns,
        source_stat.st_size,
    )


def test_same_file_through_relative_and_absolute_paths_is_one_cache_entry(tmp_path, monkeypatch):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice data")
    monkeypatch.chdir(tmp_path)
    model = MultiVoiceFakeTtsModel()
    calls: list[str] = []

    first = model.get_or_create("voice.wav", "words", make_factory(calls, "prepared"))
    second = model.get_or_create(str(voice_path), "words", make_factory(calls, "unexpected"))

    assert first == second == "prepared"
    assert calls == ["prepared"]


def test_changed_transcript_or_file_rebuilds_and_replaces_same_path_entry(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice data")
    model = MultiVoiceFakeTtsModel()
    calls: list[str] = []

    assert model.get_or_create(str(voice_path), "first", make_factory(calls, "one")) == "one"
    assert model.get_or_create(str(voice_path), "first", make_factory(calls, "unused")) == "one"
    assert model.get_or_create(str(voice_path), "second", make_factory(calls, "two")) == "two"
    assert len(model._voice_clone_cache) == 1

    voice_path.write_bytes(b"different voice data and size")
    assert model.get_or_create(str(voice_path), "second", make_factory(calls, "three")) == "three"

    source_stat = voice_path.stat()
    os.utime(
        voice_path,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns + 1_000_000),
    )
    assert model.get_or_create(str(voice_path), "second", make_factory(calls, "four")) == "four"

    assert calls == ["one", "two", "three", "four"]
    assert len(model._voice_clone_cache) == 1


def test_model_without_multi_voice_support_retains_only_current_clone(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = FakeTtsModel()
    calls: list[str] = []

    assert model.get_or_create(str(voice_a), "a", make_factory(calls, "a1")) == "a1"
    assert model.get_or_create(str(voice_b), "b", make_factory(calls, "b1")) == "b1"
    assert model.get_or_create(str(voice_a), "a", make_factory(calls, "a2")) == "a2"

    assert calls == ["a1", "b1", "a2"]
    assert len(model._voice_clone_cache) == 1


def test_model_with_multi_voice_support_reuses_a_b_a(tmp_path):
    voice_a = tmp_path / "a.wav"
    voice_b = tmp_path / "b.wav"
    voice_a.write_bytes(b"a")
    voice_b.write_bytes(b"b")
    model = MultiVoiceFakeTtsModel()
    calls: list[str] = []

    assert model.get_or_create(str(voice_a), "a", make_factory(calls, "a1")) == "a1"
    assert model.get_or_create(str(voice_b), "b", make_factory(calls, "b1")) == "b1"
    assert model.get_or_create(str(voice_a), "a", make_factory(calls, "unused")) == "a1"

    assert calls == ["a1", "b1"]
    assert len(model._voice_clone_cache) == 2


def test_factory_failure_is_not_cached(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")
    model = MultiVoiceFakeTtsModel()

    def fail():
        raise RuntimeError("preparation failed")

    with pytest.raises(RuntimeError, match="preparation failed"):
        model.get_or_create(str(voice_path), "words", fail)

    assert getattr(model, "_voice_clone_cache", {}) == {}


def test_clear_voice_clone_cache_is_idempotent(tmp_path):
    voice_path = tmp_path / "voice.wav"
    voice_path.write_bytes(b"voice")
    model = MultiVoiceFakeTtsModel()
    model.get_or_create(str(voice_path), "words", lambda: object())

    model.clear_voice_clone_cache()
    model.clear_voice_clone_cache()

    assert model._voice_clone_cache == {}
