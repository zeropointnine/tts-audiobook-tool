import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil


class FakeIndexTTS2:
    """Model the installed library's independent, path-only cache guards."""

    def __init__(self, **_kwargs):
        self.cache_spk_cond = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.speaker_loads = []
        self.emotion_loads = []
        self.infer_calls = 0
        self.fail_after_speaker = False

    def infer(self, spk_audio_prompt, emo_audio_prompt=None, emo_vector=None, **_kwargs):
        self.infer_calls += 1
        if emo_vector is not None:
            emo_audio_prompt = None
        if emo_audio_prompt is None:
            emo_audio_prompt = spk_audio_prompt

        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk_audio_prompt:
            contents = Path(spk_audio_prompt).read_bytes()
            self.speaker_loads.append(spk_audio_prompt)
            self.cache_spk_cond = contents
            self.cache_s2mel_style = ("style", contents)
            self.cache_s2mel_prompt = ("prompt", contents)
            self.cache_mel = ("mel", contents)
            self.cache_spk_audio_prompt = spk_audio_prompt

        if self.fail_after_speaker:
            self.fail_after_speaker = False
            raise RuntimeError("Inference failed after speaker preparation")

        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo_audio_prompt:
            self.emotion_loads.append(emo_audio_prompt)
            self.cache_emo_cond = Path(emo_audio_prompt).read_bytes()
            self.cache_emo_audio_prompt = emo_audio_prompt

        return 24000, np.zeros((10, 1), dtype=np.int16)


@pytest.fixture
def model(monkeypatch, tmp_path):
    # Load the real adapter, replacing only its optional model library. Avoid
    # caching this stub-backed adapter in sys.modules for unrelated tests.
    library = ModuleType("indextts.infer_v2")
    library.IndexTTS2 = FakeIndexTTS2
    package = ModuleType("indextts")
    package.__path__ = []
    monkeypatch.setitem(sys.modules, "indextts", package)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", library)
    path = Path(__file__).parents[1] / "tts_audiobook_tool/tts_models/indextts2_model.py"
    spec = importlib.util.spec_from_file_location("_test_indextts2_adapter", path)
    assert spec is not None and spec.loader is not None
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    monkeypatch.setattr(adapter.huggingface_hub, "snapshot_download", lambda **_: str(tmp_path))
    monkeypatch.setattr(adapter.app_support, "set_seed", lambda _: None)
    return adapter.IndexTts2Model(use_fp16=False)


def generate(model, voice, emotion=None, vector=None):
    return model.generate(
        text="Test sentence.", voice_path=str(voice), temperature=0.8,
        emo_alpha=0.65, emo_voice_path=str(emotion) if emotion else "",
        emo_vector=vector or [], top_p=0.8, top_k=30, seed=1,
    )


def rewrite(path, contents, *, keep_mtime=False):
    previous = path.stat()
    path.write_bytes(contents)
    mtime = previous.st_mtime_ns if keep_mtime else previous.st_mtime_ns + 1_000_000_000
    os.utime(path, ns=(previous.st_atime_ns, mtime))


def test_unchanged_reference_reuses_speaker_and_default_emotion(model, tmp_path):
    voice = tmp_path / "voice.flac"
    voice.write_bytes(b"alpha")
    assert isinstance(generate(model, voice), Sound)
    assert isinstance(generate(model, voice), Sound)
    assert model.model.speaker_loads == [str(voice)]
    assert model.model.emotion_loads == [str(voice)]


@pytest.mark.parametrize("keep_mtime,contents", [(False, b"bravo"), (True, b"longer contents")])
def test_in_place_change_reloads_all_speaker_conditioning_and_default_emotion(
    model, tmp_path, keep_mtime, contents,
):
    voice = tmp_path / "voice_crop.flac"
    voice.write_bytes(b"alpha")
    assert isinstance(generate(model, voice), Sound)
    rewrite(voice, contents, keep_mtime=keep_mtime)
    assert isinstance(generate(model, voice), Sound)
    assert model.model.speaker_loads == [str(voice), str(voice)]
    assert model.model.emotion_loads == [str(voice), str(voice)]
    assert model.model.cache_spk_cond == contents
    assert model.model.cache_s2mel_style == ("style", contents)
    assert model.model.cache_s2mel_prompt == ("prompt", contents)
    assert model.model.cache_mel == ("mel", contents)
    assert model.model.cache_emo_cond == contents


@pytest.mark.parametrize("changed", ["speaker", "emotion"])
def test_separate_emotion_reference_invalidates_only_affected_cache(model, tmp_path, changed):
    voice, emotion = tmp_path / "voice.flac", tmp_path / "emotion.flac"
    voice.write_bytes(b"speaker")
    emotion.write_bytes(b"emotion")
    assert isinstance(generate(model, voice, emotion), Sound)
    rewrite(voice if changed == "speaker" else emotion, b"changed")
    assert isinstance(generate(model, voice, emotion), Sound)
    assert len(model.model.speaker_loads) == (2 if changed == "speaker" else 1)
    assert len(model.model.emotion_loads) == (2 if changed == "emotion" else 1)
    assert model.model.cache_spk_cond == voice.read_bytes()
    assert model.model.cache_emo_cond == emotion.read_bytes()


def test_emotion_vector_uses_changed_speaker_not_ignored_external_clip(model, tmp_path):
    voice = tmp_path / "voice.flac"
    voice.write_bytes(b"alpha")
    ignored_emotion = tmp_path / "missing-emotion.flac"
    vector = [0.0] * 8
    assert isinstance(generate(model, voice, ignored_emotion, vector), Sound)
    rewrite(voice, b"bravo")
    assert isinstance(generate(model, voice, ignored_emotion, vector), Sound)
    assert model.model.emotion_loads == [str(voice), str(voice)]
    assert model.model.cache_emo_cond == b"bravo"


def test_switching_voices_keeps_library_single_slot_behavior(model, tmp_path):
    a, b = tmp_path / "a.flac", tmp_path / "b.flac"
    a.write_bytes(b"alpha")
    b.write_bytes(b"bravo")
    for voice in (a, b, a):
        assert isinstance(generate(model, voice), Sound)
    assert model.model.speaker_loads == [str(a), str(b), str(a)]
    assert model.model.emotion_loads == [str(a), str(b), str(a)]


def test_failed_inference_then_restored_file_does_not_reuse_partial_conditioning(model, tmp_path):
    voice = tmp_path / "voice.flac"
    voice.write_bytes(b"alpha")
    previous = voice.stat()
    assert isinstance(generate(model, voice), Sound)
    rewrite(voice, b"bravo")
    model.model.fail_after_speaker = True
    assert "Inference failed" in generate(model, voice)
    assert model.model.cache_spk_cond == b"bravo"

    voice.write_bytes(b"alpha")
    os.utime(voice, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert isinstance(generate(model, voice), Sound)
    assert model.model.cache_spk_cond == model.model.cache_emo_cond == b"alpha"
    assert len(model.model.speaker_loads) == 3


def test_missing_reference_returns_error_without_reusing_cached_audio(model, tmp_path):
    voice = tmp_path / "voice.flac"
    voice.write_bytes(b"alpha")
    assert isinstance(generate(model, voice), Sound)
    voice.unlink()
    result = generate(model, voice)
    assert isinstance(result, str) and "FileNotFoundError" in result
    assert model.model.infer_calls == 1


def test_reapplying_crop_is_consumed_by_retained_project_model(model, tmp_path):
    voice_dir = tmp_path / "voice"
    voice_dir.mkdir()
    data = np.concatenate([np.full(4 * 24000, 0.1), np.full(4 * 24000, 0.4)]).astype(np.float32)
    assert SoundFileUtil.save_flac(Sound(data, 24000), str(voice_dir / "voice.flac")) == ""
    project = Project.model_validate({
        "dir_path": str(tmp_path), "tts_model_type": "indextts2_local",
        "voice_references": [{"file_name": "voice.flac", "transcript": "original"}],
    })
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 0, 3, "first") == ""
    assert isinstance(model.generate_using_project(project, ["First sentence."]), list)
    entry = project.voice_references[0]
    crop_file_name = entry["crop_file_name"]
    crop = Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, entry))
    assert "/" not in crop_file_name and "\\" not in crop_file_name
    assert crop == voice_dir / "crops" / crop_file_name
    first_audio = crop.read_bytes()
    first_mtime = crop.stat().st_mtime_ns
    assert ProjectVoiceUtil.apply_voice_crop_and_save(project, 0, 4, 7, "second") == ""
    assert project.voice_references[0]["crop_file_name"] == crop_file_name
    assert Path(ProjectVoiceUtil.resolve_cropped_voice_file_path(project, project.voice_references[0])) == crop
    assert crop.read_bytes() != first_audio
    os.utime(crop, ns=(crop.stat().st_atime_ns, first_mtime + 1_000_000_000))
    assert isinstance(model.generate_using_project(project, ["Second sentence."]), list)
    assert model.model.speaker_loads == [str(crop), str(crop)]
    assert model.model.emotion_loads == [str(crop), str(crop)]
    assert model.model.cache_spk_cond == model.model.cache_emo_cond == crop.read_bytes()


def test_kill_clears_reference_bookkeeping(model, tmp_path):
    voice = tmp_path / "voice.flac"
    voice.write_bytes(b"alpha")
    assert isinstance(generate(model, voice), Sound)
    model.kill()
    assert model.model is None
    assert model._speaker_reference_key is None
    assert model._emotion_reference_key is None
    assert generate(model, voice) == "Model is not initialized"
