"""A job validates its voice dependencies, not the entire shared voice library."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_types.voice import Voice, VoiceCrop
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil


def make_project(tmp_path, mode=VoiceSelectMode.USER_DEFINED):
    project = Project(
        dir_path=str(tmp_path),
        tts_model_type="glm_local",
        voice_select_mode=mode.id,
        voice_references=[Voice("ok.flac", "text"), Voice(), Voice("gone.flac", "text"), Voice("empty.flac")],
    )
    project.phrase_groups = [
        PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=index)
        for index in (0, 1, 2, 3, -1, 99)
    ]
    return project


@pytest.mark.parametrize("mode, lines, expected", [
    (VoiceSelectMode.DISABLED, [1, 2, 3], [0]),
    (VoiceSelectMode.USER_DEFINED, [0, 4, 0], [0]),
    (VoiceSelectMode.USER_DEFINED, [1, 3], [1, 3]),
    (VoiceSelectMode.USER_DEFINED, [4, 5], [0, 3]),
    (VoiceSelectMode.AUTO_ADVANCE, [0], [0, 1, 2, 3]),
    (VoiceSelectMode.DISABLED, None, [0]),
    (VoiceSelectMode.USER_DEFINED, None, [0]),
    (VoiceSelectMode.AUTO_ADVANCE, None, [0, 1, 2, 3]),
    (VoiceSelectMode.DISABLED, [], []),
    (VoiceSelectMode.USER_DEFINED, [], []),
    (VoiceSelectMode.AUTO_ADVANCE, [], []),
])
def test_job_voice_indices_follow_inference_without_dropping_empty_slots(tmp_path, mode, lines, expected):
    project = make_project(tmp_path, mode)
    voices = list(project.voice_references)
    assignments = [group.voice_index for group in project.phrase_groups]
    # Also accept generators without losing their first entry during emptiness checks.
    indices = None if lines is None else (index for index in lines)

    assert ProjectVoiceUtil.get_job_voice_indices(project, indices) == expected
    assert project.voice_references == voices
    assert [group.voice_index for group in project.phrase_groups] == assignments


def test_job_voice_indices_no_sample_binding(tmp_path):
    project = make_project(tmp_path, VoiceSelectMode.AUTO_ADVANCE)
    project.tts_model_type = "none"
    assert ProjectVoiceUtil.get_job_voice_indices(project, [0]) == []


@pytest.mark.parametrize("mode, lines", [
    (VoiceSelectMode.DISABLED, [1, 2, 3]),
    (VoiceSelectMode.USER_DEFINED, [0, 4]),
])
def test_preflight_ignores_unused_missing_samples_files_and_transcripts(tmp_path, monkeypatch, mode, lines):
    project = make_project(tmp_path, mode)
    (tmp_path / "ok.flac").write_bytes(b"sample")
    # This file exists, but must not be loaded or auto-transcribed for this job.
    (tmp_path / "empty.flac").write_bytes(b"sample")
    state = SimpleNamespace(project=project)
    load = Mock(return_value=object())
    transcribe = Mock(side_effect=AssertionError("unused voice must not be transcribed"))
    pause = Mock()
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", load)
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", transcribe)
    monkeypatch.setattr(voice_menu_shared.ask, "ask_enter_to_continue", pause)
    scope = ProjectVoiceUtil.get_job_voice_indices(project, lines)

    assert VoiceMenuShared.get_voice_problems(state, voice_indices=scope) == []
    assert VoiceMenuShared.validate_voices(state, voice_indices=scope) is True
    assert [call.args[0] for call in load.call_args_list] == [str(tmp_path / "ok.flac")] * 2
    transcribe.assert_not_called()
    pause.assert_not_called()
    assert not (tmp_path / "project.json").exists()


def test_auto_advance_preflight_checks_all_voices_even_for_one_line(tmp_path, monkeypatch):
    project = make_project(tmp_path, VoiceSelectMode.AUTO_ADVANCE)
    (tmp_path / "ok.flac").write_bytes(b"sample")
    (tmp_path / "empty.flac").write_bytes(b"sample")
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda _: object())
    state = SimpleNamespace(project=project)
    scope = ProjectVoiceUtil.get_job_voice_indices(project, [0])

    assert VoiceMenuShared.get_voice_problems(state, voice_indices=scope) == [
        "Voice 2 has no voice sample (set a sample or remove the voice)",
        "Voice file gone.flac not found",
        "Voice file empty.flac has no transcript",
    ]


@pytest.mark.parametrize("other_has_sample", [False, True])
def test_required_sample_is_checked_on_selected_slot_not_project_inventory(tmp_path, other_has_sample):
    project = make_project(tmp_path)
    project.voice_references = [Voice("elsewhere.flac", "text") if other_has_sample else Voice(), Voice()]
    state = SimpleNamespace(project=project)

    assert VoiceMenuShared.get_voice_problems(state, voice_indices=[1]) == [
        "Voice 2 has no voice sample (set a sample or remove the voice)",
    ]


def test_selected_sample_less_voice_is_allowed_when_model_does_not_require_samples(tmp_path, monkeypatch):
    project = make_project(tmp_path)
    project.tts_model_type = "dots_local"
    state = SimpleNamespace(project=project)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda _: pytest.fail("no selected sample"))
    scope = ProjectVoiceUtil.get_job_voice_indices(project, [1])

    assert scope == [1]
    assert VoiceMenuShared.get_voice_problems(state, voice_indices=scope) == []
    assert VoiceMenuShared.validate_voices(state, voice_indices=scope) is True


@pytest.mark.parametrize("mode", list(VoiceSelectMode))
def test_empty_job_does_not_require_any_sample(tmp_path, monkeypatch, mode):
    project = make_project(tmp_path, mode)
    state = SimpleNamespace(project=project)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda _: pytest.fail("no files should be loaded"))
    scope = ProjectVoiceUtil.get_job_voice_indices(project, [])

    assert VoiceMenuShared.get_voice_problems(state, voice_indices=scope) == []
    assert VoiceMenuShared.validate_voices(state, voice_indices=scope) is True


@pytest.mark.parametrize("cropped", [False, True])
def test_scoped_preflight_repairs_only_selected_transcript_once(tmp_path, monkeypatch, cropped):
    project = make_project(tmp_path)
    selected = Voice("selected.flac")
    if cropped:
        selected = Voice("selected.flac", "original", VoiceCrop("trim-selected.flac", 0.0, 3.0, ""))
    project.voice_references = [Voice("unused.flac"), selected]
    sample_path = tmp_path / "voice" / "crops" / "trim-selected.flac" if cropped else tmp_path / "selected.flac"
    sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_path.write_bytes(b"sample")
    state = SimpleNamespace(project=project)
    load = Mock(return_value=object())
    transcribe = Mock(return_value=("repaired", ""))
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", load)
    monkeypatch.setattr(VoiceMenuShared, "transcribe_voice_sample_to_text", transcribe)

    assert VoiceMenuShared.validate_voices(state, voice_indices=iter([1, 1])) is True
    load.assert_called_once_with(str(sample_path))
    transcribe.assert_called_once()
    assert project.voice_references[0] == Voice("unused.flac")
    updated = project.voice_references[1]
    assert (updated.crop.transcript if cropped else updated.transcript) == "repaired"
    if cropped:
        assert updated.transcript == "original"


@pytest.mark.parametrize("model_id, setting, preset", [
    ("pocket_local", None, "alba"),
    ("qwen3tts_local", ("model_type", "custom_voice"), "Ryan"),
    ("qwen3tts_local", ("model_type", "voice_design"), ""),
])
def test_scoped_preflight_preserves_model_sample_bypass(tmp_path, monkeypatch, model_id, setting, preset):
    project = make_project(tmp_path)
    project.tts_model_type = model_id
    if setting:
        project.set_model_setting(model_id, *setting)
    if preset:
        # A voice's preset takes precedence over its sample
        project.voice_references = [replace(voice, presets={model_id: preset})
                                    for voice in project.voice_references]
    state = SimpleNamespace(project=project)
    monkeypatch.setattr(voice_menu_shared.SoundFileUtil, "load", lambda _: pytest.fail("samples are not active"))

    assert VoiceMenuShared.get_voice_problems(state, voice_indices=[0, 1, 2, 3]) == []
    assert VoiceMenuShared.validate_voices(state, voice_indices=[0, 1, 2, 3]) is True


def test_qwen_custom_voice_preflight_requires_speakers_of_job_voices_only(tmp_path):
    project = make_project(tmp_path)
    project.tts_model_type = "qwen3tts_local"
    project.set_model_setting("qwen3tts_local", "model_type", "custom_voice")
    voices = project.voice_references
    project.voice_references = [replace(voices[0], presets={"qwen3tts_local": "Ryan"}), *voices[1:]]
    state = SimpleNamespace(project=project)

    assert VoiceMenuShared.get_voice_problems(state, voice_indices=[0]) == []
    assert VoiceMenuShared.get_voice_problems(state, voice_indices=[0, 2]) == [
        "Voice 3 has no speaker (select a speaker)"
    ]


def test_generation_utility_preflights_only_submitted_lines(tmp_path, monkeypatch):
    from tts_audiobook_tool import generate_util
    from tts_audiobook_tool.textual import generation_app

    project = make_project(tmp_path)
    state = SimpleNamespace(project=project, prefs=object())
    monkeypatch.setattr(generate_util.ProjectUtil, "get_selected_indices_not_generated", lambda _: {3})
    monkeypatch.setattr(generate_util.app_hint_util, "show_pre_inference_hints", lambda *_: True)
    validate = Mock(return_value=True)
    run = Mock(return_value=SimpleNamespace(completed=False))
    monkeypatch.setattr(VoiceMenuShared, "validate_voices", validate)
    monkeypatch.setattr(generation_app, "run_generation_app", run)

    GenerateUtil.do_generate_using_project_and_state(state)

    validate.assert_called_once_with(state, voice_indices=[3])
    assert run.call_args.kwargs["indices"] == {3}
