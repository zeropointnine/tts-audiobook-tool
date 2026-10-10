"""Consumers of the authoritative project-wide voice list (not migration internals)."""
from __future__ import annotations

from copy import deepcopy
from importlib import import_module
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from tts_audiobook_tool.app_types import Sound, SttVariant
from tts_audiobook_tool.constants import APP_SAMPLE_RATE
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.model_worker_protocol import TtsInspected
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.menus.voice import voice_menu_shared, voice_pocket_menu
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.project_support.project_transfer_util import ProjectTransferUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.voice_reference_migration import VoiceReferenceMigrationRequired
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.pocket_base_model import PocketBaseModel
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@pytest.fixture(autouse=True)
def no_disk_save(monkeypatch):
    monkeypatch.setattr(Project, "save", lambda self: "")


def references():
    return [
        {"file_name": "a.flac", "transcript": "First reference."},
        {"file_name": "b.flac", "transcript": "Second reference."},
        {"file_name": "c.flac", "transcript": "Third reference."},
    ]


def model(model_id="mira_local"):
    return TtsModelType.require_by_id(model_id)


def sound():
    return Sound(np.full(100, 0.1, dtype=np.float32), APP_SAMPLE_RATE)


def long_sound(seconds):
    return Sound(np.full(int(APP_SAMPLE_RATE * seconds), 0.1, dtype=np.float32), APP_SAMPLE_RATE)


@pytest.mark.parametrize("model_id", ["mira_local", "dots_local", "indextts2_local", "pocket_local", "qwen3tts_sglomni"])
def test_models_read_identical_pairs_even_without_transcript_capability(model_id):
    project = Project(voice_references=references())
    assert ProjectVoiceUtil.voice_reference_pairs(project, model(model_id)) == [
        (ref["file_name"], ref["transcript"]) for ref in references()
    ]
    assert ProjectVoiceUtil.current_voice_reference_pair(project, model(model_id), 4) == (
        "b.flac", "Second reference."
    )
    detached = ProjectVoiceUtil.get_voice_values(project, model(model_id))
    detached.clear()
    assert project.voice_references == references()


def test_append_non_transcript_model_stores_text_and_disambiguates_shared_sample(tmp_path, monkeypatch):
    project = Project(dir_path=str(tmp_path), voice_references=references())
    monkeypatch.setattr(SoundFileUtil, "save_flac", lambda *_: "")
    assert not ProjectVoiceUtil.set_voice_and_save(project, sound(), "a", "New text.", model(), append=True)
    assert project.voice_references == references() + [{"file_name": "a_2.flac", "transcript": "New text."}]
    assert ProjectVoiceUtil.get_voice_transcript_values(project, model("dots_local"))[-1] == "New text."


def test_replacement_from_another_model_uses_same_list(tmp_path, monkeypatch):
    project = Project(dir_path=str(tmp_path), voice_references=references())
    monkeypatch.setattr(SoundFileUtil, "save_flac", lambda *_: "")
    assert not ProjectVoiceUtil.set_voice_and_save(project, sound(), "a", "Replacement", model())
    assert project.voice_references == [{"file_name": "a.flac", "transcript": "Replacement"}]
    assert ProjectVoiceUtil.get_voice_values(project, model("dots_local")) == ["a.flac"]


@pytest.mark.parametrize("index", [0, 1, 2])
def test_remove_atomically_preserves_remaining_pairs(index):
    project = Project(voice_references=references())
    assert ProjectVoiceUtil.remove_voice_at_index_and_save(project, model(), index) == references()[index]["file_name"]
    expected = references()
    expected.pop(index)
    assert project.voice_references == expected


@pytest.mark.parametrize("index", [-1, 3])
def test_bad_remove_index_does_not_modify_list(index):
    project = Project(voice_references=references())
    with pytest.raises(IndexError):
        ProjectVoiceUtil.remove_voice_at_index_and_save(project, model(), index)
    assert project.voice_references == references()


def test_edit_transcript_is_capability_independent_and_pair_preserving():
    project = Project(voice_references=references())
    assert not ProjectVoiceUtil.set_voice_transcript_at_index_and_save(project, 1, "Updated text")
    expected = references()
    expected[1]["transcript"] = "Updated text"
    assert project.voice_references == expected


def test_secondary_set_and_clear_never_touch_shared_list(tmp_path, monkeypatch):
    project = Project(dir_path=str(tmp_path), voice_references=references())
    monkeypatch.setattr(SoundFileUtil, "save_flac", lambda *_: "")
    assert not ProjectVoiceUtil.set_voice_and_save(
        project, sound(), "a", "Ignored secondary text", model("indextts2_local"), is_secondary=True
    )
    assert project.get_model_setting("indextts2_local", "emo_voice") == "a_2.flac"
    assert project.voice_references == references()
    ProjectVoiceUtil.clear_voice_and_save(project, model("indextts2_local"), is_secondary=True)
    assert project.get_model_setting("indextts2_local", "emo_voice") == ""
    assert project.voice_references == references()



def test_transfer_collects_shared_once_plus_secondary_files(monkeypatch):
    project = Project(voice_references=references() + [references()[0]])
    project.set_model_setting("indextts2_local", "emo_voice", "emo.flac")
    monkeypatch.setattr(ProjectVoiceUtil, "get_voice_values", lambda *_: pytest.fail("must collect shared list directly"))
    _, names = ProjectTransferUtil.collect_supporting_project_file_names(project)
    assert names == ["a.flac", "b.flac", "c.flac", "emo.flac"]


def test_pocket_preset_selection_retains_shared_list(monkeypatch):
    project = Project(tts_model_type="pocket_local", voice_references=references())
    state = cast(State, SimpleNamespace(project=project))
    captured = {}
    monkeypatch.setattr(voice_pocket_menu.MenuUtil, "options_menu", lambda **kwargs: captured.update(kwargs))
    voice_pocket_menu.select_predefined_voice(state)
    captured["on_select"]("alba")
    assert project.voice_references == references()
    voice_pocket_menu.select_predefined_voice(state)
    assert captured["current_value"] == "alba"
    assert PocketBaseModel.get_primary_voice_value(project) == "alba"
    assert ProjectVoiceUtil.get_voice_label(project) == "alba"
    assert Tts.get_voice_value_count(project) == 0
    assert Tts.get_voice_tag_for_selection_index(project, 2) == "alba"
    class StubPocket(PocketBaseModel):
        def kill(self):
            pass

        def generate_using_project(self, *args, **kwargs):
            raise AssertionError("not generating in readiness test")

    instance = StubPocket()
    assert PocketBaseModel.get_gated_error_message(project, instance) == ""
    assert not PocketBaseModel.get_blocking_issues(project, instance)
    display = PocketBaseModel.get_voice_display_info(project)
    assert display is not None
    assert "alba" in str(display) and "+2" not in str(display)
    voice_pocket_menu.clear_predefined_voice(state)
    assert project.voice_references == references()
    assert PocketBaseModel.get_primary_voice_value(project) == "a.flac"
    assert Tts.get_voice_value_count(project) == 3
    assert Tts.get_voice_tag_for_selection_index(project, 2) == "c"


def test_pocket_removing_shared_list_does_not_clear_preset():
    project = Project(voice_references=references())
    project.set_model_setting("pocket_local", "predefined_voice", "alba")
    ProjectVoiceUtil.clear_voice_and_save(project, model("pocket_local"))
    assert project.voice_references == []
    assert project.get_model_setting("pocket_local", "predefined_voice") == "alba"


def test_transcript_menu_edits_selected_pair_for_model_without_requirement(monkeypatch):
    project = Project(tts_model_type="mira_local", voice_references=references())
    state = cast(State, SimpleNamespace(project=project))
    answers = iter(["2", "Edited Reference."])
    inputs = []
    def ask_input(**kwargs):
        inputs.append(kwargs)
        return next(answers)
    monkeypatch.setattr(voice_menu_shared.ask, "ask_input", ask_input)
    monkeypatch.setattr(voice_menu_shared, "print_feedback", lambda *_: None)
    VoiceMenuShared.edit_voice_sample_transcript(state)
    expected = references()
    expected[1]["transcript"] = "Edited Reference."
    assert project.voice_references == expected
    # The transcript prompt prefilled with the current transcript.
    assert inputs[-1]["prefill"] == "Second reference."


@pytest.mark.parametrize("secondary", [False, True])
def test_import_nonrequiring_model_retains_sidecar_but_secondary_ignores_it(tmp_path, monkeypatch, secondary):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"sound")
    path.with_suffix(".txt").write_text("Sidecar text", encoding="utf-8")
    project = Project(dir_path=str(tmp_path))
    prefs = SimpleNamespace(last_voice_dir="", save=lambda: None, stt_variant=SttVariant.DISABLED)
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    monkeypatch.setattr(VoiceMenuShared, "ask_voice_file", lambda *_: str(path))
    monkeypatch.setattr(SoundFileUtil, "load", lambda *_: long_sound(2.5))
    monkeypatch.setattr(SoundFileUtil, "save_flac", lambda *_: "")
    monkeypatch.setattr(voice_menu_shared.SoundPipeline, "apply_voice_clone_post_processing", lambda value: value)
    monkeypatch.setattr(voice_menu_shared.PlaySoundUtil, "play_sound_async", lambda *_: None)
    monkeypatch.setattr(voice_menu_shared.hints, "show_hint_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(voice_menu_shared.Transcriber, "transcribe_to_words", lambda *_: pytest.fail("unexpected STT"))
    monkeypatch.setattr(voice_menu_shared, "print_feedback", lambda *_: None)
    VoiceMenuShared.ask_and_set_voice_file(state, model("indextts2_local" if secondary else "mira_local"), is_secondary=secondary)
    assert project.voice_references == ([] if secondary else [{"file_name": "clip.flac", "transcript": "Sidecar text"}])


def test_transcript_sidecar_prefers_stem_txt_and_skips_blank(tmp_path):
    # "clip.txt" wins over "clip.wav.txt", but a blank one falls through to
    # the appended-extension file instead of hiding it.
    path = tmp_path / "clip.wav"
    stem_txt = path.with_suffix(".txt")
    appended_txt = tmp_path / "clip.wav.txt"
    appended_txt.write_text("Appended text", encoding="utf-8")
    assert VoiceMenuShared.load_transcript_sidecar(str(path)) == (appended_txt, "Appended text")
    stem_txt.write_text("  \n", encoding="utf-8")
    assert VoiceMenuShared.load_transcript_sidecar(str(path)) == (appended_txt, "Appended text")
    stem_txt.write_text("Stem text\n", encoding="utf-8")
    assert VoiceMenuShared.load_transcript_sidecar(str(path)) == (stem_txt, "Stem text")
    appended_txt.unlink()
    stem_txt.write_text("", encoding="utf-8")
    assert VoiceMenuShared.load_transcript_sidecar(str(path)) is None


def _setup_short_import(tmp_path, monkeypatch, raw_seconds, trimmed_seconds):
    path = tmp_path / "clip.wav"
    path.write_bytes(b"sound")
    project = Project(dir_path=str(tmp_path))
    prefs = SimpleNamespace(last_voice_dir="", save=lambda: None, stt_variant=SttVariant.DISABLED)
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))
    errors: list[str] = []
    monkeypatch.setattr(VoiceMenuShared, "ask_voice_file", lambda *_: str(path))
    monkeypatch.setattr(SoundFileUtil, "load", lambda *_: long_sound(raw_seconds))
    monkeypatch.setattr(voice_menu_shared.SoundPipeline, "apply_voice_clone_post_processing", lambda value: long_sound(trimmed_seconds))
    monkeypatch.setattr(voice_menu_shared.ask, "ask_error", errors.append)
    monkeypatch.setattr(voice_menu_shared.PlaySoundUtil, "play_sound_async", lambda *_: pytest.fail("unexpected playback"))
    monkeypatch.setattr(voice_menu_shared.hints, "show_hint_if_necessary", lambda *_, **__: None)
    monkeypatch.setattr(voice_menu_shared.Transcriber, "transcribe_to_words", lambda *_: pytest.fail("unexpected STT"))
    return state, project, errors


def test_import_rejects_raw_sound_under_two_seconds(tmp_path, monkeypatch):
    state, project, errors = _setup_short_import(tmp_path, monkeypatch, 1.0, 1.0)
    VoiceMenuShared.ask_and_set_voice_file(state, model())
    assert errors == ["Sound file must be at least 2 seconds"]
    assert project.voice_references == []


def test_import_rejects_sound_under_two_seconds_after_trim(tmp_path, monkeypatch):
    state, project, errors = _setup_short_import(tmp_path, monkeypatch, 3.0, 1.0)
    VoiceMenuShared.ask_and_set_voice_file(state, model())
    assert errors == ["Sound file post silence trim must be at least 2 seconds"]
    assert project.voice_references == []


def test_abr_validation_migrates_single_source_without_mutating_snapshot():
    snapshot = {"version": 3, "mira_voice_file_name": ["a.flac"], "mira_voice_transcript": ["Legacy text"]}
    original = deepcopy(snapshot)
    result = ProjectTransferUtil.validate_abr_snapshot(snapshot)
    assert result.voice_references == [{"file_name": "a.flac", "transcript": "Legacy text"}]
    assert snapshot == original


def test_abr_validation_requires_explicit_choice_and_can_prompt(monkeypatch):
    snapshot = {"version": 3, "mira_voice_file_name": ["a.flac"], "dots_voice_file_name": ["b.flac"]}
    original = deepcopy(snapshot)
    with pytest.raises(VoiceReferenceMigrationRequired):
        ProjectTransferUtil.validate_abr_snapshot(snapshot)
    import tts_audiobook_tool.project_support.voice_reference_migration as migration
    monkeypatch.setattr(migration, "choose_voice_reference_source", lambda sources, dir_name=None: next(s for s in sources if s.references[0]["file_name"] == "b.flac"))
    result = ProjectTransferUtil.validate_abr_snapshot(snapshot, prompt_on_migration=True)
    assert result.voice_references == [{"file_name": "b.flac", "transcript": ""}]
    assert snapshot == original


@pytest.mark.parametrize("stale_refs", ["not an array", [{"file_name": 123, "transcript": False}]])
def test_abr_authoritative_empty_list_ignores_malformed_scoped_clones(stale_refs):
    snapshot = {
        "version": 3,
        "voice_references": [],
        "model_settings": {"models": {"mira_local": {"voice_references": stale_refs}}},
    }
    original = deepcopy(snapshot)
    result = ProjectTransferUtil.validate_abr_snapshot(snapshot)
    assert result.voice_references == []
    assert all("voice_references" not in obj for obj in result.model_settings.models.values())
    assert snapshot == original


def test_abr_preparation_does_not_hide_malformed_nonvoice_settings():
    snapshot = {
        "voice_references": [],
        "model_settings": {"models": {"indextts2_local": {"files": "not an object"}}},
    }
    with pytest.raises(ValueError, match="files must be an object"):
        ProjectTransferUtil.validate_abr_snapshot(snapshot)


def test_abr_nonvoice_parameter_pruning_still_follows_normal_validation():
    snapshot = {
        "voice_references": [],
        "model_settings": {"models": {"mira_local": {"parameters": "not an object"}}},
    }
    result = ProjectTransferUtil.validate_abr_snapshot(snapshot)
    assert result.get_model_setting("mira_local", "temperature") == Project().get_model_setting("mira_local", "temperature")


@pytest.mark.parametrize("snapshot", [{"unrelated": 1}, {"model_settings": {"unrelated": 1}}, {"version": True, "voice_references": []}, {"version": "3", "voice_references": []}])
def test_abr_preparation_does_not_hide_invalid_original_snapshot(snapshot):
    with pytest.raises(ValueError):
        ProjectTransferUtil.validate_abr_snapshot(snapshot)


STATIC_MENUS = [
    ("chatterbox", "Chatterbox", "chatterbox_local"),
    ("dots", "Dots", "dots_local"),
    ("fish_s1", "FishS1", "fish_s1_local"),
    ("fish_s2", "FishS2", "fish_s2_local"),
    ("glm", "Glm", "glm_local"),
    ("higgs_v2", "HiggsV2", "higgs_v2_local"),
    ("indextts2", "IndexTts2", "indextts2_local"),
    ("mira", "Mira", "mira_local"),
    ("moss", "Moss", "moss_local"),
    ("omnivoice", "OmniVoice", "omnivoice_local"),
    ("pocket", "Pocket", "pocket_local"),
    ("qwen3", "Qwen3", "qwen3tts_local"),
    ("vibevoice", "VibeVoice", "vibevoice_local"),
]


@pytest.mark.parametrize("module_name,class_name,model_id", STATIC_MENUS)
def test_static_voice_menus_keep_entire_sample_group(monkeypatch, module_name, class_name, model_id):
    project = Project(tts_model_type=model_id, voice_references=references())
    state = cast(State, SimpleNamespace(project=project))
    labels = []

    def capture_menu(current, make_items):
        labels.extend(strip_ansi_codes(get_string_from(current, item.label)) for item in make_items(current))

    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", capture_menu)
    module = import_module(f"tts_audiobook_tool.menus.voice.voice_{module_name}_menu")
    getattr(module, f"Voice{class_name}Menu").menu(state)

    expected = [
        voice_menu_shared.LABEL_ADD_VOICE_SAMPLE,
        voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
        voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
        voice_menu_shared.LABEL_VOICE_SELECTION_MODE,
        voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS,
    ]
    expected.extend({
        "indextts2_local": ["Select emotion voice sample", "Emotion vector", "Emotion alpha (strength)"],
        "omnivoice_local": ["Voice design instructions"],
        "pocket_local": ["Select predefined voice"],
        "vibevoice_local": ["Select LoRA"],
    }.get(model_id, []))
    assert len(labels) == len(expected)
    assert all(label.startswith(prefix) for label, prefix in zip(labels, expected))


@pytest.mark.parametrize("module_name,class_name,model_id", STATIC_MENUS)
def test_static_model_menus_never_append_shared_sample_group(monkeypatch, module_name, class_name, model_id):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type=model_id, voice_references=references())))
    labels = []

    def capture_menu(current, make_items):
        labels.extend(strip_ansi_codes(get_string_from(current, item.label)) for item in make_items(current))

    monkeypatch.setattr(ModelMenuShared, "menu_wrapper", capture_menu)
    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", lambda *_args, **_kwargs: pytest.fail("Samples belong to Voice"))
    module = import_module(f"tts_audiobook_tool.menus.model.model_{module_name}_menu")
    menu = getattr(module, f"Model{class_name}Menu").menu
    if model_id == "qwen3tts_local":
        menu(state, TtsInspected(operation_id="test", tts_type_id=model_id, metadata={"model_type": "base"}))
    else:
        menu(state)

    assert labels
    assert any(label.startswith("Seed") for label in labels)
    assert not any(label.startswith((voice_menu_shared.LABEL_ADD_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
                                     "Edit voice sample transcript",
                                     "Add/remove voice samples", "Select voice clone sample",
                                     voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS,
                                     voice_menu_shared.LABEL_VOICE_SELECTION_MODE)) for label in labels)


def test_pocket_voice_menu_retains_validation_callback(monkeypatch):
    state = cast(State, SimpleNamespace(project=Project(tts_model_type="pocket_local")))
    captured = {}
    validated = []

    def sample_items(current, model_type, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", sample_items)
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", lambda current, make_items: make_items(current))
    monkeypatch.setattr(voice_pocket_menu, "validate_voice_file", validated.append)
    voice_pocket_menu.VoicePocketMenu.menu(state)
    captured["on_set_callback"]()
    assert validated == [state]


def test_mira_voice_menu_retains_clear_unload_callback(monkeypatch):
    from tts_audiobook_tool.menus.voice import voice_mira_menu

    state = cast(State, SimpleNamespace(project=Project(tts_model_type="mira_local")))
    captured = {}
    clears = []

    def sample_items(current, model_type, **kwargs):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", sample_items)
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", lambda current, make_items: make_items(current))
    monkeypatch.setattr(voice_mira_menu.ModelWorker, "clear_models_if_running_blocking", lambda: clears.append(True))
    voice_mira_menu.VoiceMiraMenu.menu(state)
    captured["on_clear_callback"]()
    assert clears == [True]
