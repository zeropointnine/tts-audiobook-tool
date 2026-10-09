"""Entry-point and sample-management regressions for the voice/model split."""
from types import SimpleNamespace
from typing import cast

import pytest

from tts_audiobook_tool.menus import main_menu
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.model_settings import REGISTRY
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts


LOCAL_MODELS = [
    "chatterbox_local", "dots_local", "fish_s1_local", "fish_s2_local",
    "glm_local", "higgs_v2_local", "indextts2_local", "mira_local",
    "moss_local", "omnivoice_local", "pocket_local", "qwen3tts_local",
    "vibevoice_local",
]


def make_state(model_id="mira_local", voice_count=0, dir_path="/example/book"):
    project = Project(
        tts_model_type=model_id,
        voice_references=[{"file_name": f"voice-{i}.flac", "transcript": "text"}
                          for i in range(voice_count)],
    )
    # Routing only needs the path value; do not start a real directory watcher.
    project.dir_path = dir_path
    return cast(State, SimpleNamespace(project=project))


def test_main_menu_has_separate_voice_and_model_hotkeys(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_menu.MenuUtil, "menu",
                        lambda state, heading, items, **kwargs: captured.update(items=items(state)))
    main_menu.MainMenu.menu(make_state())
    items = captured["items"]
    voice = next(item for item in items if item.hotkey == "v")
    model = next(item for item in items if item.hotkey == "m")
    assert get_string_from(make_state(), voice.label) == "Voice clone"
    assert get_string_from(make_state(), model.label).startswith("Model settings")
    assert voice.handler is main_menu.on_voice
    assert model.handler is main_menu.on_model
    assert len({item.hotkey for item in items}) == len(items)


@pytest.mark.parametrize("entrypoint,menu_class", [
    (main_menu.on_voice, VoiceMenuShared), (main_menu.on_model, ModelMenuShared),
])
@pytest.mark.parametrize("model_id,dir_path,expected", [
    ("none", "/example/book", "Requires TTS model"),
    ("mira_local", "", "Requires a project"),
    ("mira_local", "/example/book", None),
])
def test_entrypoint_prerequisites_do_not_require_clone(
        monkeypatch, entrypoint, menu_class, model_id, dir_path, expected):
    state = make_state(model_id, dir_path=dir_path)
    errors, opened, bound = [], [], []
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: False)
    monkeypatch.setattr(Tts, "bind_project", bound.append)
    monkeypatch.setattr(main_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(menu_class, "menu", opened.append)
    entrypoint(state, None)
    assert bound == [state.project]
    assert errors == ([expected] if expected else [])
    assert opened == ([] if expected else [state])


@pytest.mark.parametrize("model_id", LOCAL_MODELS)
@pytest.mark.parametrize("voice_count", [0, 1, 2, 9])
def test_all_local_voice_menus_keep_sample_selection_group(
        monkeypatch, model_id, voice_count):
    state = make_state(model_id, voice_count)
    captured = {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items(state)))
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Sample management must not load the TTS model"))
    VoiceMenuShared.menu(state)
    items = captured["items"]
    labels = [get_string_from(state, item.label) for item in items]
    if model_id == "indextts2_local":
        assert [item.superlabel for item in items[-3:]] == ["Emotion", "", ""]
        emotion_labels = [strip_ansi_codes(label) for label in labels[-3:]]
        assert emotion_labels[:2] == ["Select emotion voice sample (optional)", "Emotion vector (optional)"]
        assert emotion_labels[2].startswith("Emotion alpha (strength)")
        assert "0.65" in emotion_labels[2] and "default" in emotion_labels[2]
        items, labels = items[:-3], labels[:-3]
    elif model_id in ("omnivoice_local", "pocket_local", "vibevoice_local"):
        expected_label = {
            "omnivoice_local": "Voice design instructions",
            "pocket_local": "Select predefined voice",
            "vibevoice_local": "Select LoRA",
        }[model_id]
        assert labels[-1].startswith(expected_label)
        assert items[-1].blank_line_before
        items, labels = items[:-1], labels[:-1]
    assert all(not item.superlabel for item in items)
    assert not any(label.startswith("Add/remove voice samples") for label in labels)
    if voice_count < 9:
        add_label = labels.pop(0)
        assert "voice" in add_label.lower() and "sample" in add_label.lower()
        if voice_count:
            assert add_label == voice_menu_shared.LABEL_ADD_VOICE_SAMPLE
    expected = [
        voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
        voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
    ]
    if voice_count >= 2:
        expected.insert(1, voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE)
        expected.append(voice_menu_shared.LABEL_VOICE_SELECTION_MODE)
    expected.append(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    assert len(labels) == len(expected)
    assert all(label.startswith(prefix) for label, prefix in zip(labels, expected))


@pytest.mark.parametrize("model_id", LOCAL_MODELS)
def test_all_local_models_route_settings_without_sample_controls(monkeypatch, model_id):
    state = make_state(model_id, voice_count=2)
    captured = {}
    monkeypatch.setattr(ModelMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items(state)))
    monkeypatch.setattr(ModelWorker, "get_model_state_blocking", lambda: (None, ""))
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking",
                        lambda *_: (SimpleNamespace(metadata={"model_type": "base"}), ""))
    ModelMenuShared.menu(state)
    labels = [get_string_from(state, item.label) for item in captured["items"]]
    assert labels
    assert all(not item.superlabel for item in captured["items"])
    assert not any(label.startswith((voice_menu_shared.LABEL_ADD_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
                                     "Add/remove voice samples", "Select voice clone sample",
                                     voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS,
                                     voice_menu_shared.LABEL_VOICE_SELECTION_MODE)) for label in labels)
    if model_id == "indextts2_local":
        assert not any("emotion" in label.lower() for label in labels)
        assert labels[0].startswith("FP16")
    elif model_id == "omnivoice_local":
        assert not any(label.startswith(("Voice design instructions", "Clear instructions")) for label in labels)
    elif model_id == "pocket_local":
        assert not any(label.startswith(("Select predefined voice", "Clear predefined voice")) for label in labels)
    elif model_id == "vibevoice_local":
        assert not any(label.startswith(("Select LoRA", "Clear LoRA")) for label in labels)
        assert labels[0].startswith("Select model")


@pytest.mark.parametrize("voice_count", [0, 1, 2])
@pytest.mark.parametrize("predefined_voice", ["", "alba"])
def test_local_pocket_label_notes_when_preset_overrides_clones(monkeypatch, voice_count, predefined_voice):
    from tts_audiobook_tool.menus.voice import voice_pocket_menu as menu

    state = make_state("pocket_local", voice_count=voice_count)
    state.project.set_model_setting("pocket_local", "predefined_voice", predefined_voice)
    captured = {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items(state)))
    VoiceMenuShared.menu(state)
    select = captured["items"][-2 if predefined_voice else -1]
    label = get_string_from(state, select.label)
    suffix = f" {menu.COL_DIM}(supercedes voice clone sample)"
    assert label.endswith(suffix) == bool(predefined_voice and voice_count)
    # This label is rebuilt dynamically after clone samples are removed.
    state.project.voice_references = []
    assert "supercedes" not in get_string_from(state, select.label)


@pytest.mark.parametrize("predefined_voice", ["", "alba"])
def test_local_pocket_predefined_voice_follows_samples_with_blank_line(monkeypatch, predefined_voice):
    from tts_audiobook_tool.menus.voice import voice_pocket_menu as menu

    state = make_state("pocket_local", voice_count=2)
    state.project.set_model_setting("pocket_local", "predefined_voice", predefined_voice)
    captured, options, saved = {}, {}, []
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items))
    monkeypatch.setattr(menu.MenuUtil, "options_menu", lambda **kwargs: options.update(kwargs))
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self))
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Predefined voice selection must not load the model"))
    VoiceMenuShared.menu(state)
    items = captured["items"](state)
    select = items[7]
    assert get_string_from(state, items[6].label).startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    label = strip_ansi_codes(get_string_from(state, select.label))
    assert label.startswith("Select predefined voice")
    assert ("alba" in label) == bool(predefined_voice)
    assert select.blank_line_before
    assert items[5].blank_line_before
    assert all(not item.blank_line_before for item in items if item is not select and item is not items[5])
    assert all(not item.superlabel for item in items)
    assert len(items) == (9 if predefined_voice else 8)
    select.handler(state, select)
    assert options["heading_text"] == "Select predefined voice"
    assert options["labels"] == options["values"] == menu.PocketBaseModel.PREDEFINED_VOICES
    assert options["current_value"] == (predefined_voice or None)
    options["on_select"]("alba")
    assert state.project.get_model_setting("pocket_local", "predefined_voice") == "alba"
    clear = captured["items"](state)[-1]
    assert clear.label == "Clear predefined voice"
    assert not clear.blank_line_before
    clear.handler(state, clear)
    assert state.project.get_model_setting("pocket_local", "predefined_voice") == ""
    assert len(captured["items"](state)) == 8
    assert len(state.project.voice_references) == 2
    assert saved == [state.project, state.project]


@pytest.mark.parametrize("instruct", ["", "male, british accent, low pitch"])
def test_local_omnivoice_instructions_follow_samples_with_blank_line(monkeypatch, instruct):
    state = make_state("omnivoice_local", voice_count=2)
    state.project.set_model_setting("omnivoice_local", "instruct", instruct)
    captured, edited, saved = {}, [], []
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items(state)))
    monkeypatch.setattr(ModelMenuShared, "ask_instruct", lambda *args, **kwargs: edited.append((args, kwargs)))
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self))
    VoiceMenuShared.menu(state)
    items = captured["items"]
    design = items[7]
    assert get_string_from(state, items[6].label).startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    assert get_string_from(state, design.label).startswith("Voice design instructions")
    assert design.blank_line_before
    assert items[5].blank_line_before
    assert all(not item.blank_line_before for item in items if item is not design and item is not items[5])
    assert all(not item.superlabel for item in items)
    design.handler(state, design)
    args, kwargs = edited[0]
    assert args[:3] == (state.project, "omnivoice_local", "instruct")
    assert "instructions may have minimal effect" in args[3]
    assert kwargs["validate_omnivoice"] is True
    assert len(items) == (9 if instruct else 8)
    if instruct:
        clear = items[-1]
        assert clear.label == "Clear instructions"
        clear.handler(state, clear)
        assert state.project.get_model_setting("omnivoice_local", "instruct") == ""
        assert saved == [state.project]
        assert len(state.project.voice_references) == 2


def test_indextts2_emotion_controls_keep_secondary_reference_and_setting_targets(monkeypatch):
    from tts_audiobook_tool.menus.voice import voice_indextts2_menu as menu
    from tts_audiobook_tool.project_support.model_settings import SettingRef

    state = make_state("indextts2_local", voice_count=2)
    state.prefs = SimpleNamespace()
    project = state.project
    project.set_model_setting("indextts2_local", "emo_voice", "emotion.flac")
    project.set_model_setting("indextts2_local", "emo_alpha", 0.8)
    captured, selected, saves, numbers = {}, [], [], []
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper",
                        lambda state, items, **kwargs: captured.update(items=items))
    monkeypatch.setattr(menu.hints, "show_hint_if_necessary", lambda *_: None)
    monkeypatch.setattr(VoiceMenuShared, "ask_and_set_voice_file", lambda **kwargs: selected.append(kwargs))
    monkeypatch.setattr(Project, "save", lambda self: saves.append(self))
    monkeypatch.setattr(menu.ask, "ask_number_and_save", lambda *args, **kwargs: numbers.append((args, kwargs)))
    menu.VoiceIndexTts2Menu.menu(state)
    items = captured["items"](state)
    emotion_items = items[-4:]
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in emotion_items]
    assert [item.superlabel for item in emotion_items] == ["Emotion", "", "", ""]
    assert "emotion.flac" in labels[0]
    assert labels[1] == "Clear emotion voice sample"
    assert "0.80" in labels[3]

    emotion_items[0].handler(state, emotion_items[0])
    assert selected[0]["is_secondary"] is True
    assert selected[0]["tts_type"].id == "indextts2_local"
    assert selected[0]["message_override"] == "Enter emotion reference audio clip file path:"

    emotion_items[1].handler(state, emotion_items[1])
    assert project.get_model_setting("indextts2_local", "emo_voice") == ""
    assert len(project.voice_references) == 2
    assert len(captured["items"](state)) == len(items) - 1

    monkeypatch.setattr(menu.ask, "ask_input", lambda _: "0, 0.8, 0, 0, 0.2, 0, 0, 0")
    emotion_items[2].handler(state, emotion_items[2])
    assert project.get_model_setting("indextts2_local", "emo_vector") == [0, 0.8, 0, 0, 0.2, 0, 0, 0]
    monkeypatch.setattr(menu.ask, "ask_input", lambda _: "none")
    emotion_items[2].handler(state, emotion_items[2])
    assert project.get_model_setting("indextts2_local", "emo_vector") == []
    assert saves == [project, project, project]

    emotion_items[3].handler(state, emotion_items[3])
    args, kwargs = numbers[0]
    assert args[0] is project
    assert args[1] == SettingRef("indextts2_local", "emo_alpha")
    assert args[3:6] == (0.01, 1.0, 0.65)
    assert kwargs["is_minus_one_default"] is True


@pytest.mark.parametrize("voice_count", [0, 2])
def test_voice_wrapper_does_not_include_model_settings_note(monkeypatch, voice_count):
    state = make_state("chatterbox_audiocpp", voice_count=voice_count)
    captured = {}
    monkeypatch.setattr(main_menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr("tts_audiobook_tool.menus.voice.voice_menu_shared.AudioMetaUtil.get_audio_duration",
                        lambda _: None)
    VoiceMenuShared.menu_wrapper(state, [], subheading="Sample guidance")
    assert captured["heading"] == "Voice clone"
    assert captured["breadcrumb"] == "Voice"
    assert callable(captured["subheading"])
    rendered = strip_ansi_codes(get_string_from(state, captured["subheading"]))
    if voice_count:
        assert rendered == "- Voice sample 1: voice-0.flac\n- Voice sample 2: voice-1.flac\n\nSample guidance"
    else:
        assert rendered == "Sample guidance"


@pytest.mark.parametrize("next_id", ["higgs_v3_sglomni", "breeze_tts_2_audiocpp"])
def test_remote_voice_factory_tracks_backend_changes(monkeypatch, next_id):
    state = make_state("omnivoice_audiocpp", voice_count=2)
    captured = {}
    monkeypatch.setattr(main_menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu(state)
    assert captured["items"] is VoiceMenuShared.make_remote_items
    state.project.tts_model_type = next_id
    labels = [get_string_from(state, item.label) for item in captured["items"](state)]
    assert len(labels) == (8 if next_id == "breeze_tts_2_audiocpp" else 7)
    assert labels[:5] == [
        voice_menu_shared.LABEL_ADD_VOICE_SAMPLE,
        voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
        voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
    ]
    assert labels[5].startswith(voice_menu_shared.LABEL_VOICE_SELECTION_MODE)
    assert labels[6].startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    if next_id == "breeze_tts_2_audiocpp":
        assert labels[7].startswith("Instructions ")


@pytest.mark.parametrize("remote_mode,expected_opened", [(True, True), (False, False)])
def test_model_entrypoint_allows_unselected_model_only_in_remote_mode(
        monkeypatch, remote_mode, expected_opened):
    # The remote-mode model picker lives inside Model settings, so an
    # unselected model must not block entry there (or it could never be picked).
    state = make_state("none")
    errors, opened = [], []
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: remote_mode)
    monkeypatch.setattr(Tts, "bind_project", lambda project: None)
    monkeypatch.setattr(main_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(ModelMenuShared, "menu", opened.append)
    main_menu.on_model(state, None)
    assert opened == ([state] if expected_opened else [])
    assert errors == ([] if expected_opened else ["Requires TTS model"])
