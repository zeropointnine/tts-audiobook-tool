"""Entry-point and sample-management regressions for the voice/model split."""
from dataclasses import replace
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
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.tts import Tts


LOCAL_MODELS = [
    "breeze_tts_2_local", "chatterbox_local", "dots_local", "fish_s1_local", "fish_s2_local",
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


def sample_group_prefixes(voice_count):
    """Expected (ANSI-stripped) label prefixes of the shared voice list (voices and Add)."""
    prefixes = [f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} {i + 1}: " for i in range(voice_count)]
    return [*prefixes, voice_menu_shared.LABEL_ADD_VOICE_SAMPLE]  # One page (see VOICES_PER_PAGE)


def selection_group_prefixes(voice_count):
    """Expected label prefixes of the selection controls that close every Voices menu."""
    return [voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS,
            voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES]


def capture_voices_items(captured):
    """A menu_wrapper stand-in that keeps the wrapper's own item composition."""
    return lambda state, items, **kwargs: captured.update(
        items=VoiceMenuShared.with_voice_selection_items(items))


def test_main_menu_has_separate_voice_and_model_hotkeys(monkeypatch):
    captured = {}
    monkeypatch.setattr(main_menu.MenuUtil, "menu",
                        lambda state, heading, items, **kwargs: captured.update(items=items(state)))
    main_menu.MainMenu.menu(make_state())
    items = captured["items"]
    voice = next(item for item in items if item.hotkey == "v")
    model = next(item for item in items if item.hotkey == "m")
    assert get_string_from(make_state(), voice.label) == "Voices"
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


# Every model has its own voice menu, so all run at 2 voices. The voice-count rules
# (required label at 0, mode selector at >= 2, no Add at 9) are shared code: run the
# other counts on one model that requires a voice (glm) and one that does not (omnivoice).
@pytest.mark.parametrize("model_id, voice_count", [
    *((model_id, 2) for model_id in LOCAL_MODELS),
    *((model_id, voice_count) for model_id in ("glm_local", "omnivoice_local") for voice_count in (0, 1, 9)),
    ("qwen3tts_local", 0),  # empty slots consult Qwen's checkpoint-specific preset policy
])
def test_all_local_voice_menus_keep_sample_selection_group(
        monkeypatch, model_id, voice_count):
    state = make_state(model_id, voice_count)
    captured = {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", capture_voices_items(captured))
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Sample management must not load the TTS model"))
    VoiceMenuShared.menu(state)
    items = captured["items"](state)
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    # The selection controls always close the menu, after any model-specific items
    selection = selection_group_prefixes(max(voice_count, 1))
    count = len(selection)
    assert all(label.startswith(prefix) for label, prefix in zip(labels[-count:], selection, strict=True))
    items, labels, selection_items = items[:-count], labels[:-count], items[-count:]
    assert selection_items[0].blank_line_before
    assert not any(item.blank_line_before for item in selection_items[1:])
    if model_id == "indextts2_local":
        assert [item.superlabel for item in items[-3:]] == ["Emotion", "", ""]
        emotion_labels = labels[-3:]
        assert emotion_labels[:2] == ["Select emotion voice sample (optional)", "Emotion vector (optional)"]
        assert emotion_labels[2].startswith("Emotion alpha (strength)")
        assert "0.65" in emotion_labels[2] and "default" in emotion_labels[2]
        items, labels = items[:-3], labels[:-3]
    elif model_id == "vibevoice_local":
        assert labels[-1].startswith("Select LoRA")
        assert items[-1].blank_line_before
        items, labels = items[:-1], labels[:-1]
    # No samples still lists the project's one (sample-less) voice
    assert items[0].superlabel == voice_menu_shared.LABEL_VOICE_SAMPLES_SUPERLABEL
    assert all(not item.superlabel for item in items[1:])
    assert not any(label.startswith("Add/remove voice samples") for label in labels)
    assert not any(item.blank_line_before for item in items)
    if voice_count == 0:
        # The requirement replaces "no sample set" when the model requires a voice
        required = VoiceMenuShared.get_voice_sample_required_label(state.project, 0)
        assert labels[0].startswith(f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} 1: {required or 'no sample set'}")
    expected = sample_group_prefixes(max(voice_count, 1))
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
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in captured["items"]]
    assert labels
    assert all(not item.superlabel for item in captured["items"])
    assert not any(label.startswith((voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM,
                                     voice_menu_shared.LABEL_ADD_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
                                     voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
                                     "Add/remove voice samples", "Select voice clone sample",
                                     voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)) for label in labels)
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


def open_voice_submenu(monkeypatch, state, index=0):
    """Open Voice N's submenu from the Voices menu; returns its item maker."""
    captured, submenu = {}, {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", capture_voices_items(captured))
    VoiceMenuShared.menu(state)
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "menu", lambda **kwargs: submenu.update(kwargs))
    voice_item = captured["items"](state)[index]
    voice_item.handler(state, voice_item)
    return lambda: submenu["items"](state)


def item_labels(state, items):
    return [strip_ansi_codes(get_string_from(state, item.label)) for item in items]


@pytest.mark.parametrize("has_sample", [False, True])
@pytest.mark.parametrize("preset", ["", "alba"])
def test_local_pocket_notes_when_voice_preset_overrides_its_sample(monkeypatch, has_sample, preset):
    state = make_state("pocket_local", voice_count=2)
    voices = state.project.voice_references
    first = replace(voices[0], file_name=voices[0].file_name if has_sample else "",
                    presets={"pocket_local": preset} if preset else {})
    state.project.voice_references = [first, voices[1]]
    row = strip_ansi_codes(VoiceMenuShared.make_voice_sample_item_label(
        state.project, state.project.get_tts_model_type(), 0)[0])
    # A preset is simply the voice's value, replacing any (unused) sample details
    if preset:
        assert row == f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} 1: alba"
    else:
        assert "alba" not in row and ("voice-0" in row) == has_sample
    # The other voice is unaffected
    other = strip_ansi_codes(VoiceMenuShared.make_voice_sample_item_label(
        state.project, state.project.get_tts_model_type(), 1)[0])
    assert other.startswith(f"{voice_menu_shared.LABEL_VOICE_SAMPLE_ITEM} 2: voice-1")
    items = open_voice_submenu(monkeypatch, state)()
    sample_item = next(item for item in items if strip_ansi_codes(
        get_string_from(state, item.label)).startswith(voice_menu_shared.LABEL_SET_VOICE_SAMPLE))
    assert (sample_item.sublabel is not None and "takes precedence" in sample_item.sublabel) == bool(
        preset and has_sample)


def test_local_pocket_predefined_voice_is_set_per_voice_without_loading_model(monkeypatch):
    from tts_audiobook_tool.menus.voice import voice_pocket_menu as menu

    state = make_state("pocket_local", voice_count=2)
    options, saved = {}, []
    monkeypatch.setattr(voice_menu_shared.MenuUtil, "options_menu", lambda **kwargs: options.update(kwargs))
    monkeypatch.setattr(Project, "save", lambda self: saved.append(self) or "")
    monkeypatch.setattr(ModelWorker, "inspect_tts_blocking",
                        lambda *_: pytest.fail("Predefined voice selection must not load the model"))
    make_items = open_voice_submenu(monkeypatch, state, index=1)
    labels = item_labels(state, make_items())
    select = next(item for item, label in zip(make_items(), labels) if label.startswith("Predefined voice"))
    assert "none" in strip_ansi_codes(get_string_from(state, select.label))
    assert "Clear predefined voice" not in labels
    select.handler(state, select)
    assert options["labels"] == options["values"] == menu.PocketBaseModel.PREDEFINED_VOICES
    assert options["current_value"] is None
    options["on_select"]("alba")
    assert [voice.presets for voice in state.project.voice_references] == [{}, {"pocket_local": "alba"}]
    # The sample is kept
    assert state.project.voice_references[1].file_name == "voice-1.flac"
    items = make_items()
    labels = item_labels(state, items)
    assert any(label.startswith("Predefined voice") and "alba" in label for label in labels)
    clear = items[labels.index("Clear predefined voice")]
    clear.handler(state, clear)
    assert [voice.presets for voice in state.project.voice_references] == [{}, {}]
    assert "Clear predefined voice" not in item_labels(state, make_items())
    assert saved == [state.project, state.project]


@pytest.mark.parametrize("instruct", ["", "male, british accent, low pitch"])
def test_local_omnivoice_voice_menu_has_no_instructions_items(monkeypatch, instruct):
    state = make_state("omnivoice_local", voice_count=2)
    state.project.voice_references[0] = replace(
        state.project.voice_references[0], instructions={"omnivoice_local": instruct})
    captured = {}
    monkeypatch.setattr(VoiceMenuShared, "menu_wrapper", capture_voices_items(captured))
    VoiceMenuShared.menu(state)
    items = captured["items"](state)
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in items]
    assert labels[-2].startswith(voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS)
    assert labels[-1].startswith(voice_menu_shared.LABEL_AUTO_ASSIGN_CHARACTER_VOICES)
    assert not any(label.startswith(("Voice design", "Clear instructions")) for label in labels)


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
    assert "(currently: emotion)" in labels[0]
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
    _, kwargs = numbers[0]
    assert kwargs["saveable"] is project
    assert kwargs["target"] == SettingRef("indextts2_local", "emo_alpha")
    assert (kwargs["min_value"], kwargs["max_value"], kwargs["default_value"]) == (0.01, 1.0, 0.65)
    assert kwargs["is_minus_one_default"] is True


@pytest.mark.parametrize("voice_count", [0, 2])
def test_voice_wrapper_does_not_include_model_settings_note(monkeypatch, voice_count):
    state = make_state("chatterbox_audiocpp", voice_count=voice_count)
    captured = {}
    monkeypatch.setattr(main_menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu_wrapper(state, [], subheading="Sample guidance")
    assert captured["heading"] == "Voices"
    assert captured["breadcrumb"] == "Voices"
    # The sample list lives in the menu items; the subheading passes through unchanged.
    assert captured["subheading"] == "Sample guidance"
    VoiceMenuShared.menu_wrapper(state, [])
    assert captured["subheading"] is None


@pytest.mark.parametrize("next_id", ["higgs_v3_sglomni", "breeze_tts_2_audiocpp"])
def test_remote_voice_factory_tracks_backend_changes(monkeypatch, next_id):
    state = make_state("omnivoice_audiocpp", voice_count=2)
    captured = {}
    monkeypatch.setattr(main_menu.MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))
    VoiceMenuShared.menu(state)
    # A maker, rebuilt per redraw, so a backend change shows the new model's items
    assert callable(captured["items"])
    state.project.tts_model_type = next_id
    labels = [strip_ansi_codes(get_string_from(state, item.label)) for item in captured["items"](state)]
    assert len(labels) == 5
    expected = sample_group_prefixes(2) + selection_group_prefixes(2)
    assert all(label.startswith(prefix) for label, prefix in zip(labels, expected, strict=True))
    assert labels[2] == voice_menu_shared.LABEL_ADD_VOICE_SAMPLE
    assert not any(label.startswith("Instructions ") for label in labels)


def test_model_entrypoint_allows_unselected_model_only_in_remote_mode(monkeypatch):
    # The remote-mode model picker lives inside Model settings, so an
    # unselected model must not block entry there (or it could never be picked).
    # Local mode blocking is the "none" row of test_entrypoint_prerequisites_do_not_require_clone.
    state = make_state("none")
    errors, opened = [], []
    monkeypatch.setattr(Tts, "is_remote_mode", lambda: True)
    monkeypatch.setattr(Tts, "bind_project", lambda project: None)
    monkeypatch.setattr(main_menu.ask, "ask_error", errors.append)
    monkeypatch.setattr(ModelMenuShared, "menu", opened.append)
    main_menu.on_model(state, None)
    assert opened == [state]
    assert errors == []
