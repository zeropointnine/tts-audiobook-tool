"""Offline contract tests for audio.cpp payload, reference and response handling."""
from __future__ import annotations

import base64
from dataclasses import replace
import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
import time

import httpx
import numpy as np
import pytest
import soundfile as sf

from tts_audiobook_tool.app_support.audio_cpp_util import AudioCppUtil, MAX_REFERENCE_WAV_BYTES
from tts_audiobook_tool.constants import MAX_WORDS_PER_SEGMENT_RECO_RANGE
from tts_audiobook_tool.menus.voice import voice_menu_shared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppBackendAdapter, AudioCppModelSupport, AudioCppSettings
from tts_audiobook_tool.tts_models.audio_cpp_definition import AudioCppTextParameter, load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

MODEL_ID = "chatterbox_audiocpp"
HIGGS_ID = "higgs_v3_audiocpp"
BREEZE_ID = "breeze_tts_2_audiocpp"
ECHO_ID = "echo_tts_audiocpp"
OMNIVOICE_ID = "omnivoice_audiocpp"
FISH_S2_ID = "fish_s2_audiocpp"
DOTS_ID = "dots_audiocpp"
GLM_ID = "glm_tts_audiocpp"
INDEXTTS2_ID = "indextts2_audiocpp"
COSYVOICE3_ID = "cosyvoice3_audiocpp"


class FakeProject:
    language_code = "fr"
    word_substitutions = []
    book = SimpleNamespace(segmentation_settings=SimpleNamespace(max_words_per_segment=20))
    voice_references: list[dict[str, str]] = []

    def __init__(self, values: dict | None = None, model_id: str = MODEL_ID, transcripts: list[str] | None = None):
        self.values = values or {}
        self.model_id = model_id
        self.transcripts = transcripts or []

    def get_model_setting(self, model_id: str, name: str):
        assert model_id == self.model_id
        if name == "transcript":
            return list(self.transcripts)
        return self.values.get(name, -1 if name == "seed" else None)

    def set_model_setting(self, model_id: str, name: str, value, reset: bool = False):
        assert model_id == self.model_id
        if reset:
            self.values.pop(name, None)
        else:
            self.values[name] = value

    def save(self):
        return ""


def definition(model_id: str = MODEL_ID):
    return load_audio_cpp_definitions().models[model_id]


def test_chatterbox_definition_leaves_version_selection_to_server():
    item = definition()
    assert item.family == "chatterbox" and item.tasks == ("clon",) and item.mode == "offline"
    assert item.session_options == {}
    assert set(item.parameters) == {"temperature", "top_p", "repetition_penalty", "guidance_scale", "exaggeration"}
    assert item.parameters["exaggeration"].target == "options" and not item.reference_transcript
    assert item.spec.requires_voice and not item.spec.can_stream
    assert AudioCppSettings.get(FakeProject(), item.parameters["exaggeration"]) == 0.5
    for invalid in (float("nan"), True, 0.1, 2.1):
        with pytest.raises(ValueError):
            AudioCppSettings.set(FakeProject(), item.parameters["exaggeration"], invalid)


def test_higgs_definition_declares_its_own_controls():
    item = definition(HIGGS_ID)
    assert item.family == "higgs_audio_tts" and item.tasks == ("tts",) and item.mode == "offline"
    assert item.session_options == {} and item.reference_transcript
    assert {name: (parameter.type, parameter.default, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "temperature": ("float", 0.8, "top_level"),
        "top_p": ("float", 0.8, "top_level"),
        "top_k": ("int", 30, "top_level"),
    }
    assert not item.spec.requires_voice and not item.voice_required and not item.spec.can_stream
    assert [(control.kind, control.parameter) for control in item.menu] == [
        ("voice_samples", ""), ("parameter", "temperature"), ("parameter", "top_p"),
        ("parameter", "top_k"), ("seed", ""),
    ]
    for invalid in (True, 0, 101, 2.5):
        with pytest.raises(ValueError):
            AudioCppSettings.set(FakeProject(model_id=HIGGS_ID), item.parameters["top_k"], invalid)


def test_fish_s2_definition_uses_native_defaults_and_the_tts_route():
    # Fish only accepts task `tts` (cloning is an optional voice_ref on it), and
    # the controls keep audio.cpp's own defaults rather than local S2's.
    item = definition(FISH_S2_ID)
    assert item.family == "fish_audio" and item.tasks == ("tts",) and item.mode == "offline"
    assert item.reference_transcript and not item.voice_required and not item.spec.requires_voice
    assert item.language_policy == "omit"
    assert item.request_options == {"max_tokens": 2048, "text_chunk_size": 100000}
    assert {name: (parameter.type, parameter.default, parameter.min, parameter.max, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "temperature": ("float", 0.8, 0.01, 1.99, "top_level"),
        "top_p": ("float", 0.8, 0.01, 1.0, "top_level"),
        "top_k": ("int", 30, 1, 100, "top_level"),
    }
    assert item.spec.file_tag == "s2-pro" and item.spec.default_output_sample_rate == 44100
    assert item.spec.ui["short_name"] == "S2-Pro"
    # The server requires temperature < 2.
    with pytest.raises(ValueError):
        AudioCppSettings.set(FakeProject(model_id=FISH_S2_ID), item.parameters["temperature"], 2.0)


def test_glm_definition_requires_the_reference_transcript_and_keeps_upstream_defaults():
    item = definition(GLM_ID)
    # Both advertised GLM-TTS routes share one reference-conditioned path, so
    # the entry claims either task token.
    assert item.family == "glm_tts" and item.tasks == ("tts", "clon") and item.mode == "offline"
    assert item.session_options == {} and item.reference_transcript
    assert {name: (parameter.type, parameter.default, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "temperature": ("float", 1.0, "top_level"),
        "top_p": ("float", 0.8, "top_level"),
        "top_k": ("int", 25, "top_level"),
        "num_inference_steps": ("int", 10, "top_level"),
        "flow_guidance_scale": ("float", 0.7, "options"),
    }
    assert item.spec.requires_voice and not item.spec.can_stream
    # Unlike local glm, the port's output rate is fixed and unconfigurable.
    assert item.spec.default_output_sample_rate == 24000
    # Segment recommendation matches local glm rather than the shared default.
    assert item.max_words_range_reco == (20, 40, "")
    assert AudioCppModelSupport(item).get_max_words_range_reco(SimpleNamespace()) == (20, 40, "")
    # The session never reads a language, so none is sent.
    assert item.language_policy == "omit"
    # top_k accepts audio.cpp's documented 0 ("no limit") as well as the
    # official 25; the other controls keep their declared bounds.
    for parameter, valid, invalid in (
        ("top_k", 0, 101), ("top_k", 25, True),
        ("num_inference_steps", 1, 0), ("flow_guidance_scale", 0.0, 2.1),
    ):
        assert AudioCppSettings.set(FakeProject(model_id=GLM_ID), item.parameters[parameter], valid) == ""
        with pytest.raises(ValueError):
            AudioCppSettings.set(FakeProject(model_id=GLM_ID), item.parameters[parameter], invalid)


def test_echo_definition_is_transcript_free_with_a_tighter_segment_range():
    item = definition(ECHO_ID)
    assert item.family == "echo_tts" and item.tasks == ("clon",) and item.mode == "offline"
    assert item.session_options == {}
    # Echo clones from a reference wav alone; no transcript storage is created.
    assert not item.reference_transcript
    assert item.max_words_range_reco == (20, 50, "")
    assert {name: (parameter.type, parameter.default, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "num_inference_steps": ("int", 40, "top_level"),
        "text_guidance_scale": ("float", 3.0, "options"),
        "speaker_guidance_scale": ("float", 8.0, "options"),
    }
    assert item.spec.requires_voice and not item.spec.can_stream
    assert item.spec.default_output_sample_rate == 44100
    support = AudioCppModelSupport(item)
    assert support.get_max_words_range_reco(SimpleNamespace()) == (20, 50, "")
    # Families without a declaration keep the shared default.
    assert AudioCppModelSupport(definition(BREEZE_ID)).get_max_words_range_reco(SimpleNamespace()) \
        == MAX_WORDS_PER_SEGMENT_RECO_RANGE
    for invalid in (True, 0.5, 101):
        with pytest.raises(ValueError):
            AudioCppSettings.set(FakeProject(model_id=ECHO_ID), item.parameters["num_inference_steps"], invalid)


def test_echo_exceed_warning_uses_its_own_limit():
    project = FakeProject(model_id=ECHO_ID)
    project.book = SimpleNamespace(segmentation_settings=SimpleNamespace(max_words_per_segment=50))
    support = AudioCppModelSupport(definition(ECHO_ID))
    assert support.get_max_words_exceed_warning(project) == ""
    project.book.segmentation_settings.max_words_per_segment = 60
    warning = support.get_max_words_exceed_warning(project)
    assert "Echo-TTS" in warning and "(60)" in warning and "(50)" in warning


def test_omnivoice_definition_claims_the_clone_route_with_family_defaults():
    item = definition(OMNIVOICE_ID)
    # One `tts` route serves auto voice, clone and voice design; this entry drives
    # all three, matching local `omnivoice_local`.
    assert item.family == "omnivoice" and item.tasks == ("tts",) and item.mode == "offline"
    assert item.session_options == {}
    # Optional in the family, but declared so a reference transcript is stored
    # and sent, matching local `omnivoice_local`. OmniVoice's own clone path
    # rejects a reference WAV without one.
    assert item.reference_transcript
    assert {name: (parameter.type, parameter.default, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "num_inference_steps": ("int", 32, "top_level"),
        "guidance_scale": ("float", 2.0, "top_level"),
        # audio.cpp reads these only from the request's `options` map.
        "speed": ("float", 1.0, "options"),
        "instruct": ("str", "", "options"),
    }
    assert item.parameters["instruct"].request_key == "instruction"
    # Local OmniVoice hardcodes the chunker suppression; here it is pinned, so
    # it is neither a user control nor persisted project storage.
    assert item.request_options == {"audio_chunk_threshold": 999.0}
    assert "audio_chunk_threshold" not in {setting["name"] for setting in item.settings}
    # Voice design and auto voice are reference-less, so this entry does not
    # require a voice sample (like Breeze).
    assert not item.spec.requires_voice and not item.voice_required and not item.spec.can_stream
    assert item.spec.default_output_sample_rate == 24000 and item.spec.un_all_caps
    # Instructions belong to Voice clone; numeric settings and seed stay in Model.
    assert [(control.kind, control.target_menu, control.parameter) for control in item.menu] == [
        ("voice_samples", None, ""),
        ("voice_instructions", "voice", "instruct"),
        ("parameter", "model", "num_inference_steps"),
        ("parameter", "model", "speed"),
        ("parameter", "model", "guidance_scale"),
        ("seed", "model", ""),
    ]
    project = FakeProject(model_id=OMNIVOICE_ID)
    for parameter, valid, invalid in (
        ("num_inference_steps", 8, 7), ("num_inference_steps", 64, 65),
        ("guidance_scale", 0.0, 4.1), ("speed", 1.5, 2.1),
    ):
        assert AudioCppSettings.set(project, item.parameters[parameter], valid) == ""
        with pytest.raises(ValueError):
            AudioCppSettings.set(project, item.parameters[parameter], invalid)


def test_omnivoice_numeric_menu_prompt_shows_the_default(monkeypatch, tmp_path: Path):
    """Numeric prompts follow the app convention: range plus default value."""
    from tts_audiobook_tool import text_util
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu

    item = definition(OMNIVOICE_ID)
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": OMNIVOICE_ID})
    state = SimpleNamespace(project=project)
    printed: list[str] = []
    monkeypatch.setattr("tts_audiobook_tool.ask.printt",
                        lambda text: printed.append(text))
    # An empty response leaves storage untouched, so nothing needs saving.
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_audio_cpp_menu.ask.ask_input",
                        lambda **_: "")

    for control in item.menu:
        if control.kind != "parameter":
            continue
        parameter = item.parameters[control.parameter]
        ModelAudioCppMenu.make_parameter_item(state, parameter, control.label).handler(state, None)
        minimum = int(parameter.min) if parameter.type == "int" else parameter.min
        maximum = int(parameter.max) if parameter.type == "int" else parameter.max
        default = int(parameter.default) if parameter.type == "int" else parameter.default
        assert text_util.strip_ansi_codes(printed[-1]) == (
            f"Enter {control.label}: (valid range: {minimum}-{maximum}; default: {default})"
        )


def test_omnivoice_menu_renders_settings_in_local_order_without_headings(monkeypatch):
    """Model settings match local OmniVoice's control order without headings."""
    from tts_audiobook_tool import text_util
    from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    monkeypatch.setattr(ModelMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("Seed (currently: random)", lambda *_: None))
    monkeypatch.setattr(Project, "save", lambda self: "")
    project = Project.model_validate({"dir_path": "", "tts_model_type": OMNIVOICE_ID})
    state = SimpleNamespace(project=project)

    items = ModelAudioCppMenu.make_items(state, definition(OMNIVOICE_ID))
    assert all(not item.superlabel for item in items)
    assert [text_util.strip_ansi_codes(get_string_from(state, item.label)) for item in items] == [
        "Steps (currently: 32 default)",
        "Speed (currently: 1.00 default)",
        "Guidance scale (currently: 2.00 default)",
        "Seed (currently: random)",
    ]


def test_echo_menus_partition_voice_selections_and_settings_without_headings(monkeypatch, tmp_path: Path):
    from tts_audiobook_tool import text_util
    from tts_audiobook_tool.app_types import VoiceSelectMode
    from tts_audiobook_tool.menus.menu_util import get_string_from
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
    from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu

    monkeypatch.setattr(Project, "save", lambda self: "")
    project = Project.model_validate({"dir_path": str(tmp_path), "tts_model_type": ECHO_ID})
    project.set_model_setting(ECHO_ID, "file_name", ["suzie yeung hanya 1_4.flac", "second.flac"])
    project.voice_select_mode = VoiceSelectMode.USER_DEFINED
    state = SimpleNamespace(project=project)

    items = VoiceAudioCppMenu.make_items(state, definition(ECHO_ID))

    assert [text_util.strip_ansi_codes(get_string_from(state, item.label)) for item in items] == [
        voice_menu_shared.LABEL_ADD_VOICE_SAMPLE,
        voice_menu_shared.LABEL_REMOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_MOVE_VOICE_SAMPLE,
        voice_menu_shared.LABEL_CROP_VOICE_SAMPLE,
        voice_menu_shared.LABEL_EDIT_VOICE_TRANSCRIPTION,
        f"{voice_menu_shared.LABEL_VOICE_SELECTION_MODE} (currently: user-defined)",
        voice_menu_shared.LABEL_EDIT_VOICE_SELECTIONS,
    ]
    assert all(not item.superlabel for item in items)

    items = ModelAudioCppMenu.make_items(state, definition(ECHO_ID))
    assert [text_util.strip_ansi_codes(get_string_from(state, item.label)) for item in items] == [
        "Steps (currently: 40 default)",
        "Text guidance scale (currently: 3.00 default)",
        "Speaker guidance scale (currently: 8.00 default)",
        "Seed (currently: random)",
    ]
    assert all(not item.superlabel for item in items)


@pytest.mark.parametrize("model_id", [MODEL_ID, HIGGS_ID, BREEZE_ID, ECHO_ID, OMNIVOICE_ID])
def test_audio_cpp_seed_prompt_does_not_warn_about_unsupported_batch_mode(monkeypatch, model_id):
    from tts_audiobook_tool import text_util
    from tts_audiobook_tool.menus.menu_util import get_string_from
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    prompts = []
    monkeypatch.setattr("tts_audiobook_tool.menus.model.model_menu_shared.ask.ask_number_and_save",
                        lambda **kwargs: prompts.append(text_util.strip_ansi_codes(kwargs["prompt"])))
    state = SimpleNamespace(project=Project(tts_model_type=model_id))
    items = ModelAudioCppMenu.make_items(state, definition(model_id))
    seed_item = next(item for item in items
                     if text_util.strip_ansi_codes(get_string_from(state, item.label)) == "Seed (currently: random)")

    seed_item.handler(state, seed_item)

    assert prompts == ["Enter a static seed value (or -1 for random): "]


def test_breeze_definition_declares_free_form_instructions():
    item = definition(BREEZE_ID)
    assert item.family == "breeze_tts" and item.tasks == ("tts", "clon") and item.mode == "offline"
    assert item.session_options == {} and item.reference_transcript
    assert {name: (parameter.type, parameter.default, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "temperature": ("float", 0.9, "top_level"),
        "top_p": ("float", 1.0, "top_level"),
        "top_k": ("int", 50, "top_level"),
        "guidance_scale": ("float", 1.0, "top_level"),
        "instruct": ("str", "", "options"),
    }
    assert item.parameters["instruct"].request_key == "instruction"
    assert AudioCppSettings.get(FakeProject(model_id=BREEZE_ID), item.parameters["instruct"]) == ""
    with pytest.raises(ValueError, match="must be a string"):
        item.parameters["instruct"].validate(123)
    assert not item.spec.requires_voice and not item.voice_required and not item.spec.can_stream
    assert item.spec.default_output_sample_rate == 24000
    for invalid in (True, 50.5, 101):
        with pytest.raises(ValueError):
            AudioCppSettings.set(FakeProject(model_id=BREEZE_ID), item.parameters["top_k"], invalid)


@pytest.mark.parametrize("model_id, label, initial, edited", [
    (BREEZE_ID, "Instructions", "Read this as a quiet bedtime story.",
     "Speak Warmly and Naturally, with calm pacing."),
    (OMNIVOICE_ID, "Voice design instructions", "female, high pitch", "male, low pitch"),
])
@pytest.mark.parametrize("target_menu", ["model", "voice"])
def test_instruction_menu_prefills_saves_and_clears_with_model_specific_validation(
        monkeypatch, model_id, label, initial, edited, target_menu):
    from tts_audiobook_tool import ask, text_util
    from tts_audiobook_tool.menus.menu_util import MenuItem, get_string_from
    from tts_audiobook_tool.menus.voice import voice_instruct_util
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    monkeypatch.setattr(ModelMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("Seed", lambda *_: None))
    monkeypatch.setattr(ask, "_clear_input_buffer", lambda: None)
    saves = []
    monkeypatch.setattr(Project, "save", lambda self: saves.append(True) or "")
    validation_calls = []

    def validate(value):
        assert model_id == OMNIVOICE_ID, "Breeze must not use OmniVoice's validator"
        validation_calls.append(value)
        return "", value

    monkeypatch.setattr(voice_instruct_util, "validate_instruct", validate)
    prefills = []

    def advanced_input(message, prefill):
        prefills.append(prefill)
        return edited

    monkeypatch.setattr(ask.AskAdvanced, "ask", advanced_input)
    project = Project.model_validate({"dir_path": "", "tts_model_type": model_id})
    state = SimpleNamespace(project=project)
    from dataclasses import replace
    from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu

    item = definition(model_id)
    item = replace(item, menu=tuple(replace(control, target_menu=target_menu)
                                  if control.kind == "voice_instructions" else control
                                  for control in item.menu))
    builder = ModelAudioCppMenu if target_menu == "model" else VoiceAudioCppMenu
    other_builder = VoiceAudioCppMenu if target_menu == "model" else ModelAudioCppMenu
    # Isolate samples so Edit and Clear must stay adjacent in the destination.
    item = replace(item, menu=tuple(control for control in item.menu if control.kind != "voice_samples"))
    menus = builder.make_items(state, item)
    assert text_util.strip_ansi_codes(get_string_from(state, menus[0].label)) == f"{label} (optional)"
    assert all(not menu.superlabel for menu in menus)

    project.set_model_setting(model_id, "instruct", initial)
    menus = builder.make_items(state, item)
    assert menus[1].label == "Clear instructions" and not menus[1].superlabel
    assert all(not menu.superlabel for menu in menus)
    other_labels = [text_util.strip_ansi_codes(get_string_from(state, menu.label))
                    for menu in other_builder.make_items(state, item)]
    assert not any(value.startswith(label) or value == "Clear instructions" for value in other_labels)
    menus[0].handler(state, menus[0])
    assert prefills == [initial]
    assert project.get_model_setting(model_id, "instruct") == edited
    assert validation_calls == ([edited] if model_id == OMNIVOICE_ID else [])
    assert len(saves) == 1

    # The setting also survives project serialization.
    restored = Project.model_validate(project.model_dump(exclude={"reason_pauses"}))
    assert restored.get_model_setting(model_id, "instruct") == edited
    menus[1].handler(state, menus[1])
    assert project.get_model_setting(model_id, "instruct") == ""
    assert len(saves) == 2
    menus = builder.make_items(state, item)
    assert all(menu.label != "Clear instructions" for menu in menus)


def test_chatterbox_settings_menu_displays_catalog_note(monkeypatch):
    from tts_audiobook_tool.menus.menu_util import MenuUtil, get_string_from
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    item = definition()
    state = SimpleNamespace(project=Project(tts_model_type=MODEL_ID))
    captured = {}
    monkeypatch.setattr(MenuUtil, "menu", lambda **kwargs: captured.update(kwargs))

    ModelMenuShared.menu(state)

    assert captured["heading"] == "Model settings"
    assert get_string_from(state, captured["subheading"]) == item.spec.ui["settings_note"].strip()
    assert "To select Multilingual V3, use audio.cpp setting:" in item.spec.ui["settings_note"]


def test_chatterbox_menu_preserves_control_order_without_headings(monkeypatch):
    """The catalog settings menu renders controls without group headings."""
    from tts_audiobook_tool.menus.menu_util import MenuItem
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    monkeypatch.setattr(ModelMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("Seed", lambda *_: None))
    monkeypatch.setattr(ModelAudioCppMenu, "make_parameter_item",
                        lambda _state, _parameter, label, **_: MenuItem(label, lambda *_: None))

    # The menu reads resolved settings to ask the model's behavior which
    # controls are visible, so the state needs a project.
    items = ModelAudioCppMenu.make_items(SimpleNamespace(project=Project(tts_model_type=MODEL_ID)), definition())

    assert [item.label for item in items] == [
        "Temperature", "Exaggeration", "CFG/pace",
        "Top-P", "Repetition penalty", "Seed",
    ]
    assert all(not item.superlabel for item in items)



def test_chatterbox_guidance_menu_label_uses_cfg_pace():
    from tts_audiobook_tool import text_util
    from tts_audiobook_tool.menus.menu_util import get_string_from
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu

    item = definition()
    control = next(control for control in item.menu if control.parameter == "guidance_scale")
    state = SimpleNamespace(project=Project(tts_model_type=MODEL_ID))
    menu_item = ModelAudioCppMenu.make_parameter_item(
        state, item.parameters[control.parameter], control.label)

    assert text_util.strip_ansi_codes(get_string_from(state, menu_item.label)) == (
        "CFG/pace (currently: 0.50 default)")


def test_stereo_flac_reference_becomes_mono_pcm16_wav(tmp_path: Path):
    path = tmp_path / "reference.flac"
    sf.write(path, np.stack([np.full(500, .25), np.full(500, -.25)], axis=1), 24000, format="FLAC")
    uri = AudioCppUtil.make_voice_ref(str(path))
    assert uri.startswith("data:audio/wav;base64,")
    encoded = base64.b64decode(uri.split(",", 1)[1])
    assert len(encoded) <= MAX_REFERENCE_WAV_BYTES
    with sf.SoundFile(BytesIO(encoded)) as wav:
        assert wav.format == "WAV" and wav.subtype == "PCM_16" and wav.channels == 1 and wav.samplerate == 24000
        assert np.max(np.abs(wav.read(dtype="float32"))) < .001


def test_decoded_wav_sound_and_non_wav_rejection():
    wav = BytesIO()
    sf.write(wav, np.full(60, .25), 22050, format="WAV", subtype="PCM_16")
    sound = AudioCppUtil.sound_from_wav(wav.getvalue())
    assert sound.sr == 22050 and len(sound.data) == 60
    flac = BytesIO()
    sf.write(flac, np.full(60, .25), 22050, format="FLAC")
    with pytest.raises(ValueError, match="not WAV"):
        AudioCppUtil.sound_from_wav(flac.getvalue())


def test_corrupt_voice_is_a_clear_error(tmp_path: Path):
    path = tmp_path / "broken.flac"
    path.write_bytes(b"not a sound")
    with pytest.raises(ValueError, match="could not be decoded"):
        AudioCppUtil.make_voice_ref(str(path))


def test_reference_decoded_wav_limit(tmp_path: Path):
    # Small compressed FLAC can expand past the encoded 5 MiB WAV cap.
    path = tmp_path / "large.flac"
    sf.write(path, np.zeros((MAX_REFERENCE_WAV_BYTES // 2 + 64,), dtype=np.float32), 24000, format="FLAC")
    with pytest.raises(ValueError, match="5 MiB"):
        AudioCppUtil.make_voice_ref(str(path))


def test_exact_model_id_options_seed_language_order_and_no_transcript(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition()
    support = AudioCppModelSupport(item)
    adapter = AudioCppBackendAdapter(item, support, "custom-server-name-not-a-model-family")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("sample.flac", ""))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 4294967295)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([len(captured) - 1], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    sounds = adapter.generate_using_project(FakeProject(), ["first", "second"])
    assert not isinstance(sounds, str)
    assert [sound.data[0] for sound in sounds] == [0, 1]
    assert [payload["input"] for payload in captured] == ["first", "second"]
    for payload in captured:
        assert payload["model"] == "custom-server-name-not-a-model-family"
        assert payload["temperature"] == .8 and payload["top_p"] == .95
        assert payload["repetition_penalty"] == 1.2 and payload["guidance_scale"] == .5
        assert payload["options"] == {"exaggeration": .5, "text_chunk_size": 100000}
        assert payload["seed"] == 4294967295 and payload["language"] == "fr"
        assert payload["response_format"] == "wav"
        assert payload["voice_ref"]["type"] == "base64"
        assert payload["voice_ref"]["data"].startswith("data:audio/wav;base64,")
        assert "reference_text" not in payload and "stream" not in payload and "references" not in payload
    assert adapter.generate_using_project(FakeProject(), ["x"], on_stream_end=lambda: None) \
        == f"{item.spec.ui['proper_name']} does not support streaming"
    assert len(captured) == 2


def test_higgs_payload_carries_top_level_controls_and_reference_transcript(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(HIGGS_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "higgs-audio-tts")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    result = adapter.generate_using_project(FakeProject(model_id=HIGGS_ID, values={"top_k": 12}), ["hello"])
    assert not isinstance(result, str)
    assert len(captured) == 1
    payload = captured[0]
    assert payload["voice_ref"]["type"] == "base64"
    assert payload["voice_ref"]["data"].startswith("data:audio/wav;base64,")
    assert {key: value for key, value in payload.items() if key != "voice_ref"} == {
        "model": "higgs-audio-tts", "input": "hello", "response_format": "wav",
        "seed": 7, "language": "fr", "temperature": 0.8, "top_p": 0.8, "top_k": 12,
        "reference_text": "the exact words spoken",
        "options": {"text_chunk_size": 100000},
    }
    assert adapter.generate_using_project(FakeProject(model_id=HIGGS_ID), ["x"], on_stream_end=lambda: None) \
        == "Higgs Audio V3 does not support streaming"


def test_higgs_generates_without_a_reference_sample(monkeypatch):
    item = definition(HIGGS_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "higgs-audio-tts")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("", ""))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path",
                        lambda *args: pytest.fail("reference file accessed"))
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=HIGGS_ID, values={"seed": 7})
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    assert captured == [{
        "model": "higgs-audio-tts", "input": "hello", "response_format": "wav",
        "seed": 7, "language": "fr", "temperature": 0.8, "top_p": 0.8, "top_k": 30,
        "options": {"text_chunk_size": 100000},
    }]


@pytest.mark.parametrize("instructions", [None, "", "Speak warmly and naturally, with calm pacing."])
def test_breeze_payload_sends_instructions_and_the_reference_transcript(monkeypatch, tmp_path: Path, instructions):
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(BREEZE_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "breeze-clone")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=BREEZE_ID, values={"top_k": 0})
    if instructions is not None:
        project.values["instruct"] = instructions
    project.language_code = "en"
    result = adapter.generate_using_project(project, ["hello"])
    assert not isinstance(result, str)
    assert len(captured) == 1
    payload = captured[0]
    assert payload["voice_ref"]["type"] == "base64"
    assert payload["voice_ref"]["data"].startswith("data:audio/wav;base64,")
    # Free-form instructions ride in options alongside the pinned request options.
    # Missing/cleared instructions leave the model's own default intact.
    expected = {
        "model": "breeze-clone", "input": "hello", "response_format": "wav",
        "seed": 7, "language": "en", "temperature": 0.9, "top_p": 1.0, "top_k": 0,
        "guidance_scale": 1.0, "reference_text": "the exact words spoken",
        "options": {"text_chunk_size": 100000},
    }
    if instructions:
        expected["options"] = {"instruction": instructions, "text_chunk_size": 100000}
    assert {key: value for key, value in payload.items() if key != "voice_ref"} == expected
    assert adapter.generate_using_project(project, ["x"], on_stream_end=lambda: None) \
        == "Breeze TTS 2 does not support streaming"


@pytest.mark.parametrize("instructions", [None, "", "A warm narrator with calm pacing."])
def test_breeze_generates_without_a_voice_sample(monkeypatch, instructions):
    item = definition(BREEZE_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "breeze-tts")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("", ""))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path",
                        lambda *args: pytest.fail("reference file lookup without a sample"))
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    values = {"seed": 7}
    if instructions is not None:
        values["instruct"] = instructions
    project = FakeProject(model_id=BREEZE_ID, values=values)
    result = adapter.generate_using_project(project, ["hello"])
    assert not isinstance(result, str) and len(result) == 1
    payload = captured[0]
    assert payload["model"] == "breeze-tts" and payload["input"] == "hello"
    assert "voice_ref" not in payload and "reference_text" not in payload
    if instructions:
        assert payload["options"] == {"instruction": instructions, "text_chunk_size": 100000}
    else:
        assert payload["options"] == {"text_chunk_size": 100000}


@pytest.mark.parametrize("saved_parameters", [
    {"num_inference_steps": 32},
    {"num_inference_steps": 32, "truncation_factor": 1.0},
])
def test_echo_payload_sends_steps_top_level_and_cfg_controls_in_options(
        monkeypatch, tmp_path: Path, saved_parameters):
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(ECHO_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "echo-tts")
    # Echo has no transcript storage, so the voice pair carries an empty one.
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("sample.flac", ""))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=44100)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = Project.model_validate({
        "tts_model_type": ECHO_ID,
        "dir_path": str(tmp_path),
        "language_code": "en",
        "model_settings": {"models": {ECHO_ID: {"parameters": saved_parameters}}},
    })
    # Old saved overrides are retired too, rather than silently remaining active.
    assert "truncation_factor" not in project.model_settings.models[ECHO_ID]["parameters"]
    result = adapter.generate_using_project(project, ["hello"])
    assert not isinstance(result, str)
    assert len(captured) == 1
    payload = captured[0]
    assert payload["voice_ref"]["type"] == "base64"
    # audio.cpp forwards num_inference_steps from the body; the two guidance
    # scales are only readable inside `options`. Truncation is left to the server.
    assert {key: value for key, value in payload.items() if key != "voice_ref"} == {
        "model": "echo-tts", "input": "hello", "response_format": "wav", "seed": 7,
        "language": "en", "num_inference_steps": 32,
        "options": {"text_guidance_scale": 3.0, "speaker_guidance_scale": 8.0},
    }
    assert adapter.generate_using_project(project, ["x"], on_stream_end=lambda: None) \
        == "Echo-TTS does not support streaming"


@pytest.mark.parametrize("stored_seed, force_random, caller_cap, expected_seed, expected_stop", [
    (-1, False, -1, 2147483647, 2147483648),
    (2**32 - 1, True, -1, 2147483647, 2147483648),
    (-1, False, 7, 7, 8),
    (-1, False, 0, 0, 1),
    (-1, False, 2**32 - 1, 2147483647, 2147483648),
    (2**32 - 1, False, 7, 2**32 - 1, None),
])
def test_echo_random_seed_cap_reaches_payload_through_dispatch(
        monkeypatch, stored_seed, force_random, caller_cap, expected_seed, expected_stop):
    from tts_audiobook_tool.tts import Tts

    item = definition(ECHO_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "echo-tts")
    project = Project(tts_model_type=ECHO_ID)
    project.set_model_setting(ECHO_ID, "seed", stored_seed)
    monkeypatch.setattr(Tts, "_type", TtsModelType.require_by_id(ECHO_ID))
    monkeypatch.setattr(Tts, "get_instance", staticmethod(lambda: adapter))
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *_: ("sample.flac", ""))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *_: "sample.flac")
    monkeypatch.setattr(AudioCppUtil, "make_voice_ref", lambda _: "data:audio/wav;base64,mocked")
    draws = []

    def draw(stop):
        draws.append(stop)
        return stop - 1  # prove that the inclusive maximum itself is usable

    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", draw)
    captured = []

    def generate(_url, payload, **_kwargs):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=44100)

    monkeypatch.setattr(AudioCppUtil, "generate", generate)
    result = Tts.generate_using_project(
        project, ["hello", "second"], force_random_seed=force_random, max_random_seed=caller_cap)

    assert not isinstance(result, str)
    assert draws == ([] if expected_stop is None else [expected_stop])
    assert [payload["seed"] for payload in captured] == [expected_seed, expected_seed]
    assert all(type(payload["seed"]) is int for payload in captured)
    assert all("max_random_seed" not in payload and "max_random_seed" not in payload["options"]
               for payload in captured)
    assert project.get_model_setting(ECHO_ID, "seed") == stored_seed


def test_omnivoice_payload_sends_steps_top_level_and_request_options_in_options(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(OMNIVOICE_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "omnivoice")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=OMNIVOICE_ID, values={
        "num_inference_steps": 24, "speed": 1.25})
    project.language_code = "en"
    result = adapter.generate_using_project(project, ["hello"])
    assert not isinstance(result, str)
    assert len(captured) == 1
    payload = captured[0]
    assert payload["voice_ref"]["type"] == "base64"
    # audio.cpp forwards num_inference_steps/guidance_scale from the body; the
    # request options (`speed`, plus the pinned chunk threshold) are only
    # readable inside `options`. The transcript rides as `reference_text`.
    assert {key: value for key, value in payload.items() if key != "voice_ref"} == {
        "model": "omnivoice", "input": "hello", "response_format": "wav", "seed": 7,
        "language": "en", "num_inference_steps": 24, "guidance_scale": 2.0,
        "options": {"audio_chunk_threshold": 999.0, "speed": 1.25},
        "reference_text": "the exact words spoken",
    }
    assert adapter.generate_using_project(project, ["x"], on_stream_end=lambda: None) \
        == "OmniVoice does not support streaming"


def test_omnivoice_sends_instruction_and_omits_voice_ref_without_a_sample(monkeypatch, tmp_path: Path):
    """Voice design / auto voice: no reference WAV, instruction in `options`."""
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(OMNIVOICE_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "omnivoice")
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    # An instruction with no voice reference is the voice-design route.
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("", ""))
    project = FakeProject(model_id=OMNIVOICE_ID, values={"instruct": "female, young adult, high pitch"})
    project.language_code = "en"
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    design = captured[0]
    assert "voice_ref" not in design and "reference_text" not in design
    assert design["options"] == {
        "audio_chunk_threshold": 999.0, "speed": 1.0, "instruction": "female, young adult, high pitch",
    }

    # Neither a voice nor an instruction is the reference-less auto voice route,
    # and a cleared instruction must not be sent as an empty one.
    project.values.pop("instruct")
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    auto = captured[1]
    assert "voice_ref" not in auto
    assert auto["options"] == {"audio_chunk_threshold": 999.0, "speed": 1.0}

    # With a reference the same instruction rides along as a style hint.
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    project.values["instruct"] = "female, young adult, high pitch"
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    clone = captured[2]
    assert clone["voice_ref"]["type"] == "base64"
    assert clone["reference_text"] == "the exact words spoken"
    assert clone["options"]["instruction"] == "female, young adult, high pitch"


def test_fish_s2_payload_omits_language_and_pins_token_and_chunk_options(monkeypatch, tmp_path: Path):
    # Fish auto-detects language, so no language field is sent even though the
    # project has one; the reference transcript and top-level controls ride along.
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(FISH_S2_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "fish-audio-s2-pro")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=44100)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=FISH_S2_ID, values={"seed": 7, "top_k": 100})
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    payload = captured[0]
    assert payload["voice_ref"]["data"].startswith("data:audio/wav;base64,")
    assert {key: value for key, value in payload.items() if key != "voice_ref"} == {
        "model": "fish-audio-s2-pro", "input": "hello", "response_format": "wav",
        "seed": 7, "temperature": 0.8, "top_p": 0.8, "top_k": 100,
        "reference_text": "the exact words spoken",
        "options": {"max_tokens": 2048, "text_chunk_size": 100000},
    }


def test_dots_definition_defaults_to_soar_and_accepts_both_clone_task_spellings():
    # audio.cpp cannot say whether a dots entry serves SOAR or MeanFlow, so the
    # defaults are SOAR's and the operator is on the hook; both task tokens match.
    item = definition(DOTS_ID)
    assert item.family == "dots_tts" and item.tasks == ("tts", "clon") and item.mode == "offline"
    assert item.session_options == {}
    assert item.reference_transcript and not item.voice_required and not item.spec.requires_voice
    assert item.language_policy == "normalized" and item.request_options == {"max_tokens": 500, "text_chunk_size": 100000}
    assert {name: (parameter.type, parameter.default, parameter.min, parameter.max, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "num_inference_steps": ("int", 10, 1, 32, "top_level"),
        "guidance_scale": ("float", 1.2, 1.0, 3.0, "top_level"),
        "speaker_scale": ("float", 1.5, 0.0, 3.0, "options"),
    }
    assert "MeanFlow is 4" in item.parameters["num_inference_steps"].input_prompt_suffix
    assert [(control.kind, control.parameter) for control in item.menu] == [
        ("voice_samples", ""), ("parameter", "num_inference_steps"), ("parameter", "guidance_scale"),
        ("parameter", "speaker_scale"), ("seed", ""),
    ]
    assert item.spec.default_output_sample_rate == 48000 and item.spec.file_tag == "dots"
    # Untouched project: random seed per request, like every other audio.cpp entry.
    assert [s["default"] for s in item.settings if s["name"] == "seed"] == [-1]


@pytest.mark.parametrize("project_language, expected", [
    ("en", "en"), ("en-US", "en"), ("English", "en"), ("zh", "zh"), ("", None),
])
def test_dots_payload_places_controls_and_normalizes_language(monkeypatch, tmp_path: Path, project_language, expected):
    # speaker_scale is not a server-forwarded top-level field, so it rides in
    # `options` with the pinned token budget; the language becomes a base code
    # (or is omitted) so the server does not build a bad "[EN-US]" tag.
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(DOTS_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "my-dots")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=48000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=DOTS_ID, values={"seed": 7, "num_inference_steps": 4})
    project.language_code = project_language
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    payload = captured[0]
    assert payload.pop("voice_ref")["data"].startswith("data:audio/wav;base64,")
    expected_payload = {
        "model": "my-dots", "input": "hello", "response_format": "wav", "seed": 7,
        "num_inference_steps": 4, "guidance_scale": 1.2,
        "reference_text": "the exact words spoken",
        "options": {"max_tokens": 500, "text_chunk_size": 100000, "speaker_scale": 1.5},
    }
    if expected:
        expected_payload["language"] = expected
    assert payload == expected_payload


def test_indextts2_definition_serves_either_variant_with_local_defaults():
    # One entry covers IndexTTS2 and IndexTTS2.5: audio.cpp serves both as
    # family `index_tts2` and does not advertise which variant is loaded.
    item = definition(INDEXTTS2_ID)
    assert item.family == "index_tts2" and item.tasks == ("tts", "clon") and item.mode == "offline"
    assert item.session_options == {} and item.request_options == {}
    assert item.voice_required and item.spec.requires_voice and not item.reference_transcript
    assert item.language_policy == "normalized" and item.language_target == "options"
    assert item.max_words_range_reco == (40, 60, "")
    assert {name: (parameter.type, parameter.default, parameter.min, parameter.max, parameter.target)
            for name, parameter in item.parameters.items()} == {
        "temperature": ("float", 0.8, 0.01, 2.0, "top_level"),
        "top_p": ("float", 0.8, 0.01, 1.0, "top_level"),
        "top_k": ("int", 30, 1, 100, "top_level"),
        "duration_factor": ("float", 1.0, 0.5, 2.0, "options"),
    }
    assert [(control.kind, control.parameter) for control in item.menu] == [
        ("voice_samples", ""), ("parameter", "temperature"), ("parameter", "top_p"),
        ("parameter", "top_k"), ("parameter", "duration_factor"), ("seed", ""),
    ]
    assert item.spec.default_output_sample_rate == 22050 and item.spec.file_tag == "indextts2"


@pytest.mark.parametrize("project_language, expected", [
    ("es", "es"), ("ja-JP", "ja"), ("English", "en"), ("", None),
])
def test_indextts2_payload_sends_language_and_duration_in_options(
        monkeypatch, tmp_path: Path, project_language, expected):
    # IndexTTS2.5 auto-detects only zh-vs-en, so the base language code rides in
    # `options` (where the session reads it) beside duration_factor; no
    # transcript is sent and no text_chunk_size is pinned.
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(INDEXTTS2_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "index-tts2.5")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "ignored transcript"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=22050)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=INDEXTTS2_ID, values={"seed": 7, "duration_factor": 1.2})
    project.language_code = project_language
    assert not isinstance(adapter.generate_using_project(project, ["hola"]), str)
    payload = captured[0]
    assert payload.pop("voice_ref")["data"].startswith("data:audio/wav;base64,")
    options: dict = {"duration_factor": 1.2}
    if expected:
        options["language"] = expected
    assert payload == {
        "model": "index-tts2.5", "input": "hola", "response_format": "wav", "seed": 7,
        "temperature": 0.8, "top_p": 0.8, "top_k": 30, "options": options,
    }


def test_indextts2_requires_a_voice_sample_before_network(monkeypatch):
    item = definition(INDEXTTS2_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "index-tts2")
    monkeypatch.setattr(AudioCppUtil, "generate", lambda *args, **kwargs: pytest.fail("network attempted"))
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("", ""))
    assert adapter.generate_using_project(FakeProject(model_id=INDEXTTS2_ID), ["hello"]) == (
        "A voice clone sample is required"
    )


@pytest.mark.parametrize("project_language, expected", [
    ("es-MX", "Passing language hint to model: es (applies to IndexTTS2.5 only)"),
    ("", "Passing language hint to model: auto (applies to IndexTTS2.5 only)"),
])
def test_indextts2_preflight_reports_language_hint_with_variant_note(project_language, expected):
    # The warning shows the code actually sent and qualifies it, since IndexTTS2
    # ignores the hint and the server does not reveal which variant is loaded.
    item = definition(INDEXTTS2_ID)
    project = FakeProject(model_id=INDEXTTS2_ID)
    project.language_code = project_language
    assert expected in AudioCppModelSupport(item).get_warning_issues(project)


@pytest.mark.parametrize("model_id", [BREEZE_ID, OMNIVOICE_ID, FISH_S2_ID, DOTS_ID])
def test_optional_voice_requires_a_reference_transcript_when_a_voice_is_set(monkeypatch, model_id):
    """audio.cpp rejects a reference WAV without reference_text; report it first."""
    item = definition(model_id)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "server-model")
    monkeypatch.setattr(AudioCppUtil, "generate", lambda *args, **kwargs: pytest.fail("network attempted"))
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", ""))
    project = FakeProject(model_id=model_id)
    assert adapter.generate_using_project(project, ["hello"]) == (
        "Voice clone transcript required when a voice clone sample is supplied"
    )


@pytest.mark.parametrize("model_id", [BREEZE_ID, OMNIVOICE_ID, HIGGS_ID, FISH_S2_ID, DOTS_ID])
def test_voice_readiness_is_lazy_for_every_audio_cpp_behavior(monkeypatch, model_id):
    item = definition(model_id)
    support = AudioCppModelSupport(item)
    project = FakeProject(model_id=model_id, values={"seed": -1})

    def local_issues() -> list:
        # The server-availability issue is the cached observation; this test is
        # about voice state, which readiness no longer inspects.
        return [issue for issue in support.get_blocking_issues(project) if issue.short != "audio.cpp server"]

    assert local_issues() == []
    assert support.get_voice_display_info(project).value.endswith("none")
    assert any("no voice reference" in warning for warning in support.get_warning_issues(project))

    # A configured-but-missing sample and a missing transcript are validated
    # lazily (pre-flight/generation), not as readiness blockers.
    monkeypatch.setattr(ProjectVoiceUtil, "get_voice_values", lambda *args: ["missing.flac"])
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: "/nonexistent/missing.flac")
    assert local_issues() == []
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: __file__)
    assert local_issues() == []

    # A required-voice family is equally lazy.
    monkeypatch.undo()
    other = definition()
    other_issues = AudioCppModelSupport(other).get_blocking_issues(FakeProject())
    assert not any(issue.short == "voice sample" for issue in other_issues)


def test_omnivoice_audio_cpp_instructions_menu_item_sets_and_clears(monkeypatch, capsys):
    from tts_audiobook_tool.ask_advanced import AskAdvanced
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    project = Project.model_validate({"dir_path": "", "tts_model_type": OMNIVOICE_ID})
    state = SimpleNamespace(project=project)
    # The standard string-input helper prefills the stored value, so the prompt
    # is editable in place.
    prefills: list[str] = []

    def fake_ask(message: str = "", prefill: str = "") -> str:
        prefills.append(prefill)
        return "female, young adult"

    monkeypatch.setattr(AskAdvanced, "ask", fake_ask)
    monkeypatch.setattr(Project, "save", lambda self: "")
    model_type = TtsModelType.require_by_id(OMNIVOICE_ID)

    items = ModelMenuShared.make_voice_instructions_item(state, OMNIVOICE_ID, validate_omnivoice=True)
    assert len(items) == 1
    assert "(optional)" in items[0].label(state)
    assert ProjectVoiceUtil.get_primary_voice_value(project, model_type) == ""

    # Without a voice, the prompt does not carry the cloning caveat.
    items[0].handler(state, items[0])
    assert project.get_model_setting(OMNIVOICE_ID, "instruct") == "female, young adult"
    assert prefills == [""]
    output = capsys.readouterr().out
    assert "Enter voice design instructions" in output
    assert "minimal effect" not in output
    assert "female, young adult" in items[0].label(state)

    # Editing again offers the stored value as the prefill, and the cloning
    # caveat is present because a voice is configured.
    monkeypatch.setattr(ProjectVoiceUtil, "get_voice_values", lambda *args: ["sample.flac"])
    items[0].handler(state, items[0])
    assert prefills[-1] == "female, young adult"
    assert "minimal effect" in capsys.readouterr().out

    items = ModelMenuShared.make_voice_instructions_item(state, OMNIVOICE_ID, validate_omnivoice=True)
    assert len(items) == 2 and items[1].label == "Clear instructions"
    items[1].handler(state, items[1])
    assert not project.get_model_setting(OMNIVOICE_ID, "instruct")
    assert "(optional)" in items[0].label(state)


def test_glm_payload_splits_flow_guidance_into_options(monkeypatch, tmp_path: Path):
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(GLM_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "glm-tts-q8")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=GLM_ID, values={"num_inference_steps": 12})
    project.language_code = "en"
    result = adapter.generate_using_project(project, ["hello"])
    assert not isinstance(result, str)
    assert len(captured) == 1
    payload = captured[0]
    assert payload["voice_ref"]["type"] == "base64"
    # num_inference_steps is in audio.cpp's forwarded top-level set, while Flow
    # guidance is only readable inside `options`; the transcript rides as
    # reference_text because the family requires it. No language is sent even
    # though the project has one.
    assert {key: value for key, value in payload.items() if key != "voice_ref"} == {
        "model": "glm-tts-q8", "input": "hello", "response_format": "wav", "seed": 7,
        "temperature": 1.0, "top_p": 0.8, "top_k": 25,
        "num_inference_steps": 12, "options": {"flow_guidance_scale": 0.7},
        "reference_text": "the exact words spoken",
    }
    assert adapter.generate_using_project(project, ["x"], on_stream_end=lambda: None) \
        == "GLM-TTS does not support streaming"


def test_cosyvoice3_definition_declares_only_controls_the_session_accepts():
    # The audio.cpp port rejects undeclared request options and has no
    # temperature/top-p/repetition-penalty knobs, unlike `cosyvoice3_sglomni`.
    item = definition(COSYVOICE3_ID)
    assert item.family == "cosyvoice3" and item.tasks == ("tts", "clon") and item.mode == "offline"
    assert item.session_options == {}
    assert item.voice_required and item.spec.requires_voice and item.reference_transcript
    assert item.language_policy == "omit"
    assert item.request_options == {"text_chunk_size": 100000}
    numeric = {name: parameter for name, parameter in item.parameters.items()
               if not isinstance(parameter, AudioCppTextParameter)}
    assert {name: (parameter.type, parameter.default, parameter.min, parameter.max, parameter.target)
            for name, parameter in numeric.items()} == {
        "top_k": ("int", 25, 1, 100, "top_level"),
        "num_inference_steps": ("int", 10, 1, 50, "top_level"),
    }
    # "Mode" picks audio.cpp's request template from a fixed list; the
    # instruction is free text, read only by the `instruct` template.
    template = item.parameters["template_name"]
    assert isinstance(template, AudioCppTextParameter) and template.target == "options"
    assert template.default == "zero_shot"
    assert [choice.value for choice in template.choices] == ["zero_shot", "cross_lingual", "instruct"]
    instruction = item.parameters["instruction"]
    assert isinstance(instruction, AudioCppTextParameter) and instruction.target == "options"
    assert instruction.default == "" and instruction.choices == ()
    assert [(control.kind, control.parameter, control.target_menu) for control in item.menu] == [
        ("voice_samples", "", None), ("choice", "template_name", "model"),
        ("voice_instructions", "instruction", "model"), ("parameter", "top_k", "model"),
        ("parameter", "num_inference_steps", "model"), ("seed", "", "model"),
    ]
    assert item.spec.default_output_sample_rate == 24000 and item.spec.file_tag == "cosyvoice3"
    assert not item.spec.can_stream


def test_cosyvoice3_mode_rejects_values_outside_its_choices():
    item = definition(COSYVOICE3_ID)
    template = item.parameters["template_name"]
    for value in ("cross_lingual", "instruct", "zero_shot"):
        assert AudioCppSettings.set(FakeProject(model_id=COSYVOICE3_ID), template, value) == ""
    # A hand-edited project value outside the list surfaces as a readiness issue
    # rather than reaching the server, which would reject the unknown template.
    for invalid in ("sft", "", 1):
        with pytest.raises(ValueError, match="must be"):
            AudioCppSettings.get(FakeProject(model_id=COSYVOICE3_ID, values={"template_name": invalid}), template)


def _capture_cosyvoice3_payload(monkeypatch, tmp_path: Path, values: dict) -> dict:
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition(COSYVOICE3_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "cosyvoice3-q8")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    captured = []

    def fake_generate(url, payload, print_request=False):
        captured.append(payload)
        return SimpleNamespace(data=np.asarray([0], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    project = FakeProject(model_id=COSYVOICE3_ID, values={"seed": 3, **values})
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)
    payload = captured[0]
    assert payload.pop("voice_ref")["data"].startswith("data:audio/wav;base64,")
    return payload


def test_cosyvoice3_payload_sends_transcript_without_language(monkeypatch, tmp_path: Path):
    # The zero_shot template (the session default) conditions on reference_text;
    # no language is sent, an unset instruction is omitted, and the native
    # 600-codepoint splitter is disabled so one app segment stays one generation.
    payload = _capture_cosyvoice3_payload(monkeypatch, tmp_path, {"top_k": 40})
    assert payload == {
        "model": "cosyvoice3-q8", "input": "hello", "response_format": "wav", "seed": 3,
        "top_k": 40, "num_inference_steps": 10,
        "options": {"text_chunk_size": 100000, "template_name": "zero_shot"},
        "reference_text": "the exact words spoken",
    }


def test_cosyvoice3_payload_sends_mode_and_instruction_in_options(monkeypatch, tmp_path: Path):
    # Both are audio.cpp request options, not forwarded top-level fields. The
    # instruct template ignores the transcript, so it is left out of the request.
    payload = _capture_cosyvoice3_payload(monkeypatch, tmp_path, {
        "template_name": "instruct", "instruction": "Speak warmly with clear articulation.",
    })
    assert payload["options"] == {
        "text_chunk_size": 100000, "template_name": "instruct",
        "instruction": "Speak warmly with clear articulation.",
    }
    assert "reference_text" not in payload


def test_cosyvoice3_cross_lingual_omits_and_does_not_require_transcript(monkeypatch, tmp_path: Path):
    # CosyVoice3Behavior withdraws the transcript for templates that ignore it:
    # it is not sent, and a missing one does not block generation. A stored
    # instruction is likewise dropped outside the instruct template.
    payload = _capture_cosyvoice3_payload(monkeypatch, tmp_path, {
        "template_name": "cross_lingual", "instruction": "Speak warmly.",
    })
    assert payload["options"] == {"text_chunk_size": 100000, "template_name": "cross_lingual"}
    assert "reference_text" not in payload

    item = definition(COSYVOICE3_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "cosyvoice3-q8")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("sample.flac", ""))
    project = FakeProject(model_id=COSYVOICE3_ID, values={"template_name": "cross_lingual"})
    assert not isinstance(adapter.generate_using_project(project, ["hello"]), str)


def test_cosyvoice3_zero_shot_drops_a_stored_instruction(monkeypatch, tmp_path: Path):
    # Only the instruct template reads `instruction`; a value left over from an
    # earlier instruct session must not appear in a zero_shot request.
    payload = _capture_cosyvoice3_payload(monkeypatch, tmp_path, {"instruction": "Speak warmly."})
    assert payload["options"] == {"text_chunk_size": 100000, "template_name": "zero_shot"}
    assert payload["reference_text"] == "the exact words spoken"


def test_adjust_payload_cannot_change_adapter_owned_fields(monkeypatch, tmp_path: Path):
    # adjust_payload is an escape hatch, but model identity, the resolved seed,
    # voice and transcript stay the adapter's decisions; a hook that rewrites
    # them is a programming error, caught before the request is sent.
    from tts_audiobook_tool.tts_models.audio_cpp_behavior_cosyvoice3 import CosyVoice3Behavior
    monkeypatch.setattr(CosyVoice3Behavior, "adjust_payload",
                        lambda self, payload, values: payload.update(seed=0, model="other"))
    with pytest.raises(RuntimeError, match="adapter-owned request fields: model, seed"):
        _capture_cosyvoice3_payload(monkeypatch, tmp_path, {})


INSTRUCT_BLOCKER = "Instructions are required when Mode is Instruct"


@pytest.mark.parametrize("values, expected", [
    ({"template_name": "instruct"}, [INSTRUCT_BLOCKER]),
    ({"template_name": "instruct", "instruction": "   "}, [INSTRUCT_BLOCKER]),
    ({"template_name": "instruct", "instruction": "Speak warmly."}, []),
    ({"template_name": "zero_shot"}, []),
])
def test_cosyvoice3_blocks_instruct_mode_without_instructions(values, expected):
    # audio.cpp would accept an empty instruction but fall back to a generic
    # prompt (effectively cross-lingual), so readiness blocks it instead.
    item = definition(COSYVOICE3_ID)
    support = AudioCppModelSupport(item)
    project = FakeProject(model_id=COSYVOICE3_ID, values=values)
    issues = [issue for issue in support.get_blocking_issues(project) if issue.short != "audio.cpp server"]
    assert [issue.verbose for issue in issues] == expected
    assert all(issue.short == "instructions" for issue in issues)
    assert INSTRUCT_BLOCKER not in support.get_warning_issues(project)


def test_cosyvoice3_behavior_blockers_wait_for_valid_values():
    # Behavior rules only see validated values; an invalid stored Mode is
    # reported as that parameter's own issue instead of reaching the hook.
    item = definition(COSYVOICE3_ID)
    project = FakeProject(model_id=COSYVOICE3_ID, values={"template_name": "sft"})
    issues = [issue for issue in AudioCppModelSupport(item).get_blocking_issues(project)
              if issue.short != "audio.cpp server"]
    assert [issue.short for issue in issues] == ["template_name"]


def test_cosyvoice3_generation_refuses_instruct_without_instructions(monkeypatch):
    # The adapter rechecks behavior blockers, so a caller that skipped readiness
    # still never sends the request.
    item = definition(COSYVOICE3_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "cosyvoice3-q8")
    monkeypatch.setattr(AudioCppUtil, "generate", lambda *args, **kwargs: pytest.fail("network attempted"))
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair",
                        lambda *args: ("sample.flac", "the exact words spoken"))
    project = FakeProject(model_id=COSYVOICE3_ID, values={"template_name": "instruct"})
    assert adapter.generate_using_project(project, ["hello"]) == INSTRUCT_BLOCKER


def test_models_default_to_catalog_driven_behavior():
    # Entries without a registered subclass get the base behavior, which answers
    # from the catalog alone and leaves payloads untouched. Constructing every
    # shipped behavior also verifies each subclass's REQUIRED_PARAMETERS against
    # the shipped catalog.
    from tts_audiobook_tool.tts_models.audio_cpp_behavior import AudioCppModelBehavior
    from tts_audiobook_tool.tts_models.audio_cpp_behavior_cosyvoice3 import CosyVoice3Behavior
    for model_id, item in load_audio_cpp_definitions().models.items():
        behavior = AudioCppModelSupport(item).behavior
        expected = CosyVoice3Behavior if model_id == COSYVOICE3_ID else AudioCppModelBehavior
        assert type(behavior) is expected
        if expected is AudioCppModelBehavior:
            assert behavior.uses_reference_transcript({}) == item.reference_transcript


@pytest.mark.parametrize("mutate, match", [
    (lambda parameters: parameters.pop("template_name"), "requires catalog parameter 'template_name'"),
    (lambda parameters: parameters.update(
        template_name=replace(parameters["template_name"], choices=())),
     "expects 'template_name' to be a choice parameter, not text"),
    (lambda parameters: parameters.update(instruction=parameters["top_k"]),
     "expects 'instruction' to be a text parameter, not number"),
])
def test_behavior_rejects_catalog_parameters_it_cannot_read(mutate, match):
    # A catalog rename or retype of a parameter a subclass reads fails when the
    # behavior is built, not as a KeyError partway through a generation.
    from tts_audiobook_tool.tts_models.audio_cpp_behavior_cosyvoice3 import CosyVoice3Behavior
    item = definition(COSYVOICE3_ID)
    parameters = dict(item.parameters)
    mutate(parameters)
    with pytest.raises(ValueError, match=match):
        CosyVoice3Behavior(replace(item, parameters=parameters))


def test_cosyvoice3_requires_voice_and_transcript_before_network(monkeypatch):
    item = definition(COSYVOICE3_ID)
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "cosyvoice3-q8")
    monkeypatch.setattr(AudioCppUtil, "generate", lambda *args, **kwargs: pytest.fail("network attempted"))
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("", ""))
    assert adapter.generate_using_project(FakeProject(model_id=COSYVOICE3_ID), ["hello"]) == (
        "A voice clone sample is required"
    )
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("sample.flac", ""))
    assert adapter.generate_using_project(FakeProject(model_id=COSYVOICE3_ID), ["hello"]) == (
        "Voice clone transcript required when a voice clone sample is supplied"
    )


def test_http_speech_contract_and_safe_request_log(monkeypatch, capsys):
    wav = BytesIO()
    sf.write(wav, np.full(40, .2), 24000, format="WAV", subtype="PCM_16")
    recorded = []

    def handle(request):
        recorded.append(request)
        return httpx.Response(200, content=wav.getvalue())

    real_client = httpx.Client
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: real_client(
        transport=httpx.MockTransport(handle), **kwargs))
    payload = {"model": "operator-model",
               "voice_ref": {"type": "base64", "data": "data:audio/wav;base64," + "QUJD" * 40 + "TOPSECRET"}}
    sound = AudioCppUtil.generate("http://audio.example", payload, print_request=True)
    assert not isinstance(sound, str) and sound.sr == 24000
    assert len(recorded) == 1 and recorded[0].url.path == "/v1/audio/speech"
    assert recorded[0].headers["accept"] == "audio/wav"
    assert recorded[0].method == "POST"
    assert json.loads(recorded[0].content) == payload
    out = capsys.readouterr().out
    # The request log uses the same presentation as the SGL-Omni backend: a
    # "Sending generation request" header followed by pretty-printed JSON whose
    # long string values (the voice sample data URI) are ellipsized.
    assert "Sending generation request to" in out
    assert '"model": "operator-model"' in out
    assert "..." in out
    assert "TOPSECRET" not in out


def test_prompts_are_requested_one_at_a_time_in_order(monkeypatch, tmp_path: Path):
    # audio.cpp's server runs one request at a time per model, so the adapter must
    # never pipeline: one request per prompt, in order, no overlap.
    path = tmp_path / "sample.flac"
    sf.write(path, np.full(100, .2), 24000, format="FLAC")
    item = definition()
    adapter = AudioCppBackendAdapter(item, AudioCppModelSupport(item), "operator-model")
    monkeypatch.setattr(ProjectVoiceUtil, "current_voice_reference_pair", lambda *args: ("sample.flac", ""))
    monkeypatch.setattr(ProjectVoiceUtil, "resolve_voice_file_path", lambda *args: str(path))
    monkeypatch.setattr("tts_audiobook_tool.tts_models.audio_cpp_configured.random.randrange", lambda stop: 7)
    inputs: list[str] = []
    in_flight = 0
    max_in_flight = 0

    def fake_generate(url, payload, print_request=False):
        nonlocal in_flight, max_in_flight
        in_flight += 1
        max_in_flight = max(max_in_flight, in_flight)
        time.sleep(0.02)
        inputs.append(payload["input"])
        in_flight -= 1
        return SimpleNamespace(data=np.asarray([len(inputs) - 1], dtype=np.float32), sr=24000)

    monkeypatch.setattr(AudioCppUtil, "generate", fake_generate)
    result = adapter.generate_using_project(FakeProject(), ["first", "second", "third"])
    assert not isinstance(result, str)
    assert [int(sound.data[0]) for sound in result] == [0, 1, 2]
    assert inputs == ["first", "second", "third"]
    assert max_in_flight == 1


def test_values_rejected_before_network(monkeypatch):
    item = definition()
    support = AudioCppModelSupport(item)
    adapter = AudioCppBackendAdapter(item, support, "exact-id")
    monkeypatch.setattr(AudioCppUtil, "generate", lambda *args, **kwargs: pytest.fail("network attempted"))
    assert "seed" in adapter.generate_using_project(FakeProject({"seed": 2**32}), ["text"])
    assert "exaggeration" in adapter.generate_using_project(FakeProject({"exaggeration": float("inf")}), ["text"])


@pytest.mark.parametrize("model_id", tuple(load_audio_cpp_definitions().models))
def test_catalog_controls_are_partitioned_without_changing_order(monkeypatch, model_id):
    from tts_audiobook_tool.menus.menu_util import MenuItem
    from tts_audiobook_tool.menus.voice.voice_audio_cpp_menu import VoiceAudioCppMenu
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import ModelAudioCppMenu
    from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
    from tts_audiobook_tool.menus.model.model_menu_shared import ModelMenuShared

    definition = load_audio_cpp_definitions().models[model_id]
    state = SimpleNamespace(project=Project(tts_model_type=model_id))
    voice_items = [MenuItem(label, lambda *_: None)
                   for label in ("samples", "selection mode", "selections")]
    voice_calls = []

    def make_voice_items(current, model_type):
        assert current is state and model_type.id == model_id
        voice_calls.append(model_type)
        return voice_items

    monkeypatch.setattr(VoiceMenuShared, "make_voice_sample_items", make_voice_items)
    monkeypatch.setattr(ModelAudioCppMenu, "make_parameter_item",
                        lambda _state, parameter, _label, *_, **__: MenuItem(parameter.name, lambda *_: None))
    monkeypatch.setattr(ModelAudioCppMenu, "make_choice_item",
                        lambda _state, parameter, _label: MenuItem(parameter.name, lambda *_: None))
    monkeypatch.setattr(ModelMenuShared, "make_seed_item",
                        lambda *_, **__: MenuItem("seed", lambda *_: None))
    monkeypatch.setattr(ModelMenuShared, "make_voice_instructions_item",
                        lambda *_, **__: [MenuItem("instructions", lambda *_: None),
                                          MenuItem("clear instructions", lambda *_: None)])

    # Controls the model's behavior hides at its defaults (eg CosyVoice3's
    # Instructions outside Instruct mode) are absent; order is otherwise kept.
    from tts_audiobook_tool.menus.model.model_audio_cpp_menu import AudioCppMenuContext
    context = AudioCppMenuContext(state, definition)

    actual_voice = VoiceAudioCppMenu.make_items(state, definition)
    voice_control_count = sum(control.kind == "voice_samples" for control in definition.menu)
    expected_voice = []
    for control in definition.menu:
        if control.kind == "voice_samples":
            expected_voice.extend(item.label for item in voice_items)
        elif control.target_menu == "voice" and context.is_visible(control):
            if control.kind == "voice_instructions":
                expected_voice.extend(("instructions", "clear instructions"))
            elif control.kind == "seed":
                expected_voice.append("seed")
            else:
                expected_voice.append(control.parameter)
    assert [item.label for item in actual_voice] == expected_voice
    assert len(voice_calls) == voice_control_count
    assert all(not item.superlabel for item in actual_voice)
    assert not hasattr(VoiceAudioCppMenu, "make_parameter_item")

    actual_model = ModelAudioCppMenu.make_items(state, definition)
    expected = []
    for control in definition.menu:
        if control.target_menu != "model" or not context.is_visible(control):
            continue
        if control.kind == "seed":
            expected.append("seed")
        elif control.kind == "voice_instructions":
            expected.extend(("instructions", "clear instructions"))
        else:
            expected.append(control.parameter)
    assert [item.label for item in actual_model] == expected
    assert all(not item.superlabel for item in actual_model)
    assert len(voice_calls) == voice_control_count  # Model rendering never expands the voice group.
