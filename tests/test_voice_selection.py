from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from tts_audiobook_tool.app_types import VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.generate_util import GenerateUtil, TtsModelError
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.voice.voice_menu_shared import VoiceMenuShared
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


def make_phrase_group(voice_index: int = -1) -> PhraseGroup:
    return PhraseGroup([Phrase("Hello.", Reason.SENTENCE)], voice_index=voice_index)


@pytest.fixture(autouse=True)
def initialized_tts_type():
    had_type = hasattr(Tts, "_type")
    previous_type = getattr(Tts, "_type", TtsModelType.MIRA)
    Tts._type = TtsModelType.MIRA
    try:
        yield
    finally:
        if not had_type:
            del Tts._type
        else:
            Tts._type = previous_type


def test_voice_select_mode_ids_and_project_default() -> None:
    project = Project.model_validate({})

    assert VoiceSelectMode.DISABLED.id == "disabled"
    assert VoiceSelectMode.AUTO_ADVANCE.id == "auto_advance"
    assert VoiceSelectMode.CUSTOM.id == "custom"
    assert VoiceSelectMode.DISABLED.description == (
        "Always use the first voice sample for every generation."
    )
    assert VoiceSelectMode.AUTO_ADVANCE.current_label == "auto-advance"
    assert VoiceSelectMode.AUTO_ADVANCE.label == "Auto-advance"
    assert VoiceSelectMode.AUTO_ADVANCE.description == (
        "Cycles through voice samples in order, one per generation."
    )
    assert VoiceSelectMode.get_default() == VoiceSelectMode.AUTO_ADVANCE
    assert VoiceSelectMode.get_by_id("disabled") == VoiceSelectMode.DISABLED
    assert VoiceSelectMode.get_by_id("custom") == VoiceSelectMode.CUSTOM
    assert VoiceSelectMode.get_by_id("invalid") is None
    assert project.voice_select_mode == VoiceSelectMode.AUTO_ADVANCE


@pytest.mark.parametrize("raw_value", [None, "invalid", 3])
def test_project_invalid_or_missing_voice_select_mode_defaults_to_auto_advance(raw_value: object) -> None:
    payload = {} if raw_value is None else {"voice_select_mode": raw_value}

    project = Project.model_validate(payload)

    assert project.voice_select_mode == VoiceSelectMode.AUTO_ADVANCE


def test_project_voice_select_mode_json_round_trip() -> None:
    project = Project.model_validate({"voice_select_mode": "custom"})

    payload = ProjectSerializationUtil.to_project_json_dict(project)
    assert payload["voice_select_mode"] == "custom"

    reloaded = Project.model_validate(payload)

    assert reloaded.voice_select_mode == VoiceSelectMode.CUSTOM


@pytest.mark.parametrize("voice_count, expected_item_count", [(0, 1), (1, 1), (2, 2)])
def test_voice_sample_selection_mode_item_requires_multiple_samples(
        voice_count: int,
        expected_item_count: int,
) -> None:
    project = Project.model_validate({
        "mira_voice_file_name": [f"voice-{index}.flac" for index in range(voice_count)],
    })
    state = cast(State, SimpleNamespace(project=project))

    items = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.MIRA)

    assert len(items) == expected_item_count
    assert get_string_from(state, items[0].label).startswith("Manage voice sample/s") or voice_count == 0
    if voice_count > 1:
        assert "Voice sample selection mode" in get_string_from(state, items[1].label)
        assert VoiceSelectMode.AUTO_ADVANCE.current_label in get_string_from(state, items[1].label)


def test_voice_sample_selection_mode_item_label_tracks_project_value() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))
    item = VoiceMenuShared.make_voice_sample_items(state, TtsModelType.MIRA)[1]

    assert VoiceSelectMode.AUTO_ADVANCE.current_label in get_string_from(state, item.label)

    project.voice_select_mode = VoiceSelectMode.CUSTOM

    assert VoiceSelectMode.CUSTOM.current_label in get_string_from(state, item.label)


def test_voice_sample_selection_mode_submenu_uses_options_menu_and_saves_selection() -> None:
    project = Project.model_validate({
        "mira_voice_file_name": ["voice-a.flac", "voice-b.flac"],
    })
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, \
            patch("tts_audiobook_tool.menus.voice.voice_menu_shared.MenuUtil.options_menu") as options_menu:
        VoiceMenuShared.voice_sample_selection_mode_submenu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](VoiceSelectMode.CUSTOM)

    assert kwargs["heading_text"] == "Voice sample selection mode"
    assert kwargs["labels"] == [mode.label for mode in VoiceSelectMode]
    assert kwargs["values"] == list(VoiceSelectMode)
    assert kwargs["current_value"] == VoiceSelectMode.AUTO_ADVANCE
    assert kwargs["default_value"] == VoiceSelectMode.get_default()
    assert kwargs["sublabels"] == [mode.description for mode in VoiceSelectMode]
    assert project.voice_select_mode == VoiceSelectMode.CUSTOM
    save.assert_called_once_with()


def run_generation(
        voice_select_mode: VoiceSelectMode,
        phrase_groups: list[PhraseGroup],
        indices: list[int],
        voices: list[str],
        next_rotation_index: int = 7,
) -> tuple[int, str, int]:
    project = cast(Project, SimpleNamespace(
        voice_select_mode=voice_select_mode,
        mira_voice_file_name=voices,
    ))
    captured_index = -1

    def generate_using_project(*args, **kwargs):
        nonlocal captured_index
        captured_index = kwargs["voice_rotation_index"]
        return "test stop"

    with patch("tts_audiobook_tool.generate_util.Tts.get_type", return_value=TtsModelType.MIRA), \
            patch("tts_audiobook_tool.generate_util.Tts.get_instance_if_exists", return_value=None), \
            patch("tts_audiobook_tool.generate_util.Tts.get_next_voice_rotation_index", return_value=next_rotation_index) as get_next, \
            patch("tts_audiobook_tool.generate_util.Tts.get_voice_tag_for_rotation_index", side_effect=lambda _, index: f"voice-{index}") as get_tag, \
            patch("tts_audiobook_tool.generate_util.Tts.generate_using_project", side_effect=generate_using_project):
        results = GenerateUtil.generate(
            project=project,
            indices=indices,
            phrase_groups=phrase_groups,
            force_random_seed=False,
            is_realtime=False,
            save_debug_files=False,
        )

    assert all(isinstance(result, TtsModelError) for result in results)
    tagged_index = get_tag.call_args.args[1]
    return captured_index, f"voice-{tagged_index}", get_next.call_count


def test_auto_advance_mode_retains_rotation() -> None:
    index, voice_tag, advance_count = run_generation(
        VoiceSelectMode.AUTO_ADVANCE,
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag, advance_count) == (7, "voice-7", 1)


@pytest.mark.parametrize("indices", [[0], [0, 1]])
def test_disabled_mode_always_uses_first_voice_without_advancing(indices: list[int]) -> None:
    index, voice_tag, advance_count = run_generation(
        VoiceSelectMode.DISABLED,
        [make_phrase_group(1), make_phrase_group(1)],
        indices,
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag, advance_count) == (0, "voice-0", 0)


@pytest.mark.parametrize("requested_index", [-2, -1, 2, 100])
def test_custom_mode_invalid_index_falls_back_to_zero(requested_index: int) -> None:
    index, voice_tag, advance_count = run_generation(
        VoiceSelectMode.CUSTOM,
        [make_phrase_group(requested_index)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag, advance_count) == (0, "voice-0", 0)


def test_custom_mode_uses_explicit_phrase_voice_index() -> None:
    index, voice_tag, advance_count = run_generation(
        VoiceSelectMode.CUSTOM,
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag, advance_count) == (1, "voice-1", 0)


def test_custom_mode_without_configured_voice_retains_rotation() -> None:
    index, voice_tag, advance_count = run_generation(
        VoiceSelectMode.CUSTOM,
        [make_phrase_group(1)],
        [0],
        [],
    )

    assert (index, voice_tag, advance_count) == (7, "voice-7", 1)


def test_custom_mode_is_ignored_for_batch_generation() -> None:
    index, voice_tag, advance_count = run_generation(
        VoiceSelectMode.CUSTOM,
        [make_phrase_group(0), make_phrase_group(1)],
        [0, 1],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert (index, voice_tag, advance_count) == (7, "voice-7", 1)


def test_custom_voice_selection_is_stable_across_realtime_style_retries() -> None:
    first = run_generation(
        VoiceSelectMode.CUSTOM,
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )
    retry = run_generation(
        VoiceSelectMode.CUSTOM,
        [make_phrase_group(1)],
        [0],
        ["voice-a.flac", "voice-b.flac"],
    )

    assert first == retry == (1, "voice-1", 0)
