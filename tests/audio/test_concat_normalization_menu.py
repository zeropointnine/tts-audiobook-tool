from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool.app_types import NormalizationType
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State


def test_normalization_menu_uses_simple_options_and_saves_selection() -> None:
    project = Project.model_validate({})
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.options_menu"
    ) as options_menu:
        ConcatMenu.normalization_menu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](NormalizationType.STRONGER)

    assert kwargs["heading_text"] == "Loudness"
    assert kwargs["labels"] == [item.value.label for item in NormalizationType]
    assert kwargs["values"] == list(NormalizationType)
    assert kwargs["current_value"] is NormalizationType.DEFAULT
    assert kwargs["default_value"] is NormalizationType.DEFAULT
    # Loudness is a simple preset choice, without explanatory menu copy.
    assert "sublabels" not in kwargs
    assert "subheading" not in kwargs
    assert project.normalization_type is NormalizationType.STRONGER
    save.assert_called_once_with()


def test_loudness_presets_keep_targets_and_stable_ids() -> None:
    # Cosmetic labels can change without changing saved IDs or audio targets.
    for preset, preset_id, target_lufs, peak_db in (
        (NormalizationType.DEFAULT, "default", -19.0, -3.0),
        (NormalizationType.STRONGER, "stronger", -16.0, -2.5),
    ):
        assert preset.value.id == preset_id
        assert preset.value.i == target_lufs
        assert preset.value.tp == peak_db
        assert NormalizationType.from_id(preset_id) is preset
        project = Project.model_validate({"normalization_type": preset_id})
        assert project.normalization_type is preset


def test_legacy_disabled_normalization_loads_as_default() -> None:
    # Removing "Disabled" adopts the default policy, not the louder preset.
    assert list(NormalizationType) == [NormalizationType.DEFAULT, NormalizationType.STRONGER]
    assert NormalizationType.from_id("none") is NormalizationType.DEFAULT
    project = Project.model_validate({"normalization_type": "none"})
    assert project.normalization_type is NormalizationType.DEFAULT
