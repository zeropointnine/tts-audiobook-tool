from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool.app_types import NormalizationType
from tts_audiobook_tool.menus.concat_menu import ConcatMenu, LOUDNORM_SUBHEADING
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State


def test_normalization_menu_uses_profile_descriptions_and_saves_selection() -> None:
    project = Project.model_validate({})
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.options_menu"
    ) as options_menu, patch(
        "tts_audiobook_tool.menus.concat_menu.Tts.get_type",
        return_value=None,
    ):
        ConcatMenu.normalization_menu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](NormalizationType.STRONGER)

    assert kwargs["heading_text"] == "Loudness normalization"
    assert kwargs["labels"] == [item.value.label for item in NormalizationType]
    assert kwargs["values"] == list(NormalizationType)
    assert kwargs["current_value"] is NormalizationType.DEFAULT
    assert kwargs["default_value"] is NormalizationType.DEFAULT
    assert kwargs["sublabels"] == [
        item.value.description for item in NormalizationType
    ]
    assert kwargs["subheading"] == LOUDNORM_SUBHEADING
    assert project.normalization_type is NormalizationType.STRONGER
    save.assert_called_once_with()


def test_normalization_copy_describes_whole_file_processing() -> None:
    assert "standardize overall loudness" in LOUDNORM_SUBHEADING
    assert "control peaks" in LOUDNORM_SUBHEADING
    assert "TTS generations" not in LOUDNORM_SUBHEADING
    assert NormalizationType.DEFAULT.value.description.startswith(
        "Approximates ACX loudness and peak requirements"
    )
    assert "mobile and noisy environments" in (
        NormalizationType.STRONGER.value.description
    )
    assert NormalizationType.DISABLED.value.description == (
        "Skips final loudness normalization"
    )
