from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool.menus.concat_menu import (
    ConcatMenu,
    REASON_PAUSES_SUBHEADING,
)
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.state import State


def test_concat_menu_places_reason_pauses_below_file_type() -> None:
    project = Project.model_validate({})
    prefs = SimpleNamespace(aac_bitrate="128k")
    state = cast(State, SimpleNamespace(project=project, prefs=prefs))

    with patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.menu"
    ) as menu, patch(
        "tts_audiobook_tool.menus.concat_menu.ProjectUtil.get_latest_concat_files",
        return_value=[],
    ):
        ConcatMenu.menu(state)
        make_items = menu.call_args.args[2]
        items = make_items(state)

    labels = [get_string_from(state, item.label) for item in items]
    file_type_index = next(
        index for index, label in enumerate(labels) if label.startswith("File type")
    )

    assert labels[file_type_index + 1].startswith("Pauses between segments")
    assert "currently:" in labels[file_type_index + 1]
    assert ReasonPauseTypes.NORMAL.value.menu_label in labels[file_type_index + 1]


def test_reason_pauses_menu_uses_options_and_saves_selection() -> None:
    project = Project.model_validate({})
    state = cast(State, SimpleNamespace(project=project))

    with patch.object(Project, "save") as save, patch(
        "tts_audiobook_tool.menus.concat_menu.MenuUtil.options_menu"
    ) as options_menu:
        ConcatMenu.reason_pauses_menu(state)
        kwargs = options_menu.call_args.kwargs
        kwargs["on_select"](ReasonPauseTypes.SHORTER)

    assert kwargs["heading_text"] == "Pauses between segments"
    assert kwargs["labels"] == [item.value.label for item in ReasonPauseTypes]
    assert kwargs["values"] == list(ReasonPauseTypes)
    assert kwargs["current_value"] is ReasonPauseTypes.NORMAL
    assert kwargs["default_value"] is ReasonPauseTypes.default()
    assert kwargs["sublabels"] == [
        item.value.description for item in ReasonPauseTypes
    ]
    assert kwargs["subheading"] == REASON_PAUSES_SUBHEADING
    assert project.reason_pauses is ReasonPauseTypes.SHORTER.value
    save.assert_called_once_with()
