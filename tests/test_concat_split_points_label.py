from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.concat_menu import ConcatMenu
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.util import make_file_line_ranges


def make_group(text: str) -> PhraseGroup:
    return PhraseGroup([Phrase(text, Reason.SENTENCE)])


def make_state() -> State:
    project = Project.model_validate({
        "book": Book(sections=[
            BookSection(phrase_groups=[make_group(f"Line {i}.") for i in range(6)]),
            BookSection(phrase_groups=[make_group(f"Line {i}.") for i in range(6, 9)]),
        ]),
    })
    state = cast(State, SimpleNamespace(project=project, prefs=SimpleNamespace(aac_bitrate="128k")))
    return state


def get_split_points_label(state: State) -> str:
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
    index = next(index for index, label in enumerate(labels) if label.startswith("File split points"))
    return strip_ansi_codes(labels[index])


def test_split_points_label_file_count_matches_make_file_line_ranges() -> None:
    state = make_state()
    state.project.markers = [3]

    label = get_split_points_label(state)

    num_files = len(make_file_line_ranges(state.project.markers, len(state.project.phrase_groups)))
    assert num_files == 2
    assert label == "File split points (currently: 1 item = 2 files)"


def test_split_points_label_counts_marker_zero_like_make_file_line_ranges() -> None:
    state = make_state()
    # Marker 0 is dropped by make_file_line_ranges, so 2 markers yield 2 files,
    # not the 3 implied by the old `num_markers + 1` assumption.
    state.project.markers = [0, 3]

    label = get_split_points_label(state)

    num_files = len(make_file_line_ranges(state.project.markers, len(state.project.phrase_groups)))
    assert num_files == 2
    assert label == "File split points (currently: 2 items = 2 files)"


def test_split_points_label_optional_when_no_markers() -> None:
    state = make_state()
    state.project.markers = []

    label = get_split_points_label(state)

    assert label == "File split points (optional)"
