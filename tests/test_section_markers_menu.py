from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.menu_util import get_string_from
from tts_audiobook_tool.menus.section_markers_menu import (
    LIMITED_SUBLABEL,
    SectionMarkersMenu,
)
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.text_util import strip_ansi_codes
from tts_audiobook_tool.textual.section_markers_dialog import (
    make_blank_line_marker_indices,
)


def make_group(text: str, reason: Reason = Reason.SENTENCE) -> PhraseGroup:
    return PhraseGroup([Phrase(text, reason)])


def make_multisection_state() -> State:
    project = Project.model_validate({
        "book": Book(sections=[
            BookSection(phrase_groups=[make_group(f"Line {i}.") for i in range(6)]),
            BookSection(phrase_groups=[]),
        ]),
    })
    return cast(State, SimpleNamespace(project=project))


def get_menu_subheading(state: State) -> str:
    with patch(
        "tts_audiobook_tool.menus.section_markers_menu.MenuUtil.menu"
    ) as menu:
        SectionMarkersMenu.menu(state)

    subheading = menu.call_args.kwargs["subheading"]
    return strip_ansi_codes(get_string_from(state, subheading))


def test_split_points_subheading_lists_output_file_ranges() -> None:
    state = make_multisection_state()
    state.project.markers = {2}

    subheading = get_menu_subheading(state)

    assert subheading == (
        "File 1: lines 1 to 2 (0/2 generated)\n"
        "File 2: lines 3 to 6 (0/4 generated)\n"
        + strip_ansi_codes(LIMITED_SUBLABEL)
    )


def test_split_points_subheading_omits_file_range_for_single_output() -> None:
    state = make_multisection_state()

    assert get_menu_subheading(state) == strip_ansi_codes(LIMITED_SUBLABEL)


def test_blank_line_markers_start_the_group_after_each_space_break() -> None:
    groups = [
        make_group("First section.\n\n\n", Reason.SPACE_BREAK),
        make_group("Second section.\n\n\n", Reason.SPACE_BREAK),
        make_group("Third section.", Reason.SENTENCE),
    ]

    assert make_blank_line_marker_indices(groups) == [1, 2]


def test_blank_line_markers_ignore_other_breaks_and_trailing_space_break() -> None:
    groups = [
        make_group("First paragraph.\n\n", Reason.PARAGRAPH),
        make_group("Last section.\n\n\n", Reason.SPACE_BREAK),
    ]

    assert make_blank_line_marker_indices(groups) == []


def test_blank_line_markers_detect_space_break_inside_group() -> None:
    groups = [
        PhraseGroup([
            Phrase("First section.\n\n\n", Reason.SPACE_BREAK),
            Phrase("Heading. ", Reason.SENTENCE),
        ]),
        make_group("Body.", Reason.SENTENCE),
    ]

    assert make_blank_line_marker_indices(groups) == [1]


def test_blank_line_marker_uses_section_start_from_segmented_text() -> None:
    groups = PhraseGrouper.text_to_groups(
        "First section.\n\n\nSecond section.",
        max_words=100,
    )

    assert make_blank_line_marker_indices(groups) == [1]
