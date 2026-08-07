from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.menus.section_markers_menu import make_blank_line_marker_indices
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper


def make_group(text: str, reason: Reason) -> PhraseGroup:
    return PhraseGroup([Phrase(text, reason)])


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
