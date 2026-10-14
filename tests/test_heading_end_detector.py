"""Heading detection is exact, section-local, and safe at TTS group boundaries."""

import pytest

from tts_audiobook_tool.app_types import BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops.heading_end_detector import (
    heading_title_matches,
    recompute_heading_ends,
)


def _group(*phrases: tuple[str, Reason]) -> PhraseGroup:
    return PhraseGroup([Phrase(text, reason) for text, reason in phrases])


def _section(title: str, *texts: str) -> BookSection:
    return BookSection([_group((text, Reason.PARAGRAPH)) for text in texts], title)


def _reasons(section: BookSection) -> list[list[Reason]]:
    return [[phrase.reason for phrase in group.phrases] for group in section.phrase_groups]


@pytest.mark.parametrize("language", ["en", "en-US", "en_GB", "es", "de", "zz", ""])
def test_exact_matching_casefolds_and_collapses_whitespace_in_every_language(language):
    assert heading_title_matches("  STRASSE\t und   Meer \r\n", "Straße und Meer", language)
    assert heading_title_matches("CHAPTER XI: Arrival", "Chapter XI: Arrival", language)


@pytest.mark.parametrize("language", ["en", "en-US", "en_GB"])
@pytest.mark.parametrize("number", ["12", "XIV", "xiv"])
@pytest.mark.parametrize("separator", [":", ".", "-", "–", "—"])
def test_english_metadata_chapter_prefix(language, number, separator):
    assert heading_title_matches("The Arrival", f"Chapter {number}{separator} The Arrival", language)


@pytest.mark.parametrize("language", ["es", "de", "zz", ""])
def test_non_english_cannot_strip_english_metadata_prefix(language):
    assert not heading_title_matches("Arrival", "Chapter 1: Arrival", language)


@pytest.mark.parametrize(
    ("text", "title"),
    [
        ("Arrival", "The Arrival"),
        ("The Arrival continues", "The Arrival"),
        ("The Arrival", "The Arrival continues"),
        ("The Arival", "The Arrival"),
        ("Arrival", "Chapter 1 Arrival"),  # Separator is mandatory.
        ("Arrival", "Chapter One: Arrival"),
        ("Arrival", "Part 1: Arrival"),
        ("Arrival", "Chapter 1/ Arrival"),
        ("Arrival", "Preface Chapter 1: Arrival"),  # Prefix must start the title.
        ("Chapter 1: Arrival", "Arrival"),  # Only metadata gets prefix stripping.
        ("", ""),
        (" \t\r\n", " \t"),
        ("", "Arrival"),
        ("Arrival", ""),
        ("", "Chapter 1:"),  # Stripping must leave a nonempty title.
        (" \t", "Chapter XIV — \t"),
    ],
)
def test_no_empty_fuzzy_substring_or_invalid_prefix_matches(text, title):
    assert not heading_title_matches(text, title, "en")


def test_candidate_spans_groups_and_changes_only_ending_last_phrase():
    first = _group(("The ", Reason.WORD), ("Long ", Reason.PHRASE))
    last = _group(("Way ", Reason.WORD), ("Home\n\n", Reason.PARAGRAPH))
    body = _group(("Body prose.\n", Reason.PARAGRAPH))
    section = BookSection([first, last, body], "The Long Way Home")
    groups = tuple(section.phrase_groups)
    phrases = tuple(phrase for group in groups for phrase in group.phrases)
    texts = tuple(phrase.text for phrase in phrases)
    for index, group in enumerate(groups):
        group.voice_index = index + 2

    assert recompute_heading_ends((section,), "en") is None

    assert _reasons(section) == [
        [Reason.WORD, Reason.PHRASE],
        [Reason.WORD, Reason.HEADING_END],
        [Reason.PARAGRAPH],
    ]
    # Reasons are the only mutation: no text rewriting, regrouping, or voice edits.
    assert len(section.phrase_groups) == len(groups)
    assert all(current is original for current, original in zip(section.phrase_groups, groups))
    assert all(current is original for current, original in zip(
        (phrase for group in section.phrase_groups for phrase in group.phrases), phrases
    ))
    assert tuple(phrase.text for phrase in phrases) == texts
    assert [group.voice_index for group in groups] == [2, 3, 4]
    assert section.title == "The Long Way Home"


@pytest.mark.parametrize("terminator", ["\n", "\n\n", "\n\n\n", "\r\n", "\r\n\r\n\r\n"])
def test_first_newline_completes_heading_and_whitespace_space_break_can_be_replaced(terminator):
    section = BookSection([_group(("Arrival" + terminator, Reason.SPACE_BREAK))], "Arrival")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.HEADING_END]]


@pytest.mark.parametrize("count", [1, 5])
@pytest.mark.parametrize("terminator", ["", "\n"])
def test_complete_heading_through_fifth_group_including_section_end(count, terminator):
    words = [f"word{index}" for index in range(count)]
    texts = [word + " " for word in words[:-1]] + [words[-1] + terminator]
    section = _section(" ".join(words), *texts)
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.PARAGRAPH]] * (count - 1) + [[Reason.HEADING_END]]


def test_six_group_candidate_is_out_of_bounds_even_when_it_matches():
    section = _section("one two three four five six", "one ", "two ", "three ", "four ", "five ", "six\n")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.PARAGRAPH]] * 6


def test_matching_prefix_at_group_five_is_incomplete_when_section_continues():
    # The window boundary is not a paragraph or section boundary.
    section = _section("one two three four five", "one ", "two ", "three ", "four ", "five", " six\n")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.PARAGRAPH]] * 6


def test_group_limit_does_not_count_internal_phrases():
    group = _group(*[(f"word{index} ", Reason.WORD) for index in range(7)], ("last\n", Reason.PARAGRAPH))
    section = BookSection([group], " ".join([f"word{index}" for index in range(7)] + ["last"]))
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.WORD] * 7 + [Reason.HEADING_END]]


@pytest.mark.parametrize("texts", [("Not the title\n", "Arrival\n"), ("The\n", "Arrival\n")])
def test_scan_stops_at_first_newline_not_later_paragraph(texts):
    section = _section("Arrival" if texts[0].startswith("Not") else "The Arrival", *texts)
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.PARAGRAPH], [Reason.PARAGRAPH]]


@pytest.mark.parametrize(
    "phrases",
    [
        [("Arrival\nBody prose.\n", Reason.PARAGRAPH)],
        [("Arrival\n", Reason.PARAGRAPH), ("Body prose.\n", Reason.PARAGRAPH)],
        [("Arrival\r\nBody prose.\r\n", Reason.PARAGRAPH)],
    ],
)
def test_heading_and_body_in_same_tts_group_cannot_receive_heading_pause(phrases):
    section = BookSection([_group(*phrases)], "Arrival")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[reason for _, reason in phrases]]


def test_heading_ending_group_must_not_also_contain_later_prose():
    section = BookSection([
        _group(("The ", Reason.WORD)),
        _group(("Arrival\n", Reason.PARAGRAPH), ("Body prose.", Reason.SENTENCE)),
    ], "The Arrival")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.WORD], [Reason.PARAGRAPH, Reason.SENTENCE]]


def test_repeated_title_later_or_in_other_sections_is_not_marked():
    sections = [
        _section("Arrival", "Arrival\n", "Body.\n", "Arrival\n"),
        _section("Departure", "Arrival\n", "Departure\n"),
        _section("Arrival", "Body.\n", "Arrival\n"),
    ]
    recompute_heading_ends(sections, "en")
    assert _reasons(sections[0]) == [[Reason.HEADING_END], [Reason.PARAGRAPH], [Reason.PARAGRAPH]]
    assert _reasons(sections[1]) == [[Reason.PARAGRAPH], [Reason.PARAGRAPH]]
    assert _reasons(sections[2]) == [[Reason.PARAGRAPH], [Reason.PARAGRAPH]]


@pytest.mark.parametrize("text", ["Arrival", "Arrival\n", "Arrival\n\n\n"])
def test_section_break_always_wins(text):
    section = BookSection([_group((text, Reason.SECTION_BREAK))], "Arrival")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.SECTION_BREAK]]


@pytest.mark.parametrize("reason", [Reason.SPACE_BREAK, Reason.HEADING_END])
def test_explicit_ornament_keeps_space_break_even_for_matching_heading(reason):
    section = BookSection([_group(("Arrival\n\n◇\n\n", reason))], "Arrival")
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.SPACE_BREAK]]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("stale", Reason.SENTENCE),
        ("stale\n", Reason.PARAGRAPH),
        ("stale\n\n", Reason.PARAGRAPH),
        ("stale\n\n\n", Reason.SPACE_BREAK),
        ("stale\r\n", Reason.PARAGRAPH),
        ("stale\r\n\r\n\r\n", Reason.SPACE_BREAK),
        ("stale\n\n◇\n\n", Reason.SPACE_BREAK),
    ],
)
def test_stale_heading_reasons_reset_anywhere_from_text(text, expected):
    # Place stale headings both beyond the scan window and inside a group's phrases.
    section = _section("Different", *["body\n"] * 6)
    section.phrase_groups.append(_group((text, Reason.HEADING_END), ("tail", Reason.WORD)))
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.PARAGRAPH]] * 6 + [[expected, Reason.WORD]]


@pytest.mark.parametrize(("title", "text"), [("", "Arrival\n"), ("Arrival", " \t\n"), ("", "")])
def test_empty_title_or_candidate_does_not_mark(title, text):
    section = _section(title, text)
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.PARAGRAPH]]


def test_empty_sections_and_empty_sequence_are_safe():
    assert recompute_heading_ends([], "en") is None
    empty = BookSection([], "Arrival")
    assert recompute_heading_ends([empty], "en") is None
    assert empty.phrase_groups == []


def test_recompute_is_idempotent_and_tracks_text_and_title_edits():
    section = _section("Arrival", "Arrival\n\n\n", "Body.\n")
    heading = section.phrase_groups[0].phrases[0]
    recompute_heading_ends([section], "en")
    assert heading.reason == Reason.HEADING_END
    recompute_heading_ends([section], "en")
    assert _reasons(section) == [[Reason.HEADING_END], [Reason.PARAGRAPH]]

    heading.text = "Departure\n\n\n"
    recompute_heading_ends([section], "en")
    assert heading.reason == Reason.SPACE_BREAK  # Lost match restores whitespace break.
    section.title = "Departure"
    recompute_heading_ends([section], "en")
    assert heading.reason == Reason.HEADING_END  # Title edit gains the match again.
    section.title = "Something else"
    recompute_heading_ends([section], "en")
    assert heading.reason == Reason.SPACE_BREAK
    heading.text = "Something else\n"
    recompute_heading_ends([section], "en")
    assert heading.reason == Reason.HEADING_END
    section.title = ""
    recompute_heading_ends([section], "en")
    assert heading.reason == Reason.PARAGRAPH
