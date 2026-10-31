import random

import pytest

from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.text_ops import text_format
from tts_audiobook_tool.text_ops.format_attach import attach_formatting, attach_formatting_to_groups
from tts_audiobook_tool.text_ops.text_format import BOLD, HEADING, ITALIC, UNDERLINE, StyledText


def styled(*runs: tuple[str, int]) -> StyledText:
    """StyledText from (text, mask) runs."""
    text = "".join(run for run, _ in runs)
    masks = [mask for run, mask in runs for _ in run]
    return StyledText(text, masks)


def test_parse_returns_text_and_per_character_masks() -> None:
    parsed = text_format.parse("a <b>b<i>c</i></b> &amp;&lt;&gt;")
    assert parsed == ("a bc &<>", [0, 0, BOLD, BOLD | ITALIC, 0, 0, 0, 0])


@pytest.mark.parametrize("fmt", [
    "<x>a</x>",          # unknown tag
    "<b>a",              # unclosed
    "a</b>",             # stray close
    "<b><i>a</b></i>",   # bad nesting
    "<b><b>a</b></b>",   # same style nested twice
    "a < b",             # unescaped <
    "a & b",             # unescaped &
    "a > b",             # unescaped >
    "<>a",
])
def test_parse_rejects_malformed_markup(fmt: str) -> None:
    assert text_format.parse(fmt) is None


def test_markup_is_valid_only_for_its_exact_text() -> None:
    # The self-check that makes stale formatting harmless: any text change
    # (including whitespace) invalidates the markup.
    assert text_format.is_valid_for("It was <i>cold</i>. ", "It was cold. ")
    assert not text_format.is_valid_for("It was <i>cold</i>. ", "It was cold.")
    assert not text_format.is_valid_for("It was <i>cold</i>. ", "It was warm. ")
    assert not text_format.is_valid_for(None, "It was cold. ")


def test_render_styles_inner_whitespace_by_shared_neighbor_styles() -> None:
    # Whitespace takes the styles common to the visible characters on both
    # sides; edge whitespace is never styled.
    text = " a b c \n"
    masks = [0, BOLD | UNDERLINE, 0, UNDERLINE, 0, ITALIC, 0, 0]
    assert text_format.render(text, masks) == " <b><u>a</u></b><u> b</u> <i>c</i> \n"


def test_render_nests_tags_canonically_and_returns_none_when_unstyled() -> None:
    assert text_format.render("Chapter One", [HEADING] * 8 + [HEADING | ITALIC] * 3) == (
        "<h>Chapter <i>One</i></h>"
    )
    assert text_format.render("plain", [0] * 5) is None
    assert text_format.render("a&b", [BOLD] * 3) == "<b>a&amp;b</b>"


def test_render_parse_round_trip() -> None:
    # Any rendered markup parses back to the same text, with whitespace
    # styled by the render rule.
    rng = random.Random(7)
    for _ in range(2000):
        text = "".join(rng.choice("ab <&> \n") for _ in range(rng.randint(0, 12)))
        masks = [rng.randint(0, 15) for _ in text]
        fmt = text_format.render(text, masks)
        if fmt is None:
            continue
        parsed = text_format.parse(fmt)
        assert parsed is not None and parsed[0] == text


def test_join_combines_self_contained_parts_and_bridges_whitespace() -> None:
    # Joining phrase markup (as for one flattened phrase group) treats invalid
    # parts as plain and styles the whitespace between continuing styles.
    parts = [
        ("She was very, ", "She was <u>very,</u> "),
        ("very cold. ", "<u>very</u> cold. "),
        ("Stale. ", "<b>Changed.</b> "),
    ]
    assert text_format.join(parts) == "She was <u>very, very</u> cold. Stale. "
    assert text_format.join([("a ", None), ("b", None)]) is None


def test_align_masks_tolerates_whitespace_differences() -> None:
    source = styled(("Chapter One\n\n\n  ", 0), ("It was ", 0), ("cold", ITALIC), (".", 0))
    alignment = text_format.align_masks(source, "Chapter One\n\nIt was cold.")
    assert alignment.is_exact
    assert text_format.render("Chapter One\n\nIt was cold.", alignment.masks) == (
        "Chapter One\n\nIt was <i>cold</i>."
    )


def test_align_masks_leaves_edited_words_unstyled() -> None:
    source = styled(("It was ", 0), ("very cold", ITALIC), (" today.", 0))
    target = "It was very chilly today."
    alignment = text_format.align_masks(source, target)
    assert alignment.num_unmatched_words == 1
    assert text_format.render(target, alignment.masks) == "It was <i>very</i> chilly today."


def test_attach_splits_cross_phrase_formatting_into_self_contained_markup() -> None:
    # Phrase boundaries can fall mid-word (after a dash); styles still land on
    # the right characters, and each phrase's markup is closed within it.
    phrases = [
        Phrase("It was very, ", Reason.PHRASE),
        Phrase("very cold—", Reason.PHRASE),
        Phrase("truly. ", Reason.SENTENCE),
    ]
    source = styled(("It was ", 0), ("very, very", ITALIC), (" cold—", 0), ("truly", UNDERLINE), (".", 0))

    alignment = attach_formatting(phrases, source)

    assert alignment.is_exact
    assert [phrase.fmt for phrase in phrases] == [
        "It was <i>very,</i> ",
        "<i>very</i> cold—",
        "<u>truly</u>. ",
    ]


def test_attach_replaces_existing_formatting() -> None:
    group = PhraseGroup([Phrase("Plain now.", Reason.SENTENCE, fmt="<b>Plain now.</b>")])
    attach_formatting_to_groups([group], StyledText.plain("Plain now."))
    assert group.phrases[0].fmt is None
