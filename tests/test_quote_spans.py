from tts_audiobook_tool.text_ops.quote_spans import (
    DOUBLE_QUOTE_STYLES,
    QuoteSpan,
    find_paragraph_chained_quote_spans,
    find_quote_spans,
)


def test_returns_half_open_spans_in_source_order() -> None:
    text = 'Before "One." and “Two.” after.'

    spans = find_quote_spans(text)

    assert [text[span.start:span.end] for span in spans] == [
        '"One."',
        '“Two.”',
    ]


def test_finds_nested_quote_styles() -> None:
    text = '“Outer ‘inner’ end”'

    assert find_quote_spans(text) == [
        QuoteSpan(0, len(text), 0),
        QuoteSpan(7, 14, 1),
    ]


def test_supports_mixed_straight_and_curly_double_quotes() -> None:
    text = 'First “Hello." Then "Goodbye.” End.'

    assert [text[span.start:span.end] for span in find_quote_spans(text)] == [
        '“Hello."',
        '"Goodbye.”',
    ]


def test_unmatched_opening_does_not_hide_later_pair() -> None:
    text = 'He typed " without closing, then "Hello." Afterwards.'

    assert [text[span.start:span.end] for span in find_quote_spans(text)] == [
        '"Hello."',
    ]


def test_paragraph_scoped_mode_does_not_pair_across_lines() -> None:
    text = 'Before "Hello.\nAfter." End.'

    assert find_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
        paragraph_scoped=True,
    ) == []


def test_apostrophes_and_escaped_quotes_are_not_delimiters() -> None:
    text = r'''It's Bob's literal \"quote\".'''

    assert find_quote_spans(text) == []


def test_chained_spans_follow_multi_paragraph_convention() -> None:
    text = (
        '“First paragraph opens,\n'
        '\n'
        '“second continues,\n'
        '“third closes,” he said.\n'
        'Narration.\n'
    )

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert [text[span.start:span.end] for span in spans] == [
        '“First paragraph opens,\n\n“second continues,\n“third closes,”',
    ]


def test_chained_spans_support_straight_and_mixed_glyphs() -> None:
    text = '"Opens here\n“and closes here.” Then.\n'

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert [text[span.start:span.end] for span in spans] == [
        '"Opens here\n“and closes here.”',
    ]


def test_chained_context_abandoned_when_paragraph_does_not_lead_with_quote() -> None:
    text = '“Opens here\nNarration interrupts.\n“Later speech.” End.\n'

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert spans == []


def test_chained_pass_ignores_self_contained_pairs() -> None:
    text = '“Matched within one paragraph.” Then.\n'

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert spans == []


def test_chained_context_does_not_survive_trailing_non_quote_paragraph() -> None:
    text = '“Opens\n“closes here.”\n\n\nTrailing narration.\n'

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert [text[span.start:span.end] for span in spans] == [
        '“Opens\n“closes here.”',
    ]


def test_trailing_unmatched_opener_chains_across_paragraphs() -> None:
    # Each speech paragraph contains complete pairs earlier in the paragraph
    # and ends with an unmatched opening quote continuing the speech.
    text = (
        '“Closed pair.” Narration. “Opens and continues\n'
        '\n'
        '“into this closed pair.” More narration. “Opens again\n'
        '\n'
        '“and closes here.” After.\n'
    )

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert [text[span.start:span.end] for span in spans] == [
        '“Opens and continues\n\n“into this closed pair.”',
        '“Opens again\n\n“and closes here.”',
    ]


def test_trailing_unmatched_opener_abandoned_by_narration_paragraph() -> None:
    text = (
        '“Closed pair.” Narration. “Opens and continues\n'
        '\n'
        'Narration breaks the chain.\n'
        '\n'
        '“Unrelated later speech.” End.\n'
    )

    spans = find_paragraph_chained_quote_spans(
        text,
        styles=DOUBLE_QUOTE_STYLES,
    )

    assert spans == []
