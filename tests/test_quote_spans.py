from tts_audiobook_tool.text_ops.quote_spans import (
    DOUBLE_QUOTE_STYLES,
    QuoteSpan,
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
