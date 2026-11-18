import os
import tempfile

import pytest

from tts_audiobook_tool.app_types import SegmentationStrategy
from tts_audiobook_tool.text_ops import epub_css, text_format
from tts_audiobook_tool.text_ops.epub_css import (
    EpubCssResolver,
    EpubStylesheets,
    StylesheetRef,
    extract_stylesheet_refs,
    media_applies,
    parse_declarations,
    parse_stylesheet,
    resolve_style,
)
from tts_audiobook_tool.text_ops.epub_extractor import (
    BeautifulSoupEpubChapterTextExtractor,
    EpubExtractor,
    EpubSourceChapter,
)
from tts_audiobook_tool.text_ops.text_format import BOLD, ITALIC, UNDERLINE


def resolver(css: str, load=None) -> EpubCssResolver:
    return EpubCssResolver(parse_stylesheet(css, "styles/main.css", load))


def render_extracted(html: str, css: str) -> str | None:
    """Extracts `html` styled by `css` and renders its formatting as markup."""
    chapter = EpubSourceChapter("T", "t.xhtml", "application/xhtml+xml", html, css=resolver(css))
    result = BeautifulSoupEpubChapterTextExtractor().extract_text(chapter)
    if result.styled is None:
        return None
    alignment = text_format.align_masks(result.styled, result.text)
    assert alignment.is_exact
    return text_format.render(result.text, alignment.masks)


def test_parse_declarations_normalizes_relevant_properties() -> None:
    # `!important` is ignored, decoration is reduced to underline-or-not, and
    # irrelevant properties are dropped.
    assert parse_declarations(
        "font-style: Italic !important; font-weight:700; text-decoration: underline overline; color: red"
    ) == {"font-style": "italic", "font-weight": "700", "text-decoration": "underline"}
    assert parse_declarations("text-decoration-line: line-through") == {"text-decoration": "none"}


def test_font_shorthand_resets_unspecified_style_and_weight() -> None:
    # Per CSS, `font` resets the sub-properties it does not mention, so a plain
    # `font: 1em serif` turns italics and bold off.
    assert parse_declarations("font: italic bold 12px/1.2 Georgia, serif") == {
        "font-style": "italic", "font-weight": "bold",
    }
    assert parse_declarations("font: 1em serif") == {"font-style": "normal", "font-weight": "normal"}
    assert parse_declarations("font: italic serif") == {}  # no size: invalid
    assert parse_declarations("font: caption") == {}


@pytest.mark.parametrize("media, expected", [
    (None, True),
    ("", True),
    ("all", True),
    ("screen", True),
    ("only screen and (min-width: 600px)", True),
    ("(min-width: 600px)", True),
    ("print", False),
    ("amzn-kf8", False),
    ("amzn-mobi", False),
    ("not amzn-mobi", True),
    ("not screen", False),
    ("print, screen", True),
])
def test_media_applies_to_an_on_screen_reader(media, expected) -> None:
    # Kindle-specific overrides are skipped; their negations apply.
    assert media_applies(media) is expected


def test_stylesheet_parsing_handles_at_rules_comments_and_imports() -> None:
    css = """
        @charset "utf-8";
        @import url("base.css");
        /* .gone { font-style: italic } */
        @font-face { font-family: X; src: url(x.ttf); }
        @media amzn-kf8 { .kindle { font-weight: bold } }
        @media screen { .screen { font-weight: bold } }
        .a, .b { font-style: italic; }
        p > .complex { font-weight: bold }
        span:first-child { font-weight: bold }
    """
    loaded = []

    def load(path: str) -> str | None:
        loaded.append(path)
        return ".base { text-decoration: underline }"

    css_resolver = resolver(css, load)

    assert loaded == ["styles/base.css"]
    assert css_resolver.declarations_for("span", frozenset({"base"})) == {"text-decoration": "underline"}
    assert css_resolver.declarations_for("span", frozenset({"screen"})) == {"font-weight": "bold"}
    assert css_resolver.declarations_for("span", frozenset({"kindle"})) == {}
    assert css_resolver.declarations_for("span", frozenset({"gone"})) == {}
    assert css_resolver.declarations_for("span", frozenset({"b"})) == {"font-style": "italic"}
    # Combinators and pseudo-classes are not supported and simply never match
    assert css_resolver.declarations_for("span", frozenset({"complex"})) == {}
    assert css_resolver.declarations_for("span") == {}


def test_cascade_orders_by_specificity_then_source_order() -> None:
    css_resolver = resolver("""
        p.x { font-style: normal }
        .x { font-style: italic }
        p { font-weight: bold }
        p { font-weight: normal }
        #only { text-decoration: underline }
    """)
    # `p.x` outranks `.x` despite coming first; later `p` wins over earlier `p`
    assert css_resolver.declarations_for("p", frozenset({"x"})) == {
        "font-style": "normal", "font-weight": "normal",
    }
    assert css_resolver.declarations_for("span", frozenset({"x"})) == {"font-style": "italic"}
    assert css_resolver.declarations_for("p", frozenset(), "only")["text-decoration"] == "underline"


def test_resolve_style_follows_css_semantics() -> None:
    # Author CSS overrides tag defaults, italic/bold inherit and can be turned
    # off, `bolder`/`lighter` are relative, and underline cannot be cancelled
    # by a descendant.
    assert resolve_style("em", 0, {"font-style": "normal"}) == 0
    assert resolve_style("em", 0, {}) == ITALIC
    assert resolve_style("span", ITALIC, {"font-style": "normal"}) == 0
    assert resolve_style("span", BOLD, {"font-weight": "lighter"}) == 0
    assert resolve_style("span", 0, {"font-weight": "bolder"}) == BOLD
    assert resolve_style("span", 0, {"font-weight": "650"}) == BOLD
    assert resolve_style("span", 0, {"font-weight": "500"}) == 0
    assert resolve_style("span", UNDERLINE, {"text-decoration": "none"}) == UNDERLINE
    assert resolve_style("u", 0, {"text-decoration": "none"}) == 0
    assert resolve_style("a", 0, {}) == 0  # links are not underlined by default


def test_extractor_applies_calibre_style_classes() -> None:
    # The most common pattern: Calibre conversions name classes by style.
    css = ".italic { font-style: italic } .bold { font-weight: bold } .underline { text-decoration: underline }"
    html = ('<p class="calibre_8">It was <span class="italic">very</span> '
            '<span class="bold">dark</span> at <span class="underline">noon</span>.</p>')
    assert render_extracted(html, css) == "It was <i>very</i> <b>dark</b> at <u>noon</u>."


def test_extractor_applies_resets_on_body_paragraphs() -> None:
    # Void Star style: every paragraph resets style and weight, and spans add
    # them back. A paragraph class making a whole letter italic also works.
    css = """
        p.TX { font-weight: normal; font-style: normal; text-decoration: none }
        p.LT { font-style: italic }
        span.ePub-I { font-style: italic }
        em { font-style: normal; font-weight: bold }
    """
    html = ('<p class="TX">She <span class="ePub-I">ran</span>. <em>Now.</em></p>'
            '<p class="LT">Dear friend,</p>')
    assert render_extracted(html, css) == "She <i>ran</i>. <b>Now.</b>\n\n<i>Dear friend,</i>"


def test_stylesheet_refs_are_read_in_order_with_media_and_relative_paths() -> None:
    raw = """<html><head>
        <link rel="stylesheet" type="text/css" href="../styles/main.css"/>
        <link rel="alternate stylesheet" href="alt.css"/>
        <link rel="stylesheet" href="http://example.com/remote.css"/>
        <link rel="stylesheet" href="print.css" media="print"/>
        <style>.inline { font-style: italic }</style>
    </head><body><p>Text</p></body></html>"""

    refs = extract_stylesheet_refs(raw, "text/chapter.xhtml")

    assert [(ref.is_inline, ref.value if not ref.is_inline else "inline", ref.media) for ref in refs] == [
        (False, "styles/main.css", None),
        (False, "text/print.css", "print"),
        (True, "inline", None),
    ]
    css_resolver = EpubStylesheets({
        "styles/main.css": ".main { font-weight: bold }",
        "text/print.css": ".main { font-weight: normal }",
    }).resolver_for(refs)
    assert css_resolver is not None
    # The print sheet does not apply; the inline sheet does
    assert css_resolver.declarations_for("span", frozenset({"main", "inline"})) == {
        "font-weight": "bold", "font-style": "italic",
    }


def test_import_epub_applies_linked_stylesheets() -> None:
    # End to end through EbookLib, whose get_content() drops <head> links: the
    # extractor must read them from the original document bytes.
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("css-test")
    book.set_title("CSS Test")
    book.set_language("en")
    style = epub.EpubItem(
        uid="style", file_name="Styles/main.css", media_type="text/css",
        content=b".italic { font-style: italic } @media amzn-kf8 { .italic { font-style: normal } }",
    )
    book.add_item(style)
    chapter = epub.EpubHtml(uid="c0", file_name="Text/c0.xhtml", title="One")
    chapter.content = '<html><body><p>It was <span class="italic">cold</span>.</p></body></html>'
    chapter.add_link(href="../Styles/main.css", rel="stylesheet", type="text/css")
    book.add_item(chapter)
    book.spine = [chapter]
    book.add_item(epub.EpubNav())
    book.toc = [chapter]

    with tempfile.TemporaryDirectory() as temp_dir:
        path = os.path.join(temp_dir, "book.epub")
        epub.write_epub(path, book)
        result = EpubExtractor.import_epub(
            epub_path=path, max_words=40,
            segmentation_strategy=SegmentationStrategy.SENTENCE, language_code="en",
        )

    fmts = [phrase.fmt for group in result.phrase_groups for phrase in group.phrases]
    assert fmts == ["It was <i>cold</i>."]


def test_unknown_or_missing_stylesheets_degrade_to_no_styles() -> None:
    refs = (StylesheetRef(False, "missing.css", None, "t.xhtml"),)
    assert EpubStylesheets({}).resolver_for(refs) is None
    assert epub_css.parse_stylesheet("}}} .a { font-style: italic ") == []


# --- CSS-only scene breaks (class-name matching; see docs-dev/epub-import-flow.md)


def test_parse_declarations_reads_top_spacing_including_shorthands() -> None:
    # Only the top side matters; a later shorthand overrides an earlier longhand.
    assert parse_declarations("margin: 1em 0 0.5em") == {"margin-top": "1em"}
    assert parse_declarations("margin-top: 2em; margin: 0 auto") == {"margin-top": "0"}
    assert parse_declarations("padding-top: 3px") == {"padding-top": "3px"}


@pytest.mark.parametrize("declarations, expected", [
    ({"margin-top": "1.5em"}, True),
    ({"margin-top": "22.5px"}, True),
    ({"padding-top": "0.2em"}, True),
    ({"margin-top": "0em"}, False),
    ({"margin-top": "0"}, False),
    ({"margin-top": "-1em"}, False),
    ({"margin-top": "auto"}, False),
    ({"margin-top": "inherit"}, False),
    ({}, False),
])
def test_has_space_above_is_a_crisp_positive_length_test(declarations, expected) -> None:
    assert epub_css.has_space_above(declarations) is expected


@pytest.mark.parametrize("class_name, expected", [
    # Names observed in publisher markup
    ("space-break", True),
    ("space-break1", True),
    ("space-break-1", True),
    ("x04-Space-Break-FL", True),
    ("section-break", True),
    ("section_break", True),
    ("scenebreak", True),
    ("secbreak", True),
    ("textbreak3", True),
    ("SB1", True),
    ("sb3", True),
    # Decoys: a bare "space"/"break"/"transition", or negated and Kindle names
    ("copyright-space", False),
    ("text-standard-space-before--tx", False),
    ("nospaceabove", False),
    ("pagebreak", False),
    ("mbp_pagebreak", False),
    ("softbreak", False),
    ("transition", False),
    ("sborn", False),
    ("sb-std", False),
])
def test_is_scene_break_class(class_name, expected) -> None:
    assert BeautifulSoupEpubChapterTextExtractor.is_scene_break_class(class_name) is expected


def extract(html: str, css: str | None) -> str:
    chapter = EpubSourceChapter(
        "T", "t.xhtml", "application/xhtml+xml", html, css=resolver(css) if css is not None else None
    )
    return BeautifulSoupEpubChapterTextExtractor().extract_text(chapter).text


SCENE_BREAK_CSS = """
p { margin-top: 0; margin-bottom: 0; text-indent: 20pt; }
.space-break1 { margin-top: 1.5em; }
.space-break4 { margin-left: 7%; }
.copyright-space { margin-top: 1em; }
.x04-Space-Break-Orn { margin: 1em 0; text-align: center; }
"""


def test_scene_break_class_with_space_above_becomes_space_break() -> None:
    # The motivating case: the break exists only as a classed paragraph's top margin.
    html = '<body><p>Before.</p><p class="space-break1">After.</p></body>'
    assert extract(html, SCENE_BREAK_CSS) == "Before.\n\n\nAfter."
    # An inline style can supply the space too
    html = '<body><p>Before.</p><p class="scene-break" style="margin-top: 2em">After.</p></body>'
    assert extract(html, SCENE_BREAK_CSS) == "Before.\n\n\nAfter."


@pytest.mark.parametrize("html", [
    # A matching name without space above (eg, Slow Gods' verse indents)
    '<body><p>Before.</p><p class="space-break4">After.</p></body>',
    # A non-matching name with space above
    '<body><p>Before.</p><p class="copyright-space">After.</p></body>',
    # First text in the document: nothing to break from
    '<body><p class="space-break1">After.</p></body>',
])
def test_scene_break_class_requires_name_space_and_preceding_text(html) -> None:
    assert "\n\n\n" not in extract(html, SCENE_BREAK_CSS)


def test_scene_break_class_needs_a_stylesheet_to_confirm_it() -> None:
    # Without CSS the space-above check cannot pass, so the name alone does nothing.
    html = '<body><p>Before.</p><p class="space-break1">After.</p></body>'
    assert extract(html, None) == "Before.\n\nAfter."


def test_scene_break_class_on_ornament_is_left_to_ornament_handling() -> None:
    # The ornament paragraph itself is not vocalizable, so it gets no extra break;
    # segmentation already treats a lone ornament line as a space break.
    html = '<body><p>Before.</p><p class="x04-Space-Break-Orn">* * *</p><p>After.</p></body>'
    assert extract(html, SCENE_BREAK_CSS) == "Before.\n\n* * *\n\nAfter."
