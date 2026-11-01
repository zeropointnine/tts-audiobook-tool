"""
Minimal EPUB stylesheet support for display formatting (bold, italic, underline).

Only what the formatting layer needs is modeled: `font-style`, `font-weight`,
`text-decoration`/`text-decoration-line` and the `font` shorthand, matched by
simple compound selectors (`tag`, `.class`, `#id`, `tag.a.b`, `*`). Everything
else is ignored, and malformed CSS degrades to "no styles". Extracted text is
never affected. Rules and rationale: docs-dev/text-formatting-layer.md
"""

from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field
from typing import Callable, NamedTuple
from urllib.parse import unquote, urlsplit

from tts_audiobook_tool.text_ops import text_format

ITALIC_TAGS = frozenset({"i", "em", "cite"})
BOLD_TAGS = frozenset({"b", "strong"})
UNDERLINE_TAGS = frozenset({"u"})
HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})

MAX_IMPORT_DEPTH = 5

_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
_IMPORTANT_RE = re.compile(r"\s*!\s*important\s*$", re.I)
_AT_RULE_RE = re.compile(r"@([\w-]+)\s*(.*)", re.S)
_IMPORT_RE = re.compile(r"""^(?:url\(\s*)?(["']?)([^"')\s]+)\1\s*\)?\s*(.*)$""", re.S)
_COMPOUND_RE = re.compile(r"(\*|[A-Za-z][\w-]*)?((?:[.#][\w-]+)*)")
_FONT_SIZE_RE = re.compile(
    r"^(?:[\d.]+(?:px|pt|pc|em|rem|ex|ch|%|vw|vh|vmin|vmax|in|cm|mm|q)?"
    r"|xx-small|x-small|small|medium|large|x-large|xx-large|xxx-large|larger|smaller)(?:/.*)?$"
)
# CSS Fonts 4 allows any number from 1 to 1000
_FONT_WEIGHT_RE = re.compile(r"^\d{1,4}(?:\.\d+)?$")
_SYSTEM_FONTS = frozenset({"caption", "icon", "menu", "message-box", "small-caption", "status-bar"})
_GLOBAL_VALUES = frozenset({"inherit", "initial", "unset", "revert"})


# --- Declarations


def parse_declarations(text: str) -> dict[str, str]:
    """
    Parses a declaration block into canonical keys: "font-style", "font-weight"
    and "text-decoration" (normalized to "underline", "none" or a global keyword).
    `!important` is ignored. Later declarations win.
    """
    result: dict[str, str] = {}
    for declaration in text.split(";"):
        prop, separator, value = declaration.partition(":")
        if not separator:
            continue
        prop = prop.strip().lower()
        value = _IMPORTANT_RE.sub("", value.strip()).strip().lower()
        if not value:
            continue
        if prop in ("font-style", "font-weight"):
            result[prop] = value
        elif prop in ("text-decoration", "text-decoration-line"):
            result["text-decoration"] = _normalize_decoration(value)
        elif prop == "font":
            result.update(_parse_font_shorthand(value))
    return result


def _normalize_decoration(value: str) -> str:
    if value in _GLOBAL_VALUES:
        return value
    return "underline" if "underline" in value.split() else "none"


def _parse_font_shorthand(value: str) -> dict[str, str]:
    """`font` sets style and weight, resetting unspecified ones to normal."""
    if value in _GLOBAL_VALUES:
        return {"font-style": value, "font-weight": value}
    if value in _SYSTEM_FONTS:
        return {}
    font_style = "normal"
    font_weight = "normal"
    for token in value.split():
        if _FONT_SIZE_RE.match(token):
            return {"font-style": font_style, "font-weight": font_weight}
        if token in ("italic", "oblique"):
            font_style = token
        elif token in ("bold", "bolder", "lighter") or _FONT_WEIGHT_RE.match(token):
            font_weight = token
    return {}  # no font size: invalid shorthand


# --- Style resolution


def resolve_style(tag: str, inherited_style: int, declarations: dict[str, str]) -> int:
    """
    Display-style mask of an element from its tag, its ancestors' mask and its
    resolved author declarations. Follows CSS semantics:
    - author CSS overrides tag defaults (eg, `em {font-style: normal}`);
    - italic and bold inherit, and a descendant can turn them off;
    - an underline propagates to descendants and cannot be cancelled by them;
    - links are not underlined by default (only when a stylesheet says so).
    """
    inherited_italic = bool(inherited_style & text_format.ITALIC)
    inherited_bold = bool(inherited_style & text_format.BOLD)

    italic = _resolve_italic(declarations.get("font-style"), tag, inherited_italic)
    bold = _resolve_bold(declarations.get("font-weight"), tag, inherited_bold)

    decoration = declarations.get("text-decoration")
    own_underline = tag in UNDERLINE_TAGS if decoration is None else decoration == "underline"

    style = inherited_style & ~(text_format.ITALIC | text_format.BOLD)
    if italic:
        style |= text_format.ITALIC
    if bold:
        style |= text_format.BOLD
    if own_underline:
        style |= text_format.UNDERLINE
    if tag in HEADING_TAGS:
        style |= text_format.HEADING
    return style


def _resolve_italic(value: str | None, tag: str, inherited: bool) -> bool:
    if value in ("italic", "oblique"):
        return True
    if value in ("normal", "initial"):
        return False
    if value in ("inherit", "unset", "revert"):
        return inherited
    return inherited or tag in ITALIC_TAGS  # not declared, or unrecognized


def _resolve_bold(value: str | None, tag: str, inherited: bool) -> bool:
    if value in ("bold", "bolder"):
        return True
    if value in ("normal", "lighter", "initial"):
        return False
    if value is not None and _FONT_WEIGHT_RE.match(value):
        return float(value) >= 600
    if value in ("inherit", "unset", "revert"):
        return inherited
    return inherited or tag in BOLD_TAGS  # not declared, or unrecognized


# --- Media queries


def media_applies(media: str | None) -> bool:
    """
    Whether a media query list applies to an on-screen reader. Only the `all` and
    `screen` media types match, so Kindle-specific blocks (`amzn-kf8`,
    `amzn-mobi`) are skipped while `not amzn-mobi` applies. Feature conditions
    such as `(min-width: ...)` are assumed to match.
    """
    if media is None or not media.strip():
        return True
    for query in media.lower().split(","):
        tokens = query.split()
        if not tokens:
            continue
        negated = tokens[0] == "not"
        if tokens[0] in ("not", "only"):
            tokens = tokens[1:]
        if not tokens:
            continue
        media_type = "all" if tokens[0].startswith("(") else tokens[0]
        matches = media_type in ("all", "screen")
        if matches != negated:
            return True
    return False


# --- Selectors and stylesheets


class SimpleSelector(NamedTuple):
    tag: str | None  # None matches any element
    element_id: str | None
    classes: frozenset[str]

    @property
    def specificity(self) -> tuple[int, int, int]:
        return (1 if self.element_id else 0, len(self.classes), 1 if self.tag else 0)

    def matches(self, tag: str, classes: frozenset[str], element_id: str | None) -> bool:
        if self.tag is not None and self.tag != tag:
            return False
        if self.element_id is not None and self.element_id != element_id:
            return False
        return self.classes <= classes


def parse_selector(text: str) -> SimpleSelector | None:
    """Parses one compound selector; None for anything else (combinators, pseudo-classes, attributes)."""
    match = _COMPOUND_RE.fullmatch(text.strip())
    if match is None or not match.group(0):
        return None
    tag = match.group(1)
    tag = None if tag in (None, "*") else tag.lower()
    element_id: str | None = None
    classes: set[str] = set()
    for part in re.findall(r"[.#][\w-]+", match.group(2)):
        if part[0] == "#":
            if element_id is not None:
                return None
            element_id = part[1:]
        else:
            classes.add(part[1:])
    return SimpleSelector(tag, element_id, frozenset(classes))


@dataclass(frozen=True)
class CssRule:
    selector: SimpleSelector
    order: int
    declarations: dict[str, str]


@dataclass
class _ParseState:
    rules: list[CssRule] = field(default_factory=list)
    order: int = 0


def parse_stylesheet(
        css: str,
        base_path: str = "",
        load_stylesheet: Callable[[str], str | None] | None = None,
        state: _ParseState | None = None,
        depth: int = 0,
) -> list[CssRule]:
    """
    Parses relevant rules from a stylesheet. `@import` is followed through
    `load_stylesheet` (given a package path resolved against `base_path`),
    applicable `@media` and `@supports` blocks are entered, and other at-rules
    are skipped.
    """
    state = state or _ParseState()
    for prelude, block in _split_statements(_COMMENT_RE.sub("", css)):
        if prelude.startswith("@"):
            at_rule = _AT_RULE_RE.match(prelude)
            if at_rule is None:
                continue
            keyword = at_rule.group(1).lower()
            rest = at_rule.group(2).strip()
            if keyword == "import" and block is None:
                _parse_import(rest, base_path, load_stylesheet, state, depth)
            elif keyword == "media" and block is not None:
                if media_applies(rest):
                    parse_stylesheet(block, base_path, load_stylesheet, state, depth)
            elif keyword == "supports" and block is not None:
                parse_stylesheet(block, base_path, load_stylesheet, state, depth)
            continue
        if block is None:
            continue
        declarations = parse_declarations(block)
        if not declarations:
            continue
        for selector_text in prelude.split(","):
            selector = parse_selector(selector_text)
            if selector is not None:
                state.rules.append(CssRule(selector, state.order, declarations))
                state.order += 1
    return state.rules


def _parse_import(
        rest: str,
        base_path: str,
        load_stylesheet: Callable[[str], str | None] | None,
        state: _ParseState,
        depth: int,
) -> None:
    match = _IMPORT_RE.match(rest)
    if match is None or load_stylesheet is None or depth >= MAX_IMPORT_DEPTH:
        return
    if not media_applies(match.group(3)):
        return
    path = resolve_package_path(base_path, match.group(2))
    if path is None:
        return
    css = load_stylesheet(path)
    if css:
        parse_stylesheet(css, path, load_stylesheet, state, depth + 1)


def _split_statements(css: str) -> list[tuple[str, str | None]]:
    """Top-level (prelude, block) pairs; block is None for `;`-terminated statements."""
    result: list[tuple[str, str | None]] = []
    depth = 0
    start = 0
    prelude = ""
    block_start = 0
    quote: str | None = None
    index = 0
    while index < len(css):
        char = css[index]
        if quote is not None:
            if char == "\\":
                index += 1
            elif char == quote:
                quote = None
        elif char in "\"'":
            quote = char
        elif char == "{":
            if depth == 0:
                prelude = css[start:index].strip()
                block_start = index + 1
            depth += 1
        elif char == "}":
            if depth == 0:
                start = index + 1  # stray brace
            else:
                depth -= 1
                if depth == 0:
                    result.append((prelude, css[block_start:index]))
                    start = index + 1
        elif char == ";" and depth == 0:
            statement = css[start:index].strip()
            if statement:
                result.append((statement, None))
            start = index + 1
        index += 1
    return result


def resolve_package_path(base_path: str, href: str) -> str | None:
    """Package-relative path of `href` as referenced from `base_path`; None for external URLs."""
    parts = urlsplit(href)
    if parts.scheme or parts.netloc:
        return None
    path = unquote(parts.path).replace("\\", "/").strip()
    if not path:
        return None
    joined = path.lstrip("/") if path.startswith("/") else posixpath.join(posixpath.dirname(base_path), path)
    normalized = posixpath.normpath(joined)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if normalized.startswith("../") or normalized == "..":
        return None
    return normalized


class EpubCssResolver:
    """Resolves the author declarations that apply to an element."""

    def __init__(self, rules: list[CssRule]) -> None:
        self._rules_by_tag: dict[str, list[CssRule]] = {}
        self._rules_by_class: dict[str, list[CssRule]] = {}
        self._rules_by_id: dict[str, list[CssRule]] = {}
        self._universal_rules: list[CssRule] = []
        for rule in rules:
            selector = rule.selector
            if selector.element_id is not None:
                self._rules_by_id.setdefault(selector.element_id, []).append(rule)
            elif selector.classes:
                self._rules_by_class.setdefault(min(selector.classes), []).append(rule)
            elif selector.tag is not None:
                self._rules_by_tag.setdefault(selector.tag, []).append(rule)
            else:
                self._universal_rules.append(rule)
        self._cache: dict[tuple[str, frozenset[str], str | None], dict[str, str]] = {}

    def declarations_for(
            self,
            tag: str,
            classes: frozenset[str] = frozenset(),
            element_id: str | None = None,
    ) -> dict[str, str]:
        key = (tag, classes, element_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        candidates = list(self._universal_rules)
        candidates.extend(self._rules_by_tag.get(tag, []))
        for class_name in classes:
            candidates.extend(self._rules_by_class.get(class_name, []))
        if element_id is not None:
            candidates.extend(self._rules_by_id.get(element_id, []))
        matching = [rule for rule in candidates if rule.selector.matches(tag, classes, element_id)]
        matching.sort(key=lambda rule: (rule.selector.specificity, rule.order))

        result: dict[str, str] = {}
        for rule in matching:
            result.update(rule.declarations)
        self._cache[key] = result
        return result


# --- Book-level stylesheet collection


class StylesheetRef(NamedTuple):
    """A stylesheet referenced by a document: a linked package path or inline CSS text."""
    is_inline: bool
    value: str
    media: str | None
    base_path: str


def extract_stylesheet_refs(raw_html: str, document_path: str) -> tuple[StylesheetRef, ...]:
    """
    Stylesheet references of a document, in document order, from its raw
    markup. (EbookLib's `get_content()` rebuilds `<head>` without them.)
    """
    head_end = raw_html.lower().find("</head>")
    head = raw_html if head_end < 0 else raw_html[:head_end]
    if "<link" not in head.lower() and "<style" not in head.lower():
        return ()
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(head, "html.parser")
    refs: list[StylesheetRef] = []
    for element in soup.find_all(["link", "style"]):
        media = element.get("media")
        media = media if isinstance(media, str) else None
        if element.name == "style":
            refs.append(StylesheetRef(True, element.get_text(), media, document_path))
            continue
        rel = element.get("rel") or []
        rel_values = rel.lower().split() if isinstance(rel, str) else [value.lower() for value in rel]
        href = element.get("href")
        if "stylesheet" not in rel_values or "alternate" in rel_values or not isinstance(href, str):
            continue
        path = resolve_package_path(document_path, href)
        if path is not None:
            refs.append(StylesheetRef(False, path, media, document_path))
    return tuple(refs)


class EpubStylesheets:
    """The stylesheets of one EPUB, compiled into resolvers per distinct reference list."""

    def __init__(self, css_by_path: dict[str, str]) -> None:
        self._css_by_path = {
            posixpath.normpath(path).lstrip("/"): css for path, css in css_by_path.items()
        }
        self._resolvers: dict[tuple[StylesheetRef, ...], EpubCssResolver | None] = {}

    def resolver_for(self, refs: tuple[StylesheetRef, ...]) -> EpubCssResolver | None:
        if not refs:
            return None
        if refs in self._resolvers:
            return self._resolvers[refs]
        state = _ParseState()
        for ref in refs:
            if not media_applies(ref.media):
                continue
            if ref.is_inline:
                parse_stylesheet(ref.value, ref.base_path, self._css_by_path.get, state)
            else:
                css = self._css_by_path.get(ref.value)
                if css:
                    parse_stylesheet(css, ref.value, self._css_by_path.get, state)
        resolver = EpubCssResolver(state.rules) if state.rules else None
        self._resolvers[refs] = resolver
        return resolver
