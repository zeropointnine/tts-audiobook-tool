"""Recompute section-opening heading boundaries from editable text and titles."""

import re
from collections.abc import Sequence

from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types import BookSection
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason


MAX_HEADING_SEGMENTS = 5
_ENGLISH_CHAPTER_PREFIX = re.compile(
    r"^chapter\s+(?:[0-9]+|[ivxlcdm]+)\s*[:.\-–—]\s*(.+)$"
)


def _normalize_title(text: str) -> str:
    return " ".join(text.split()).casefold()


def heading_title_matches(text: str, title: str, language_code: str) -> bool:
    """Compare complete titles, with a narrow English-only chapter-prefix rule."""
    candidate = _normalize_title(text)
    title = _normalize_title(title)
    if not candidate or not title:
        return False
    if candidate == title:
        return True
    if language_code.replace("_", "-").casefold().split("-", 1)[0] != "en":
        return False
    match = _ENGLISH_CHAPTER_PREFIX.fullmatch(title)
    return match is not None and candidate == match.group(1)


def _text_end_reason(text: str) -> Reason:
    """Recover the ordinary boundary when clearing a derived heading ending."""
    text = app_text.normalize_line_terminators(text)
    if app_text.has_trailing_ornamental_line(text):
        return Reason.SPACE_BREAK
    num_lf = app_text.num_trailing_line_breaks(text)
    if num_lf >= 3:
        return Reason.SPACE_BREAK
    if num_lf:
        return Reason.PARAGRAPH
    return Reason.SENTENCE


def _opening_paragraph(section: BookSection) -> tuple[str, PhraseGroup] | None:
    """Find a complete opening paragraph within the fixed prompt-unit budget."""
    parts: list[str] = []
    for index, group in enumerate(section.phrase_groups[:MAX_HEADING_SEGMENTS]):
        text = app_text.normalize_line_terminators(group.text)
        paragraph, line_break, remainder = text.partition("\n")
        parts.append(paragraph)
        if line_break:
            # Never place an after-heading pause after body text or an explicit
            # ornament sharing the same prompt. Do not regroup user-edited text.
            if remainder.strip():
                return None
            return "".join(parts), group
        if index == len(section.phrase_groups) - 1:
            return "".join(parts), group
    # Reaching the budget is not evidence that the paragraph has ended.
    return None


def recompute_heading_ends(
    sections: Sequence[BookSection], language_code: str
) -> None:
    """Clear stale heading reasons globally, then classify each section opening.

    Text, grouping, and voice assignments are unchanged. Section endings and
    explicit ornamental boundaries retain precedence over heading pauses.
    """
    for section in sections:
        for group in section.phrase_groups:
            for phrase in group.phrases:
                if phrase.reason == Reason.HEADING_END:
                    phrase.reason = _text_end_reason(phrase.text)

        opening = _opening_paragraph(section)
        if opening is None:
            continue
        text, ending_group = opening
        if not ending_group.phrases:
            continue
        if not heading_title_matches(text, section.title, language_code):
            continue
        ending = ending_group.phrases[-1]
        if ending.reason == Reason.SECTION_BREAK:
            continue
        if app_text.has_trailing_ornamental_line(ending_group.text):
            continue
        ending.reason = Reason.HEADING_END
