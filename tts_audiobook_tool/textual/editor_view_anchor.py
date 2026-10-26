"""Session-scoped view position memory for the full-screen content editors.

An anchor records where an editor was scrolled when it closed, in Project line
indices rather than visible row indices, so it survives differing row layouts
(section headings, filters) between sessions. Restoration is best effort: the
stored line text is used to follow a line which moved by a modest amount, and
otherwise the stored index is clamped to the current line range.

Deliberately free of Textual imports so non-UI code (``State``) can hold it.
"""

from collections.abc import Callable
from dataclasses import dataclass

# How far (in lines, each direction) to look for a line whose text matches the
# stored text before falling back to the stored index.
TEXT_MATCH_SEARCH_RADIUS = 500


@dataclass(frozen=True)
class EditorViewAnchor:
    """One editor's last view position, in Project line terms."""

    # First content line at or below the top visible row
    top_line: int
    top_line_text: str
    # Structural rows (eg, section headings) shown above top_line at the top
    top_leading_rows: int = 0
    highlighted_line: int | None = None
    highlighted_line_text: str = ""


# Keyed by editor (see run_content_textual_app)
EditorViewAnchors = dict[str, EditorViewAnchor]


def resolve_anchor_line(
    line: int,
    text: str,
    line_count: int,
    line_text: Callable[[int], str],
) -> int | None:
    """Return the current line which best corresponds to a stored line.

    Prefers the nearest line (searching outward from the stored index) whose
    text still matches; otherwise clamps the stored index into range.
    """
    if line_count <= 0:
        return None
    if text:
        for offset in range(TEXT_MATCH_SEARCH_RADIUS + 1):
            candidates = (line,) if offset == 0 else (line - offset, line + offset)
            for candidate in candidates:
                if 0 <= candidate < line_count and line_text(candidate) == text:
                    return candidate
    return min(max(line, 0), line_count - 1)
