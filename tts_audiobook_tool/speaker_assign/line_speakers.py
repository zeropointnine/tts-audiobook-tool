"""
Read-only access to saved speaker attributions for project views (eg the
voice/line editor's character column). The attributions stay speaker
assignment data, not project data: views only display them.
"""
from __future__ import annotations

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.l import L
from tts_audiobook_tool.speaker_assign.speaker_assign_file_util import SpeakerAssignFileUtil


def load_line_speakers(project_dir: str, book: Book) -> dict[int, str]:
    """
    Speaker label by book-wide line, for every line of each attributed
    utterance whose text is unchanged since it was attributed. Empty when the
    project has no directory, no saved attributions, or an unreadable
    attributions file (logged, since callers treat the data as optional).
    """
    if not project_dir:
        return {}
    attributions = SpeakerAssignFileUtil.load_attributions_file(project_dir)
    if isinstance(attributions, str):
        L.w(attributions)
        return {}
    return attributions.speaker_of_each_line(book)
