"""Small hand-built books for speaker assignment tests."""
from __future__ import annotations

from tts_audiobook_tool.app_types import Book, BookSection
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason

# Line spec: (text, ending reason, is_dialog)
LineSpec = tuple[str, Reason, bool]

NARRATION = False
DIALOG = True


def make_book(*sections: list[LineSpec], titles: list[str] | None = None) -> Book:
    titles = titles or [f"Chapter {i + 1}" for i in range(len(sections))]
    return Book(sections=[
        BookSection(
            phrase_groups=[
                PhraseGroup([Phrase(text, reason)], is_dialog=is_dialog)
                for text, reason, is_dialog in lines
            ],
            title=title,
        )
        for lines, title in zip(sections, titles)
    ])
