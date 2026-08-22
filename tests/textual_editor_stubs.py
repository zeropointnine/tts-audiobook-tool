"""Shared fixtures for the ContentTextualApp editor test modules.

``test_content_textual_app.py`` keeps its own richer stubs; the concrete
editor test modules (generate, voice line, text, section markers) and
``test_project_util.py`` share the equivalents defined here.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tts_audiobook_tool.app_types import Book, BookSection, VoiceSelectMode
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.project import Project


def run(coroutine) -> None:
    asyncio.run(coroutine)


def make_phrase_group(text: str, voice_index: int = -1) -> PhraseGroup:
    return PhraseGroup(
        phrases=[Phrase(text, Reason.SENTENCE)],
        voice_index=voice_index,
    )


def make_project(sections: list[BookSection]) -> Project:
    return Project.model_validate({"book": Book(sections=sections)})


def make_project_with_markers(
    sections: list[BookSection], markers: list[int]
) -> Project:
    return Project.model_validate(
        {"book": Book(sections=sections), "markers": markers}
    )


@dataclass
class StubPhraseGroup:
    presentable_text: str
    voice_index: int = -1


@dataclass
class StubSoundSegment:
    file_name: str
    num_errors: int = -1


@dataclass
class StubSoundSegments:
    sound_segments_map: dict[int, list[StubSoundSegment]]
    failed_segment_files: set[str] = field(default_factory=set)
    deleted_index_batches: list[set[int]] = field(default_factory=list)
    invalidation_count: int = 0
    best_item_call_count: int = 0

    def get_existing_indices(self) -> set[int]:
        return set(self.sound_segments_map)

    def get_best_item_for(self, index: int) -> StubSoundSegment | None:
        self.best_item_call_count += 1
        items = self.sound_segments_map.get(index, [])
        return min(
            items,
            key=lambda item: item.num_errors if item.num_errors != -1 else 10_000,
            default=None,
        )

    def is_segment_failed(self, index: int, item: StubSoundSegment) -> bool:
        return item.file_name in self.failed_segment_files

    def delete_by_indices(self, indices: set[int]) -> None:
        self.deleted_index_batches.append(set(indices))
        for index in indices:
            self.sound_segments_map.pop(index, None)

    def force_invalidate(self) -> None:
        self.invalidation_count += 1


@dataclass
class StubProject:
    phrase_groups: list[StubPhraseGroup | PhraseGroup]
    sound_segments: StubSoundSegments | None = None
    sound_segments_path: str = "/project/segments"
    generate_range_string: str = "none"
    gen_auto_concat: bool = False
    voice_select_mode: VoiceSelectMode = VoiceSelectMode.AUTO_ADVANCE
    save_calls: list[str] = field(default_factory=list)
    save_error: str = ""
    book: Book | None = None

    def save(self) -> str:
        self.save_calls.append(self.generate_range_string)
        return self.save_error