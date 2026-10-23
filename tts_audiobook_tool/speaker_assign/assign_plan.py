from __future__ import annotations

from dataclasses import dataclass

from tts_audiobook_tool.app_types import Book
from tts_audiobook_tool.speaker_assign.attribution import AttributionsFile
from tts_audiobook_tool.speaker_assign.chunk_plan_util import DEFAULT_CHUNK_WORDS, ChunkPlanUtil
from tts_audiobook_tool.speaker_assign.utterance_util import UtteranceUtil


@dataclass(frozen=True)
class PlannedChunk:
    """A whole-book attribution chunk [start, end) that contains dialog.
    `number` is 1-based among the plan's chunks; `done` means it has saved
    results that are still valid for the book's text."""
    number: int
    start: int
    end: int
    utterance_count: int
    done: bool


@dataclass(frozen=True)
class AssignPlan:
    """
    The whole-book attribution plan (ChunkPlanUtil's chunks, in reading
    order), with each chunk's completion. Chunks without dialog are left out:
    they need no request.
    """
    chunks: tuple[PlannedChunk, ...]

    @property
    def done_count(self) -> int:
        return sum(1 for chunk in self.chunks if chunk.done)

    @property
    def remaining(self) -> list[PlannedChunk]:
        return [chunk for chunk in self.chunks if not chunk.done]

    @property
    def is_complete(self) -> bool:
        return all(chunk.done for chunk in self.chunks)

    @staticmethod
    def make(book: Book, attributions: AttributionsFile, chunk_words: int = DEFAULT_CHUNK_WORDS) -> AssignPlan:
        chunks: list[PlannedChunk] = []
        for start, end in ChunkPlanUtil.plan_chunks(book, chunk_words):
            utterance_count = len(UtteranceUtil.find_utterances(book, start, end))
            if not utterance_count:
                continue
            saved = attributions.get(start, end)
            chunks.append(PlannedChunk(
                number=len(chunks) + 1,
                start=start,
                end=end,
                utterance_count=utterance_count,
                done=saved is not None and saved.is_valid_for(book),
            ))
        return AssignPlan(tuple(chunks))
