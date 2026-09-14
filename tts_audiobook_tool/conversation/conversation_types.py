from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from tts_audiobook_tool.app_types import SegmentationStrategy


class ChatInputMode(Enum):
    MIC_IMMEDIATE = "mic_immediate"
    MIC_ENTER = "mic_enter"
    TEXT = "text"

    @property
    def id(self) -> str:
        return self.value

    @staticmethod
    def from_id(s: str) -> ChatInputMode | None:
        for item in ChatInputMode:
            if s == item.id:
                return item
        return None


@dataclass(frozen=True)
class ChunkingConfig:
    language_code: str
    max_words: int = 40
    strategy: SegmentationStrategy = SegmentationStrategy.SENTENCE_PLUS


@dataclass(frozen=True)
class ResponseSnapshot:
    spoken_segments: tuple[tuple[str, int, int], ...] = ()
    pending_sentences: tuple[str, ...] = ()
    render_buffer: str = "..."
    play_position_samples: int = 0
    playback_done: bool = False
    llm_content_received: bool = False
    interrupted: bool = False

    @property
    def text(self) -> str:
        parts = [text for text, _start, _end in self.spoken_segments]
        parts.extend(self.pending_sentences)
        if self.render_buffer and self.render_buffer not in ("...", "(thinking...)"):
            parts.append(self.render_buffer)
        return "".join(parts)


@dataclass(frozen=True)
class ResponseResult:
    text: str
    interrupted: bool = False
    error: str = ""

    @property
    def has_content(self) -> bool:
        return bool(self.text)
