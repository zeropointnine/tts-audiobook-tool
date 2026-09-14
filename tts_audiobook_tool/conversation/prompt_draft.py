from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tts_audiobook_tool.app_types import Segment
from tts_audiobook_tool.sound.sound_util import SoundUtil


@dataclass(frozen=True)
class PromptSubmission:
    text: str
    audio: np.ndarray | None = None


class PromptDraft:
    """UI-neutral phrase and microphone-audio draft for one user turn.

    The selection follows the tail while it sits on the last phrase:
    appending a new phrase moves the highlight to the new last phrase. Once
    the user moves the highlight away from the last phrase, appending new
    phrases leaves it untouched. Only the first phrase of an otherwise
    empty draft is selected.
    """

    def __init__(self) -> None:
        self.phrases: list[str] = []
        self.audio_chunks: list[np.ndarray | None] = []
        self.selected_index: int | None = None

    @property
    def text(self) -> str:
        return " ".join(self.phrases)

    @property
    def is_empty(self) -> bool:
        return not self.phrases

    def add_transcription(
        self,
        segments: list[Segment],
        audio: np.ndarray | None = None,
    ) -> bool:
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return self.add_phrase(text, audio)

    def add_phrase(self, text: str, audio: np.ndarray | None = None) -> bool:
        text = text.strip()
        if not text:
            return False
        self.phrases.append(text)
        self.audio_chunks.append(np.copy(audio) if audio is not None else None)
        if self.selected_index is None:
            # First phrase of the draft.
            self.selected_index = len(self.phrases) - 1
        elif self.selected_index == len(self.phrases) - 2:
            # The highlight was on the last phrase; follow the new one.
            self.selected_index = len(self.phrases) - 1
        return True

    def select_previous(self) -> None:
        if not self.phrases:
            self.selected_index = None
        elif self.selected_index is None:
            self.selected_index = len(self.phrases) - 1
        else:
            self.selected_index = max(0, self.selected_index - 1)

    def select_next(self) -> None:
        if not self.phrases:
            self.selected_index = None
        elif self.selected_index is None:
            self.selected_index = 0
        else:
            self.selected_index = min(len(self.phrases) - 1, self.selected_index + 1)

    def delete_selected(self) -> bool:
        if not self.phrases:
            self.selected_index = None
            return False
        index = self.selected_index
        if index is None:
            index = len(self.phrases) - 1
        self.phrases.pop(index)
        self.audio_chunks.pop(index)
        self.selected_index = (
            min(index, len(self.phrases) - 1) if self.phrases else None
        )
        return True

    def finalize(self) -> PromptSubmission | None:
        if not self.phrases:
            return None
        audio_parts = [
            audio for audio in self.audio_chunks if audio is not None and audio.size > 0
        ]
        audio = None
        if audio_parts:
            audio = SoundUtil.normalize(np.concatenate(audio_parts))
        submission = PromptSubmission(self.text, audio)
        self.reset()
        return submission

    def reset(self) -> None:
        self.phrases.clear()
        self.audio_chunks.clear()
        self.selected_index = None
