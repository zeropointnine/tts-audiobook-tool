from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Literal

from tts_audiobook_tool.app_types import SegmentationStrategy


@dataclass(frozen=True)
class ChunkingConfig:
    language_code: str
    max_words: int = 40
    strategy: SegmentationStrategy = SegmentationStrategy.NORMAL


@dataclass(frozen=True)
class UiOp:
    kind: Literal["render", "println", "clear", "commit_render", "stop"]
    text: str = ""
    count: int = 0


class QueuedStream:
    """
    Stdio replacement that routes line-buffered writes to a ui_queue as
    'println' ops, so every print from every thread is serialized through
    the single ui_worker — no more raw writes racing with our cursor model.
    Per-thread mute() discards writes (used to silence TTS inference spam).
    """
    def __init__(self, real: object, ui_queue: "queue.Queue[UiOp]") -> None:
        self._real = real
        self._ui_queue = ui_queue
        self._local = threading.local()

    def mute(self) -> None:
        self._local.muted = True

    def unmute(self) -> None:
        self._local.muted = False

    def write(self, data: str) -> int:
        if not data:
            return 0
        if getattr(self._local, "muted", False):
            return len(data)
        buf = getattr(self._local, "buffer", "") + data
        while "\n" in buf:
            line, _, buf = buf.partition("\n")
            self._ui_queue.put(UiOp(kind="println", text=line))
        self._local.buffer = buf
        return len(data)

    def flush(self) -> None:
        # Partial-line output stays buffered until a newline arrives. Most
        # producers terminate lines, so this is fine in practice.
        pass

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)
