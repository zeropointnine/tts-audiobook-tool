"""Shared harness for driving ConcatUtil.concatenate_sound_segments() in tests.

By default the output writer is replaced by a `RecordingWriter`, which keeps
the streamed arrays and enforces the writer lifecycle (no writes after
close/abort, close at most once), and per-segment rendering / pause insertion
are replaced, so tests exercise only the concat flow itself. Pass
`real_writer=True` to stream into the real FLAC/AAC writers instead.

Everything run through `concatenate()` is under the no-FFmpeg guard.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Callable, Sequence
from unittest.mock import MagicMock, patch

import numpy as np

from no_ffmpeg_guard import no_ffmpeg_exe
from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.app_types.phrase import Phrase
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.audio_writers import Chapter
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline


class RecordingWriter:
    """Stand-in for an AudioWriter that records what it receives."""

    def __init__(self, streamed: list[np.ndarray] | None = None, **open_kwargs: Any) -> None:
        self.streamed: list[np.ndarray] = streamed if streamed is not None else []
        self.open_kwargs = open_kwargs
        self.chapters: list[Chapter] = []
        self.state = "open"

    def _require_open(self) -> None:
        if self.state != "open":
            raise AssertionError(f"writer used after it was {self.state}")

    def write(self, data: np.ndarray) -> None:
        self._require_open()
        self.streamed.append(np.array(data, copy=True))

    def set_chapters(self, chapters: Sequence[Chapter]) -> None:
        self._require_open()
        self.chapters = list(chapters)

    def close(self) -> None:
        self._require_open()
        self.state = "closed"

    def abort(self) -> None:
        if self.state == "open":
            self.state = "aborted"


def passthrough_append(sound_in: Sound, **_kwargs) -> Sound:
    """append_pause_or_section_effect stand-in that adds nothing."""
    return sound_in


def concatenate(
        phrases_and_paths: list[tuple[Phrase, str]],
        *,
        render: Sound | Callable[..., Sound],
        append: Callable[..., Sound] = passthrough_append,
        streamed: list[np.ndarray] | None = None,
        open_writer: MagicMock | None = None,
        writers: list[RecordingWriter] | None = None,
        real_writer: bool = False,
        pseudo_silence_compensation: bool | None = False,
        **kwargs: Any,
) -> Any:
    """Runs concatenate_sound_segments() with rendering stubbed.

    render: the rendered segment Sound, or a callable used as its side effect.
    append: side effect for SoundPipeline.append_pause_or_section_effect.
    streamed: when given, receives a copy of every array sent to the writer.
    open_writer: a MagicMock installed as ConcatUtil.open_output_writer (to
        inspect its arguments); it creates RecordingWriters unless it already
        has a side effect.
    writers: when given, receives every RecordingWriter created.
    real_writer: use the real output writers (pass dest_path in kwargs).
    pseudo_silence_compensation: value to patch in, or None to leave it as configured.
    kwargs: override or extend the concatenate_sound_segments() arguments.
    """
    render_patch = (
        {"side_effect": render} if callable(render) and not isinstance(render, Sound)
        else {"return_value": render}
    )
    arguments: dict[str, Any] = dict(
        dest_path="output.flac",
        phrases_and_paths=phrases_and_paths,
        use_break_sound_effect=True,
        high_shelf=HighShelfEq.DISABLED,
        reason_pauses=ReasonPauseTypes.NORMAL.value,
        print_progress=False,
    )
    arguments.update(kwargs)

    def make_recording_writer(*_args: Any, **open_kwargs: Any) -> RecordingWriter:
        writer = RecordingWriter(streamed, **open_kwargs)
        if writers is not None:
            writers.append(writer)
        return writer

    with ExitStack() as stack:
        stack.enter_context(no_ffmpeg_exe())
        if real_writer:
            if open_writer is not None or streamed is not None:
                raise ValueError("open_writer/streamed only apply to the recording writer")
        else:
            factory = open_writer if open_writer is not None else MagicMock()
            if factory.side_effect is None:
                factory.side_effect = make_recording_writer
            stack.enter_context(patch.object(ConcatUtil, "open_output_writer", factory))
        if pseudo_silence_compensation is not None:
            stack.enter_context(
                patch.object(ConcatUtil, "PSEUDO_SILENCE_COMPENSATION_ENABLED", pseudo_silence_compensation)
            )
        stack.enter_context(patch.object(SoundPipeline, "make_concat_rendered_sound_segment", **render_patch))
        stack.enter_context(patch.object(SoundPipeline, "append_pause_or_section_effect", side_effect=append))
        return ConcatUtil.concatenate_sound_segments(**arguments)


def concatenate_streamed(
        phrases: list[Phrase],
        sounds: list[np.ndarray],
        sample_rate: int,
        *,
        pause_s: float = 0.4,
        effect_s: float = 0.2,
        effect_level: float = 0.05,
        paths: list[str] | None = None,
        **kwargs: Any,
) -> list[np.ndarray]:
    """Runs the concat flow over pre-rendered `sounds` (one per phrase) with a
    real `pause_s` of silence after plain segments, or a constant `effect_level`
    tone of `effect_s` for break effects. Returns the arrays streamed to the writer.
    """
    streamed: list[np.ndarray] = []
    effect = np.full(round(effect_s * sample_rate), effect_level, dtype=np.float32)
    pause = np.zeros(round(pause_s * sample_rate), dtype=np.float32)

    def append(sound_in: Sound, break_effect: Any = None, **_kwargs) -> Sound:
        tail = effect if break_effect is not None else pause
        return Sound(np.concatenate([sound_in.data, tail]), sample_rate)

    renders = iter(Sound(s, sample_rate) for s in sounds)
    concatenate(
        [(phrase, paths[i] if paths is not None else f"segment{i}.flac") for i, phrase in enumerate(phrases)],
        render=lambda *_args, **_kwargs: next(renders),
        append=append,
        streamed=streamed,
        **kwargs,
    )
    return streamed
