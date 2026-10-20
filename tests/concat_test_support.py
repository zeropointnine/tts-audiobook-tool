"""Shared harness for driving ConcatUtil.concatenate_sound_segments() in tests.

ffmpeg streaming is mocked out and per-segment rendering / pause insertion are
replaced, so tests exercise only the concat flow itself.
"""

from __future__ import annotations

from contextlib import ExitStack
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import numpy as np

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.app_types.phrase import Phrase
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.reason_pauses import ReasonPauseTypes
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline


def passthrough_append(sound_in: Sound, **_kwargs) -> Sound:
    """append_pause_or_section_effect stand-in that adds nothing."""
    return sound_in


def concatenate(
        phrases_and_paths: list[tuple[Phrase, str]],
        *,
        render: Sound | Callable[..., Sound],
        append: Callable[..., Sound] = passthrough_append,
        streamed: list[np.ndarray] | None = None,
        init_stream: Any = None,
        pseudo_silence_compensation: bool | None = False,
        **kwargs: Any,
) -> Any:
    """Runs concatenate_sound_segments() with ffmpeg and rendering stubbed.

    render: the rendered segment Sound, or a callable used as its side effect.
    append: side effect for SoundPipeline.append_pause_or_section_effect.
    streamed: when given, receives a copy of every array sent to ffmpeg.
    init_stream: replacement for ConcatUtil.init_ffmpeg_stream (default MagicMock).
    pseudo_silence_compensation: value to patch in, or None to leave it as configured.
    kwargs: override or extend the concatenate_sound_segments() arguments.
    """
    render_patch = (
        {"side_effect": render} if callable(render) and not isinstance(render, Sound)
        else {"return_value": render}
    )
    stream_patch = (
        {"side_effect": lambda _process, data: streamed.append(data.copy())}
        if streamed is not None else {}
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

    with ExitStack() as stack:
        stack.enter_context(patch.object(ConcatUtil, "init_ffmpeg_stream", init_stream or MagicMock()))
        stack.enter_context(patch.object(ConcatUtil, "close_ffmpeg_stream"))
        stack.enter_context(patch.object(ConcatUtil, "add_audio_to_ffmpeg_stream", **stream_patch))
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
    tone of `effect_s` for break effects. Returns the arrays streamed to ffmpeg.
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
