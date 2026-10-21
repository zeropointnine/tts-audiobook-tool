from __future__ import annotations

import bisect
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.sound.silence_util import SilenceGapTrim, SilenceUtil
from tts_audiobook_tool.sound.sentence_gap_util import SentenceGapInsertion, SentenceGapParams, SentenceGapUtil
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.reason_pauses import ReasonPauses
from tts_audiobook_tool.util import *

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


class BreakEffectTracker:
    """
    Render-time decision helper implementing the app's two break-sound-effect rules:

    1. The first sound segment of a section never gets a SPACE_BREAK effect.
    2. In a run of consecutive sound segments that would each get a SPACE_BREAK
       effect, only the first keeps it; the rest fall back to silence pauses.

    A "sound segment" is a segment that actually emits audio. The caller feeds
    the tracker one group index and reason per emitted segment, in emission
    order, and receives the break effect to append for that segment (or None
    to append the configured pause instead).

    Rule 2 counts run membership on the would-be assignment, so a segment
    suppressed by rule 1 still anchors its run: its followers are suppressed
    as well. Any emitted segment whose reason is not SPACE_BREAK resets the
    run. SECTION_BREAK effects are never suppressed by these rules.

    The final segment of an output stream never receives any break effect.
    """

    def __init__(self, section_start_indices: list[int], start_group_index: int = 0):
        """
        :param section_start_indices:
            Group indices where sections begin, including 0 for the first
            section (eg from `ProjectBookUtil.get_section_start_indices`).
        :param start_group_index:
            The group index the output stream begins at. Seeding from the
            group preceding it means a stream starting mid-section does not
            falsely trigger rule 1, while one starting exactly at a section
            start does.
        """

        starts = sorted(set(section_start_indices))
        if not starts or starts[0] != 0:
            starts = [0, *starts]
        self._starts = starts

        preceding = start_group_index - 1
        self._last_section_index = self.section_index_for(preceding) if preceding >= 0 else -1
        self._space_break_run_active = False

    def section_index_for(self, group_index: int) -> int:
        return max(bisect.bisect_right(self._starts, group_index) - 1, 0)

    def next_break_effect(
        self,
        group_index: int,
        reason: Reason,
        effects_enabled: bool,
        is_final: bool,
    ) -> Reason | None:
        """
        Returns the break sound effect to append for the next emitted segment
        (SPACE_BREAK or SECTION_BREAK), or None for the configured pause.

        Must be called once per emitted segment, in emission order. Disabled
        effects and the final segment short-circuit to None without touching
        tracker state.
        """

        if not effects_enabled or is_final:
            return None

        section_index = self.section_index_for(group_index)
        first_in_section = section_index != self._last_section_index
        would_space_break = reason == Reason.SPACE_BREAK
        # Rule 2 uses the previous emitted segment's would-be status, so it
        # must be read before the run state is updated below.
        suppress_space_break = first_in_section or self._space_break_run_active

        self._last_section_index = section_index
        self._space_break_run_active = would_space_break

        if reason == Reason.SECTION_BREAK:
            return Reason.SECTION_BREAK
        if would_space_break and not suppress_space_break:
            return Reason.SPACE_BREAK
        return None


class SoundPipeline:
    @staticmethod
    def apply_generate_post_processing(sound: Sound) -> Sound:
        """
        App's standard post-processing treatment on generated sounds.

        - Trims silence from both ends
        - Applies peak normalization.
        - Does *not* touch samplerate.

        If the input is entirely silence, the returned Sound's data.size will be 0.
        Callers must decide how to report that.
        """
        sound = SilenceUtil.trim_silence_ends(sound)[0]
        if sound.data.size == 0:
            return sound
        data = SoundUtil.normalize(sound.data, headroom_db=NORMALIZATION_HEADROOM_DB)
        return Sound(data, sound.sr)

    @staticmethod
    def apply_generate_post_processing_with_info(sound: Sound) -> tuple[Sound, float | None, float | None, float]:
        """
        Same as apply_generate_post_processing, but also returns trim metadata
        suitable for user-facing reporting.
        """
        original_duration = sound.duration
        sound, start, end = SilenceUtil.trim_silence_ends(sound)
        start_time = start if start > 0 else None
        end_time = end if end < original_duration else None
        if sound.data.size == 0:
            return sound, start_time, end_time, original_duration
        data = SoundUtil.normalize(sound.data, headroom_db=NORMALIZATION_HEADROOM_DB)
        return Sound(data, sound.sr), start_time, end_time, original_duration

    @staticmethod
    def generate_processed_using_project(
        project: Project,
        prompts: list[str],
        force_random_seed: bool = False,
        *,
        apply_word_substitutions: bool = True,
        log_prompts: bool = False,
    ) -> list[Sound] | str:
        """
        Non-streaming TTS generation paired with app-standard generated-sound
        post-processing.

        Returns processed sounds, including zero-length sounds for fully silent
        outputs. Callers decide how to report silence. Diagnostic prompts may
        disable project word substitutions without bypassing other inference
        text preparation.
        """
        from tts_audiobook_tool.tts import Tts

        result = Tts.generate_using_project(
            project,
            prompts,
            force_random_seed,
            apply_word_substitutions=apply_word_substitutions,
            log_prompts=log_prompts,
        )
        if isinstance(result, str):
            return result

        sounds = [result] if isinstance(result, Sound) else result
        return [SoundPipeline.apply_generate_post_processing(sound) for sound in sounds]

    @staticmethod
    def make_concat_rendered_sound_segment(
        phrase: Phrase,
        path: str,
        high_shelf: HighShelfEq,
        reason_pauses: ReasonPauses,
        use_upsampler: bool = False,
        add_pause: bool = True,
    ) -> Sound | str:
        """
        Loads a saved segment file and applies concat/export rendering steps:

        - Applies upsampler (optional)
        - Resamples to 48k
        - Applies high shelf filter (optional)
        - Adds trailing pause (unless `add_pause` is False)

        When `add_pause` is False, the trailing pause is omitted and the caller
        is responsible for appending it (e.g. via `append_pause_or_section_effect`).
        This is used by the concat flow so that the pause or break effect can be
        decided based on `BreakEffectTracker` state and the pseudo-silence
        measured across adjacent segments.
        """

        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            return result

        sound = result

        if use_upsampler:
            result = SoundPipeline.apply_lava_sr_upsampling(sound)
            if isinstance(result, str):
                return result
            sound = result

        sound = SoundPipeline.resample_for_app(sound)

        sound = SoundPipeline.apply_high_shelf(sound, high_shelf)

        if add_pause:
            sound = SoundPipeline.append_pause_or_section_effect(
                sound,
                reason=phrase.reason,
                reason_pauses=reason_pauses,
                break_effect=None,
            )
        return sound

    @staticmethod
    def prepare_generated_sound_for_playback(
        sound: Sound,
        high_shelf: HighShelfEq,
        limit_silence_gaps: bool = False,
        limit_silence_gaps_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
    ) -> Sound:
        """
        Applies interactive playback shaping to freshly generated audio.
        Used by: realtime playback, server non-streaming, voice chat, and the
        word-substitutions pronunciation preview.

        - Limit silence gaps (optional)
        - Resample to 48k if not already
        - Apply high shelf (optional)
        """
        sound = SoundPipeline.limit_silence_gaps_if_enabled(
            sound,
            enabled=limit_silence_gaps,
            max_gap_duration=limit_silence_gaps_duration,
        )
        sound = SoundPipeline.resample_for_app(sound)
        return SoundPipeline.apply_high_shelf(sound, high_shelf)

    @staticmethod
    def make_break_effect_sound(
        break_effect: Reason,
        reason_pauses: ReasonPauses,
        sr: int,
        preceding_silence: float = 0.0,
        following_silence: float = 0.0,
    ) -> Sound | str:
        """
        Builds the inserted audio for a break sound effect (SPACE_BREAK or
        SECTION_BREAK): the effect asset, resampled to `sr`, centered in
        leading and trailing silence so the whole gap lasts the configured
        pause for that reason (or the effect's own length, if longer).

        `preceding_silence` / `following_silence`: seconds of pseudo-silence
        already at the end of the preceding segment and the start of the
        following one. They count toward the gap, so only the padding shrinks;
        the effect itself is never trimmed. See `SoundUtil.center_in_silence`.

        Shared by concat and realtime playback. Returns an error string if the
        reason has no effect or the asset can't be loaded.
        """
        if break_effect == Reason.SPACE_BREAK:
            path = SPACE_BREAK_SOUND_EFFECT_PATH
        elif break_effect == Reason.SECTION_BREAK:
            path = SECTION_BREAK_SOUND_EFFECT_PATH
        else:
            return f"No break sound effect for reason: {break_effect}"

        result = SoundFileUtil.load(path, sr)
        if isinstance(result, str):
            return f"{path}: {result}"
        return SoundUtil.center_in_silence(
            result,
            reason_pauses.get_pause_for(break_effect),
            preceding_silence=preceding_silence,
            following_silence=following_silence,
        )

    @staticmethod
    def append_pause_or_section_effect(
        sound: Sound,
        reason: Reason,
        reason_pauses: ReasonPauses,
        break_effect: Reason | None,
        preceding_silence: float = 0.0,
        following_silence: float = 0.0,
    ) -> Sound:
        """
        Appends the trailing break effect or pause for a segment.

        :param break_effect:
            Precomputed decision from `BreakEffectTracker.next_break_effect`:
            which break sound effect to append (SPACE_BREAK or SECTION_BREAK),
            or None to append the pause for ``reason`` instead. The effect is
            built by `make_break_effect_sound` (centered in silence, totalling
            the configured pause). If the effect can't be loaded, the plain
            pause for ``reason`` is appended instead.
        :param preceding_silence, following_silence:
            Seconds of pseudo-silence already present at the end of `sound`
            and at the start of the following segment. They count toward the
            gap between the two segments, so the inserted silence is reduced by
            their sum (never below zero) and the gap totals the configured
            pause. For a break effect only its padding shrinks; the effect
            itself is never trimmed. Used by the concat flow's pseudo-silence
            compensation.
        """

        if break_effect is not None:
            result = SoundPipeline.make_break_effect_sound(
                break_effect, reason_pauses, sound.sr,
                preceding_silence=preceding_silence,
                following_silence=following_silence,
            )
            if isinstance(result, str):
                printt(f"{COL_ERROR}Couldn't load break sound effect: {result}")
            else:
                new_data = np.concatenate([sound.data, result.data.astype(sound.data.dtype)])
                return Sound(new_data, sound.sr)

        pause_duration = max(
            0.0,
            reason_pauses.get_pause_for(reason) - preceding_silence - following_silence,
        )
        if pause_duration > 0:
            return SoundUtil.add_silence(sound, pause_duration)

        return sound

    @staticmethod
    def effective_min_sentence_silence(
        min_sentence_silence: float,
        limit_silence_gaps: bool,
        limit_silence_gaps_duration: float,
    ) -> float:
        """
        The minimum sentence silence to actually extend to. Gap limiting runs
        after sentence gap extension, so with limiting enabled any extension
        beyond the max gap duration would just be cut back down; the minimum is
        capped at the max instead (0 when the max is 0, ie disabled).
        """
        if min_sentence_silence <= 0:
            return 0.0
        if limit_silence_gaps:
            return max(0.0, min(min_sentence_silence, limit_silence_gaps_duration))
        return min_sentence_silence

    @staticmethod
    def extend_sentence_gaps_with_insertions(
        sound: Sound,
        timed_phrases: Sequence[TimedPhrase],
        words: Sequence[Mapping[str, Any]],
        min_gap_duration: float,
    ) -> tuple[Sound, list[SentenceGapInsertion]]:
        """
        Extends short silences at sentence boundaries inside `sound` to at least
        `min_gap_duration` seconds, using the segment sidecar's timings.
        Returns the new sound and a record of every insertion (positions are in
        the coordinates of the given sound), so callers can remap timings that
        were measured against the unextended audio.
        A `min_gap_duration` of 0 or less does nothing.
        """
        if min_gap_duration <= 0:
            return sound, []

        report = SentenceGapUtil.extend_sentence_gaps(
            sound,
            timed_phrases,
            words,
            SentenceGapParams(min_gap_s=min_gap_duration),
        )
        return report.sound, report.insertions

    @staticmethod
    def limit_silence_gaps_with_trims(
        sound: Sound,
        enabled: bool = False,
        max_gap_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
    ) -> tuple[Sound, list[SilenceGapTrim]]:
        """
        Same as limit_silence_gaps_if_enabled, but also returns a record of every
        gap that was shortened, so callers can remap timings that were measured
        against the un-cut audio.
        """
        if not enabled:
            return sound, []

        new_sound, trims = SilenceUtil.limit_silence_gaps(sound, max_gap_duration)
        if abs(new_sound.duration - sound.duration) > 0.01:
            return new_sound, trims
        return sound, []

    @staticmethod
    def limit_silence_gaps_if_enabled(
        sound: Sound,
        enabled: bool = False,
        max_gap_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
    ) -> Sound:
        return SoundPipeline.limit_silence_gaps_with_trims(
            sound,
            enabled=enabled,
            max_gap_duration=max_gap_duration,
        )[0]

    @staticmethod
    def apply_lava_sr_upsampling(sound: Sound) -> Sound | str:
        from tts_audiobook_tool.model_manager import ModelManager

        upsampler = ModelManager.get_lava_sr_upsampler()
        if upsampler is None:
            return "LavaSR v2 upsampler is not installed"
        result = upsampler.process(sound, denoise=False)
        if isinstance(result, str):
            return result
        return result

    @staticmethod
    def resample_for_app(sound: Sound) -> Sound:
        return SoundUtil.resample_if_necessary(sound, APP_SAMPLE_RATE)

    @staticmethod
    def apply_high_shelf(sound: Sound, high_shelf: HighShelfEq) -> Sound:
        return SoundExtraUtil.high_shelf_eq(
            sound,
            strength=high_shelf.strength,
            boost_start_hz=high_shelf.boost_start_hz,
            q_like=high_shelf.q_like,
        )

    @staticmethod
    def apply_voice_clone_post_processing(sound: Sound) -> Sound:
        """
        App's standard post-processing treatment on voice clone source audio.

        - Trims silence from both ends
        - Resamples to the app's native sample rate. TTS models all resample
          reference audio internally as needed, so no model-specific rate is
          used here.
        - Applies peak normalization

        If the input is entirely silence, the returned Sound's data.size
        will be 0; callers should report that.
        """
        sound = SilenceUtil.trim_silence_ends(sound)[0]
        if sound.data.size == 0:
            return sound
        sound = SoundPipeline.resample_for_app(sound)
        data = SoundUtil.normalize(sound.data, headroom_db=NORMALIZATION_HEADROOM_DB)
        return Sound(data, sound.sr)
