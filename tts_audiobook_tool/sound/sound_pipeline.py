from __future__ import annotations

import bisect
from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.sound.silence_util import SilenceUtil
from tts_audiobook_tool.app_types.phrase import Phrase, Reason
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
    ) -> list[Sound] | str:
        """
        Non-streaming TTS generation paired with app-standard generated-sound
        post-processing.

        Returns processed sounds, including zero-length sounds for fully silent
        outputs. Callers decide how to report silence.
        """
        from tts_audiobook_tool.tts import Tts

        result = Tts.generate_using_project(project, prompts, force_random_seed)
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
        Used by: realtime playback, server non-streaming, and voice chat flows.

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
    def append_pause_or_section_effect(
        sound: Sound,
        reason: Reason,
        reason_pauses: ReasonPauses,
        break_effect: Reason | None,
        pause_duration_override: float | None = None,
    ) -> Sound:
        """
        Appends the trailing break effect or pause for a segment.

        :param break_effect:
            Precomputed decision from `BreakEffectTracker.next_break_effect`:
            which break sound effect to append (SPACE_BREAK or SECTION_BREAK),
            or None to append the pause for ``reason`` instead.
        :param pause_duration_override:
            When not None, uses this duration (in seconds) instead of
            the configured pause for ``reason``. Only affects the pure-silence
            branch; break sound effects ignore it. Used by the concat flow to
            compensate for pseudo-silence already present at segment boundaries.
        """

        if break_effect == Reason.SPACE_BREAK:
            return SoundUtil.append_sound_using_path(sound, SPACE_BREAK_SOUND_EFFECT_PATH)
        if break_effect == Reason.SECTION_BREAK:
            return SoundUtil.append_sound_using_path(sound, SECTION_BREAK_SOUND_EFFECT_PATH)

        pause_duration = (
            pause_duration_override
            if pause_duration_override is not None
            else reason_pauses.get_pause_for(reason)
        )
        if pause_duration > 0:
            return SoundUtil.add_silence(sound, pause_duration)

        return sound

    @staticmethod
    def limit_silence_gaps_if_enabled(
        sound: Sound,
        enabled: bool = False,
        max_gap_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
    ) -> Sound:
        if not enabled:
            return sound

        new_sound, _ = SilenceUtil.limit_silence_gaps(sound, max_gap_duration)
        if abs(new_sound.duration - sound.duration) > 0.01:
            return new_sound
        return sound

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
