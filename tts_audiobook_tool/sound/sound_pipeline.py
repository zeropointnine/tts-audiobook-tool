from __future__ import annotations

from typing import TYPE_CHECKING

from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.sound.silence_util import SilenceUtil
from tts_audiobook_tool.app_types.phrase import Phrase, Reason

if TYPE_CHECKING:
    from tts_audiobook_tool.project import Project


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
        use_break_sound_effect: bool,
        high_shelf: HighShelfEq,
        is_first_in_section: bool = False,
        use_upsampler: bool = False,
        add_pause: bool = True,
    ) -> Sound | str:
        """
        Loads a saved segment file and applies concat/export rendering steps:

        - Applies upsampler (optional)
        - Resamples to 48k
        - Applies high shelf filter (optional)
        - Adds ending silence (unless `add_pause` is False)

        When `add_pause` is False, the trailing pause/section effect is omitted
        and the caller is responsible for appending it (e.g. via
        `append_pause_or_section_effect`). This is used by the concat flow so
        that the pause duration can be adjusted based on pseudo-silence measured
        across adjacent segments.
        """

        result = SoundFileUtil.load(path)
        if isinstance(result, str):
            return result

        sound = result

        if use_upsampler:
            result = SoundPipeline.apply_sidon_upsampling(sound)
            if isinstance(result, str):
                return result
            sound = result

        sound = SoundPipeline.resample_for_app(sound)

        sound = SoundPipeline.apply_high_shelf(sound, high_shelf)

        if add_pause:
            sound = SoundPipeline.append_pause_or_section_effect(
                sound,
                reason=phrase.reason,
                use_break_sound_effect=use_break_sound_effect,
                is_first_in_section=is_first_in_section,
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
        use_break_sound_effect: bool,
        is_first_in_section: bool = False,
        pause_duration_override: float | None = None,
    ) -> Sound:
        """
        Appends the trailing pause or section break sound effect for a segment.

        :param pause_duration_override:
            When not None, uses this duration (in seconds) instead of
            `reason.pause_duration` for the inserted silence. Only affects the
            pure-silence branch; break sound effects ignore it. Used by the
            concat flow to compensate for pseudo-silence already present at
            segment boundaries.
        """

        b = SoundPipeline.should_append_break_sound_effect(
                reason,
                use_break_sound_effect=use_break_sound_effect,
                is_first_in_section=is_first_in_section,
        )
        if b:
            if reason == Reason.SPACE_BREAK:
                return SoundUtil.append_sound_using_path(sound, SPACE_BREAK_SOUND_EFFECT_PATH)
            elif reason == Reason.SECTION_BREAK:
                return SoundUtil.append_sound_using_path(sound, SECTION_BREAK_SOUND_EFFECT_PATH)

        pause_duration = (
            pause_duration_override
            if pause_duration_override is not None
            else reason.pause_duration
        )
        if pause_duration > 0:
            return SoundUtil.add_silence(sound, pause_duration)

        return sound

    @staticmethod
    def should_append_break_sound_effect(
        reason: Reason,
        use_break_sound_effect: bool,
        is_first_in_section: bool = False,
    ) -> bool:
        if not use_break_sound_effect:
            return False
        if reason == Reason.SPACE_BREAK and is_first_in_section:
            return False
        return reason in [Reason.SPACE_BREAK, Reason.SECTION_BREAK]

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
    def apply_sidon_upsampling(sound: Sound) -> Sound | str:
        from tts_audiobook_tool.model_manager import ModelManager

        upsampler = ModelManager.get_sidon_upsampler()
        assert upsampler
        result = upsampler.process(sound)
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
    def apply_voice_clone_post_processing(sound: Sound, target_sr: int) -> Sound:
        """
        App's standard post-processing treatment on voice clone source audio.

        - Resamples to the model's native sample rate
        - Applies peak normalization
        """
        sound = SoundUtil.resample_if_necessary(sound, target_sr)
        data = SoundUtil.normalize(sound.data, headroom_db=NORMALIZATION_HEADROOM_DB)
        return Sound(data, sound.sr)