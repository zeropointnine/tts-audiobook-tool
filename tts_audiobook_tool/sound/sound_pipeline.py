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
        use_section_sound_effect: bool,
        high_shelf: HighShelfEq,
        use_upsampler: bool = False,
    ) -> Sound | str:
        """
        Loads a saved segment file and applies concat/export rendering steps:

        - Applies upsampler (optional)
        - Resamples to 48k
        - Applies high shelf filter (optional)
        - Adds ending silence
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
        sound = SoundPipeline.append_pause_or_section_effect(
            sound,
            reason=phrase.reason,
            use_section_sound_effect=use_section_sound_effect,
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
        use_section_sound_effect: bool,
    ) -> Sound:
        if reason == Reason.SECTION and use_section_sound_effect:
            return SoundUtil.append_sound_using_path(sound, SECTION_SOUND_EFFECT_PATH)
        if reason.pause_duration:
            return SoundUtil.add_silence(sound, reason.pause_duration)
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
    def apply_sidon_upsampling(sound: Sound) -> Sound | str:
        from tts_audiobook_tool.models_util import ModelsUtil

        upsampler = ModelsUtil.get_sidon_upsampler()
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