from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.phrase import Phrase, Reason
from tts_audiobook_tool.project import Project


class SoundAppUtil:
    """
    """

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
        return [SoundAppUtil.apply_generate_post_processing(sound) for sound in sounds]

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
            result = SoundAppUtil.apply_sidon_upsampling(sound)
            if isinstance(result, str):
                return result
            sound = result

        sound = SoundAppUtil.resample_for_app(sound)
        sound = SoundAppUtil.apply_high_shelf(sound, high_shelf)
        sound = SoundAppUtil.append_pause_or_section_effect(
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
        sound = SoundAppUtil.limit_silence_gaps_if_enabled(
            sound,
            enabled=limit_silence_gaps,
            max_gap_duration=limit_silence_gaps_duration,
        )
        sound = SoundAppUtil.resample_for_app(sound)
        return SoundAppUtil.apply_high_shelf(sound, high_shelf)

# ---

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
        if abs(new_sound.duration - sound.duration) > 0.01: # 'epsilon'
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
        return result # rem, this is 48khz

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
