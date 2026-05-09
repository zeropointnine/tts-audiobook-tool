from tts_audiobook_tool.app_types import HighShelfEq, Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.silence_util import SilenceUtil


class SoundAppUtil:

    @staticmethod
    def apply_generate_post_processing(sound: Sound) -> Sound:
        """
        App's standard post-processing treatment on generated sounds.
        Note that this does *not* touch samplerate.

        Trims silence from both ends and applies peak normalizatio
        
        If the input is entirely silence, the returned Sound's data.size will be 0 
        (Callers must decide how to report that)
        """
        sound = SilenceUtil.trim_silence_ends(sound)[0]
        if sound.data.size == 0:
            return sound
        data = SoundUtil.normalize(sound.data, headroom_db=NORMALIZATION_HEADROOM_DB)
        return Sound(data, sound.sr)



    @staticmethod
    def apply_segment_post_processing(
        sound: Sound,
        high_shelf: HighShelfEq,
        limit_silence_gaps: bool = False,
        limit_silence_gaps_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
        use_upsampler: bool = False,
    ) -> Sound | str:
        """
        Standard audio post-processing chain for generated sound segments.
        
        Generated sound segments are expected to still be in their native samplerate, 
        and are expected to have already been trimmed on the ends for silence and peak normalized.
        
        Used by: concat, realtime playback, and voice chat flows. 

        Order:
          1. Optional silence-gap limiting
          2. Optional generative upsampler (sidon)
          3. Resample to APP_SAMPLE_RATE if not already
          4. Optional High-shelf EQ 

        Returns the processed Sound, or an error string (if upsampling was requested but failed)
        """
        if limit_silence_gaps:
            new_sound = SilenceUtil.limit_silence_gaps(sound, limit_silence_gaps_duration)
            if abs(new_sound.duration - sound.duration) > 0.01: # 'epsilon'
                sound = new_sound

        if use_upsampler:
            from tts_audiobook_tool.models_util import ModelsUtil

            upsampler = ModelsUtil.get_sidon_upsampler()
            assert upsampler
            result = upsampler.process(sound)
            if isinstance(result, str):
                return result
            sound = result # rem, this is 48khz

        sound = SoundUtil.resample_if_necessary(sound, APP_SAMPLE_RATE)

        sound = SoundExtraUtil.high_shelf_eq(
            sound,
            strength=high_shelf.strength,
            boost_start_hz=high_shelf.boost_start_hz,
            q_like=high_shelf.q_like,
        )

        return sound

