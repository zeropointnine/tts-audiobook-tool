import os
import sys
import time
import copy

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types import FailResult, TrimmableResult, PassResult, ValidationResult
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.transcribe_util import TranscribeUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_sound_segments import *
from tts_audiobook_tool.validate_util import ValidateUtil

class GenerateUtil:

    @staticmethod
    def generate_items_to_files(
            project: Project,
            indices_to_generate: set[int],
            items_to_regenerate: dict[int, str]
    ) -> bool:
        """
        Subroutine for doing a series of audio generations

        :param indices_to_generate:

        :param items_to_regenerate:
            is mutually exclusive to indices_to_generate
            "Regenerate" here simply means that the currently existing, failed file
            for the given index will be deleted first.

        Prints feedback at end of each item

        Returns True if ended because interrupted
        """

        # not great
        SoundFileUtil.debug_save_dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

        if indices_to_generate:
            is_regenerate = False
            items = {} # Convert list to dict for compatibility
            for index in indices_to_generate:
                items[index] = ""
        else:
            is_regenerate = True
            items = items_to_regenerate

        Tts.warm_up_models()

        SigIntHandler().set("generating")
        start_time = time.time()
        count = 0
        num_saved_ok = 0
        num_saved_with_error = 0
        num_failed = 0

        saved_start_time = time.time()
        saved_duration_and_elapsed: list[ tuple[float, float] ] = []

        for i, path in sorted(items.items()):

            text_segment = project.text_segments[i]

            printt()
            print_item_heading(
                is_regenerate, text_segment.text, i, count, len(items)
            )

            if is_regenerate: # First, delete original
                if path and os.path.exists(path):
                    delete_silently(path)

            opt_sound, validate_result = GenerateUtil.generate_sound_full_flow(project, text_segment)

            if not opt_sound:
                # Model failed to produce audio

                printt(f"{COL_ERROR}Skipped item")
                num_failed += 1

            else:
                # Model generated sound data, will save

                if isinstance(validate_result, TrimmableResult):
                    printt(f"{COL_OK}Fixed:{COL_DEFAULT} {validate_result.get_ui_message()}")

                if isinstance(validate_result, FailResult):
                    printt(f"{COL_ERROR}Max fails reached{COL_DEFAULT}, tagging as failed but will save anyway")
                    num_saved_with_error += 1
                else:
                    num_saved_ok += 1

                # Save file
                path = SoundSegmentUtil.make_segment_file_path(i, project)
                if isinstance(validate_result, FailResult):
                    path = AppUtil.insert_bracket_tag_file_path(path, "fail")

                err = SoundFileUtil.save_flac(opt_sound, path)
                if err:
                    printt(f"{COL_ERROR}Couldn't save file: {path}")
                else:
                    printt(f"Saved file: {path}")
                    saved_duration_and_elapsed.append( (opt_sound.duration, time.time() - saved_start_time) )
                    saved_start_time = time.time()
                    if len(saved_duration_and_elapsed) % 100 == 0:
                        print_cumulative_speed_info(saved_duration_and_elapsed, 100)

            count += 1

            if SigIntHandler().did_interrupt:
                break

        did_interrupt = SigIntHandler().did_interrupt
        SigIntHandler().clear()

        printt(f"Elapsed: {duration_string(time.time() - start_time)}")
        printt()

        printt(f"Num lines saved normally: {COL_OK}{num_saved_ok}")
        printt(f"Num lines saved, but flagged with potential errors: {COL_ACCENT}{num_saved_with_error}")
        if num_failed > 0:
            printt(f"Num lines failed to generate: {COL_ERROR}{num_failed}")
        printt()

        return did_interrupt

    @staticmethod
    def generate_sound_full_flow(
        project: Project,
        text_segment: TextSegment,
        max_passes: int = 2
    ) -> tuple[Sound | None, ValidationResult]:
        """
        Full program flow for generating sound for a text segment, including retries
        Prints error feedback only on non-final generation fail
        """

        pass_num = 0
        while True:
            pass_num += 1

            # Generate
            o = GenerateUtil.generate_single(project, text_segment)

            if isinstance(o, str):
                # Failed to generate
                err = o
                printt(f"{COL_ERROR}{err}")
                if pass_num < max_passes:
                    printt(f"{COL_ERROR}Will retry")
                    continue
                else:
                    return None, FailResult(err)
            else:
                sound = o

            # Post process generated audio
            sound = GenerateUtil.post_process(sound)

            # Transcribe generated audio
            o = SoundUtil.transcribe(sound)
            if isinstance(o, str):
                # Transcription error (is unlikely)
                err = o
                printt(f"{COL_ERROR}{err}")
                if pass_num < max_passes:
                    printt(f"{COL_ERROR}Will retry")
                    continue
                else:
                    return None, FailResult(err)
            else:
                whisper_data = o

            # Validate
            validate_result = ValidateUtil.validate_item(
                sound, text_segment.text, whisper_data,Tts.get_type().value
            )

            should_save_debug_info = isinstance(validate_result, TrimmableResult) or isinstance(validate_result, FailResult)
            if should_save_debug_info:
                transcribed_text = TranscribeUtil.get_whisper_data_text(whisper_data)
                SoundFileUtil.debug_save_result_text(validate_result, text_segment.text, transcribed_text)

            if isinstance(validate_result, PassResult):
                return sound, validate_result
            elif isinstance(validate_result, TrimmableResult):
                start_time = validate_result.start_time or 0
                end_time = validate_result.end_time or sound.duration
                new_sound = SoundUtil.trim(sound, start_time, end_time)
                return new_sound, validate_result
            else: # is invalid
                printt(f"{COL_ERROR}{validate_result.get_ui_message()}")
                if pass_num < max_passes:
                    printt(f"{COL_ERROR}Will retry")
                    continue
                else:
                    # Returns sound even though identified as invalid
                    return sound, validate_result

    @staticmethod
    def generate_post_process_save(
        index: int,
        project: Project
    ) -> tuple[str, str]:
        """
        Generates, does post process step and saves
        Ie, skips error detect / error fix steps.

        Returns saved file path, error string
        """

        text_segment = project.text_segments[index]
        sound = GenerateUtil.generate_single(project, text_segment)
        if isinstance(sound, str):
            return "", f"Couldn't generate audio clip: {sound}"

        sound = GenerateUtil.post_process(sound)

        flac_path = SoundSegmentUtil.make_segment_file_path(index, project)
        err = SoundFileUtil.save_flac(sound, flac_path)
        if err:
            return "", err
        else:
            return flac_path, err

    @staticmethod
    def post_process(sound: Sound) -> Sound:

        # Prevent 0-byte audio data as a failsafe
        if len(sound.data) == 0:
            sound = sound._replace(data=np.array([0], dtype=sound.data.dtype))

        # Trim all silence from ends of audio clip
        sound = SilenceUtil.trim_silence(sound)
        # SoundFileUtil.debug_save("after trim silence", sound)

        # Prevent 0-byte audio data as a failsafe again
        if len(sound.data) == 0:
            sound = sound._replace(data=np.array([0], dtype=sound.data.dtype))

        return sound

    @staticmethod
    def generate_single(
        project: Project,
        text_segment: TextSegment,
        print_info: bool=True
    ) -> Sound | str:
        """
        Returns model-generated sound data, in model's native samplerate.
        """

        start_time = time.time()

        text = GenerateUtil.preprocess_text(text_segment.text)

        match Tts.get_type():

            case TtsType.OUTE:
                sound = GenerateUtil.generate_oute(
                    text,
                    project.oute_voice_json,
                    project.oute_temperature)

            case TtsType.CHATTERBOX:
                sound = GenerateUtil.generate_chatterbox(text, project)

            case TtsType.FISH:
                if project.fish_voice_file_name:
                    Tts.get_fish().set_voice_clone_using(
                        source_path=os.path.join(project.dir_path, project.fish_voice_file_name),
                        transcribed_text=project.fish_voice_transcript
                    )
                else:
                    Tts.get_fish().clear_voice_clone()

                sound = Tts.get_fish().generate(text, project.fish_temperature)

            case TtsType.HIGGS:
                seed = random.randint(1, sys.maxsize)
                if project.higgs_voice_file_name:
                    voice_path = os.path.join(project.dir_path, project.higgs_voice_file_name)
                else:
                    voice_path = None
                sound = Tts.get_higgs().generate(
                    p_voice_path=voice_path,
                    p_voice_transcript=project.higgs_voice_transcript,
                    text=text,
                    seed=seed,
                    temperature=project.higgs_temperature)

            case TtsType.NONE:
                return "No active TTS model"

        if isinstance(sound, str):
            return sound
        if not sound:
            return "Model failed"

        num_nans = np.sum(np.isnan(sound.data))
        if num_nans > 0:
            return "Model outputted NaN, discarding"

        if print_info:
            elapsed = time.time() - start_time or 1.0
            print_speed_info(sound.duration, elapsed)

        SoundFileUtil.debug_save("after gen", sound)

        return sound

    @staticmethod
    def generate_oute(
        prompt: str,
        voice: dict,
        temperature: float = -1
    ) -> Sound | None:
        """ Returns sound or None on fail """

        from loguru import logger
        logger.remove()

        # First, clone GENERATION_CONFIG from config file
        from outetts.models.config import SamplerConfig # type: ignore
        from tts_audiobook_tool.config_oute import GENERATION_CONFIG
        try:
            from .config_oute_dev import GENERATION_CONFIG
        except ImportError:
            pass
        gen_config = copy.deepcopy(GENERATION_CONFIG)

        gen_config.text = prompt
        gen_config.speaker = voice
        if temperature != -1:
            gen_config.sampler_config = SamplerConfig(temperature)

        try:
            output = Tts.get_oute().generate(config=gen_config)
            audio = output.audio.cpu().clone().squeeze().numpy()
            return Sound(audio, output.sr)

        except Exception as e:
            printt(f"{COL_ERROR}Oute model error: {e}")
            return None

    @staticmethod
    def generate_chatterbox(
        prompt: str,
        project: Project
    ) -> Sound | None:
        """ Returns sound or None on fail """
        d = {}
        if project.chatterbox_voice_file_name:
            # Rem, this is actually optional
            path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
            d["audio_prompt_path"] = path
        if project.chatterbox_exaggeration != -1:
            d["exaggeration"] = project.chatterbox_exaggeration
        if project.chatterbox_cfg != -1:
            d["cfg_weight"] = project.chatterbox_cfg
        if project.chatterbox_temperature != -1:
            d["temperature"] = project.chatterbox_temperature

        try:
            data = Tts.get_chatterbox().generate(prompt, **d)
            data = data.numpy().squeeze()
            data = SoundUtil.normalize(data, headroom_db=1.0)
            return Sound(data, Tts.get_chatterbox().sr)

        except Exception as e:
            print(f"{COL_ERROR}Chatterbox model error: {e}\a", "error")
            return None

    @staticmethod
    def preprocess_text(text: str) -> str:
        """
        Transform text right before passing it off to the TTS model for audio generation.
        """

        if Tts.get_type().value.em_dash_replace:
            text = text.replace("—", Tts.get_type().value.em_dash_replace)

        # Replace fancy double-quotes (important for Higgs)
        text = text.replace("“", "\"")
        text = text.replace("”", "\"")

        # Limited case where free-standing paragraph is simply a number (eg, chapter headings)
        text = TextUtil.number_string_to_words(text)
        return text

# ---

def print_item_heading(is_regenerate: bool, text: str, index: int, count: int, total: int) -> None:
    verb = "Regenerating" if is_regenerate else "Generating"
    s  = f"{COL_ACCENT}[{COL_DEFAULT}{count+1}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
    s += f"{COL_ACCENT}{verb} audio for text segment {COL_DEFAULT}{index+1}{COL_ACCENT}:{COL_DEFAULT}"
    printt(s)
    printt(f"{COL_DIM}{Ansi.ITALICS}{text.strip()}")
    printt()

def print_speed_info(sound_duration: float, elapsed: float) -> None:
    multi = sound_duration / elapsed
    s = f"Audio duration: {COL_ACCENT}{sound_duration:.1f}s{COL_DEFAULT}, inference time: {COL_ACCENT}{elapsed:.1f}s"
    s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
    printt(s)

def print_cumulative_speed_info(duration_and_elapsed: list[tuple[float, float]], last_n=0) -> None:
    if last_n:
        duration_and_elapsed = duration_and_elapsed[-last_n:]
    duration_sum = 0
    elapsed_sum = 0
    for duration, elapsed in duration_and_elapsed:
        duration_sum += duration
        elapsed_sum += elapsed
    printt()
    if last_n:
        s = f"{COL_ACCENT}Cumulative inference speed including overhead (last {last_n} saved sound segments):"
    else:
        s = f"{COL_ACCENT}Cumulative inference speed including overhead (all {len(duration_and_elapsed)} saved sound segments)"
    printt(s)
    print_speed_info(duration_sum, elapsed_sum)
    printt()
