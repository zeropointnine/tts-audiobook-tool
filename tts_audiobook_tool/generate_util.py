import os
import sys
import time

import librosa
import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_types import FailResult, TrimmableResult, PassResult, ValidationResult
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.whisper_util import WhisperUtil

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

        SoundFileUtil.debug_save_dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR) # not great

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

            if DEV:
                printt(f"VRAM: {AppUtil.get_vram_usage_nv()}")

            if not opt_sound:
                # Model failed to produce audio

                printt(f"{COL_ERROR}Skipped item")
                num_failed += 1

            else:
                # Model generated sound data, will save

                if isinstance(validate_result, TrimmableResult):
                    printt(f"{COL_OK}Fixed:{COL_DEFAULT} {validate_result.get_ui_message()}")

                if isinstance(validate_result, FailResult):
                    printt(f"{COL_ERROR}Max fails reached{COL_DEFAULT}, tagging as failed, will save anyway")
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
        col = COL_ACCENT if num_saved_with_error else ""
        printt(f"Num lines saved, but flagged with potential errors: {col}{num_saved_with_error}")
        if num_failed:
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
            result = GenerateUtil.generate_single(project, text_segment)

            if isinstance(result, str):
                err = result
                printt(f"{err}")
                if pass_num < max_passes:
                    printt(f"{COL_ERROR}Will retry")
                    continue
                else:
                    return None, FailResult(err)
            else:
                sound = result

            # Transcribe
            result = WhisperUtil.transcribe_to_segments(sound)
            if isinstance(result, str):
                # Transcription error (unlikely)
                err = result
                printt(f"{err}")
                if pass_num < max_passes:
                    printt(f"{COL_ERROR}Will retry")
                    continue
                else:
                    return None, FailResult(err)
            segments = result

            transcribed_words = WhisperUtil.get_words_from_segments(segments)

            # temp1 = WhisperUtil.make_aligned_words(sound, segments)

            # Validate
            validation_result = ValidateUtil.validate_item(
                sound, text_segment.text, transcribed_words, Tts.get_type().value
            )

            should_save_debug_info = isinstance(validation_result, TrimmableResult) or isinstance(validation_result, FailResult)
            if should_save_debug_info:
                transcribed_text = WhisperUtil.get_flat_text(transcribed_words)
                SoundFileUtil.debug_save_result_info(validation_result, text_segment.text, transcribed_text)

            if isinstance(validation_result, PassResult):
                return sound, validation_result
            elif isinstance(validation_result, TrimmableResult):
                start_time = validation_result.start_time or 0
                end_time = validation_result.end_time or sound.duration
                new_sound = SoundUtil.trim(sound, start_time, end_time)
                return new_sound, validation_result
            else: # is invalid
                printt(f"{validation_result.get_ui_message()}")
                if pass_num < max_passes:
                    printt(f"{COL_ERROR}Will retry")
                    continue
                else:
                    # Returns sound even though identified as invalid
                    return sound, validation_result

    @staticmethod
    def generate_save_no_validation(
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

        flac_path = SoundSegmentUtil.make_segment_file_path(index, project)
        err = SoundFileUtil.save_flac(sound, flac_path)
        if err:
            return "", err
        else:
            return flac_path, err


    @staticmethod
    def generate_single(
        project: Project,
        text_segment: TextSegment,
        print_info: bool=True
    ) -> Sound | str:
        """
        Core audio generation function.
        Returns model-generated sound data, in model's native samplerate,
        or error string on model-related fail.
        """

        start_time = time.time()

        text = text_segment.text
        text = GenerateUtil.preprocess_text_common(text)
        text = Tts.get_tts_model().preprocess_text(text)

        match Tts.get_type():

            case TtsModelInfos.OUTE:
                result = Tts.get_oute().generate(
                    text,
                    project.oute_voice_json,
                    project.oute_temperature)

            case TtsModelInfos.CHATTERBOX:
                if project.chatterbox_voice_file_name:
                    voice_path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
                else:
                    voice_path = ""
                result = Tts.get_chatterbox().generate(
                    text=text,
                    voice_path=voice_path,
                    exaggeration=project.chatterbox_exaggeration,
                    cfg=project.chatterbox_cfg,
                    temperature=project.chatterbox_temperature
                )

            case TtsModelInfos.FISH:
                if project.fish_voice_file_name:
                    source_path = os.path.join(project.dir_path, project.fish_voice_file_name)
                    Tts.get_fish().set_voice_clone_using(
                        source_path=source_path,
                        transcribed_text=project.fish_voice_transcript
                    )
                else:
                    Tts.get_fish().clear_voice_clone()
                result = Tts.get_fish().generate(text, project.fish_temperature)

            case TtsModelInfos.HIGGS:
                if project.higgs_voice_file_name:
                    voice_path = os.path.join(project.dir_path, project.higgs_voice_file_name)
                    voice_transcript = project.higgs_voice_transcript
                else:
                    voice_path = ""
                    voice_transcript = ""
                if project.higgs_temperature == -1:
                    temperature = HIGGS_DEFAULT_TEMPERATURE
                else:
                    temperature = project.higgs_temperature
                result = Tts.get_higgs().generate(
                    p_voice_path=voice_path, # TODO is this loading every gen?
                    p_voice_transcript=voice_transcript,
                    text=text,
                    seed=random.randint(1, sys.maxsize),
                    temperature=temperature
                )

            case TtsModelInfos.VIBEVOICE:
                voice_path = os.path.join(project.dir_path, project.vibevoice_voice_file_name)
                cfg_scale = VIBEVOICE_DEFAULT_CFG if project.vibevoice_cfg == -1 else project.vibevoice_cfg
                num_steps = VIBEVOICE_DEFAULT_NUM_STEPS if project.vibevoice_steps == -1 else project.vibevoice_steps

                result = Tts.get_vibevoice().generate(
                    text=text,
                    voice_path=voice_path,
                    cfg_scale=cfg_scale,
                    num_steps=num_steps
                )

            case TtsModelInfos.NONE:
                return "No active TTS model"

        if isinstance(result, str):
            return result

        sound = result
        if sound.data.size == 0:
            return "Model output is empty, discarding"

        num_nans = np.sum(np.isnan(sound.data))
        if num_nans > 0:
            return "Model outputted NaN, discarding"

        SoundFileUtil.debug_save("gen", result)

        # Trim "silence" from ends of audio clip
        sound = SilenceUtil.trim_silence(sound)

        # Re-check size
        if sound.data.size == 0:
            return "Model output is silence, discarding"

        # Do peak normalization
        normalized_data = SoundUtil.normalize(sound.data, headroom_db=3.0)
        sound = Sound(normalized_data, sound.sr)

        SoundFileUtil.debug_save("post_process", sound)

        if print_info:
            elapsed = time.time() - start_time or 1.0
            print_speed_info(result.duration, elapsed)

        return sound


    @staticmethod
    def preprocess_text_common(text: str) -> str:
        """
        Transforms text for inference.
        These should be common to (ie, compatible with) any tts model
        """

        text = text.strip()

        # Replace fancy double-quotes (important for Higgs, eg)
        text = text.replace("“", "\"")
        text = text.replace("”", "\"")

        # Collapse consecutive ellipsis chars
        text = re.sub(r'…+', '…', text)

        # Collapse consecutive dots to triple-dot
        text = re.sub(r'\.{4,}', '...', text)

        # Expand "int words" to prevent TTS model from simply saying a string of digits
        text = TextUtil.expand_int_words_in_text(text)

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
        s = f"{COL_ACCENT}Cumulative inference speed including overhead (last {last_n} segments):"
    else:
        s = f"{COL_ACCENT}Cumulative inference speed including overhead (all {len(duration_and_elapsed)} sound segments)"
    printt(s)
    print_speed_info(duration_sum, elapsed_sum)
    printt()
