import os
import sys
import time

import numpy as np

from tts_audiobook_tool.app_types import SkippedResult, Sound, SttConfig, SttVariant
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
from tts_audiobook_tool.tts_model import ChatterboxProtocol, HiggsModelProtocol, VibeVoiceProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.whisper_util import WhisperUtil
from num2words import num2words

class GenerateUtil:

    @staticmethod
    def generate_items_to_files(
            project: Project,
            indices_to_generate: set[int],
            items_to_regenerate: dict[int, str],
            stt_variant: SttVariant,
            stt_config: SttConfig
    ) -> bool:
        """
        Subroutine for doing a series of audio generations to files

        :param indices_to_generate:

        :param items_to_regenerate:
            is mutually exclusive to indices_to_generate
            "Regenerate" here simply means that the currently existing, failed file
            for the given index will be deleted first.

        Prints feedback at end of each item, and summary at end of loop.

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
        num_saved_ok = 0
        num_saved_with_error = 0
        num_failed = 0
        saved_elapsed: list[float] = [] # history of end-to-end time taken to create sound files

        count = 0
        saved_start_time = time.time()

        for segment_index, path in sorted(items.items()):

            text_segment = project.text_segments[segment_index]

            printt()
            GenerateUtil.print_item_heading(
                is_regenerate, text_segment.text, segment_index, count, len(items)
            )

            if is_regenerate: # First, delete original
                if path and os.path.exists(path):
                    delete_silently(path)

            result = GenerateUtil.generate_sound_full_flow(
                project=project,
                text_segment=text_segment,
                stt_variant=stt_variant,
                stt_config=stt_config
            )

            if isinstance(result, str):
                err = result
                printt(f"{COL_ERROR}Model fail, skipping item:")
                printt(f"{COL_ERROR}{err}")
                num_failed += 1

            else:
                # Model successfully generated audio
                sound, validation_result = result

                # Print validation info
                s = "Speech-to-text validation: "
                if isinstance(validation_result, SkippedResult):
                    num_saved_ok += 1
                    s += "Skipped (Whisper disabled)"
                if isinstance(validation_result, PassResult):
                    num_saved_ok += 1
                    s += "Passed"
                elif isinstance(validation_result, TrimmableResult):
                    num_saved_ok += 1
                    s += f"{COL_OK}Fixed{COL_DEFAULT} - {validation_result.get_ui_message()}"
                elif isinstance(validation_result, FailResult):
                    num_saved_with_error += 1
                    s += f"{COL_ERROR}Max fails reached{COL_DEFAULT}, tagging as failed, will save anyway"
                else: # shouldn't get here
                    num_saved_ok += 1
                printt(s)

                # Save file
                path = SoundSegmentUtil.make_segment_file_path(segment_index, project)
                if isinstance(validation_result, FailResult):
                    path = AppUtil.insert_bracket_tag_file_path(path, "fail")
                err = SoundFileUtil.save_flac(sound, path)
                if err:
                    printt(f"{COL_ERROR}Couldn't save file: {path}")
                else:
                    printt(f"Saved: {path}")

                    # Update meta-timing info and print out
                    saved_elapsed.append(time.time() - saved_start_time)
                    saved_start_time = time.time()
                    print_eta(saved_elapsed, len(items) - count + 1)

            count += 1

            if SigIntHandler().did_interrupt:
                break

        # ---

        did_interrupt = SigIntHandler().did_interrupt
        SigIntHandler().clear()

        # Print summary
        s = "\n"
        if did_interrupt:
            s += "Interrupted. "
        s += f"Elapsed: {duration_string(time.time() - start_time)}\n"
        s += "\n"
        ok = str(num_saved_ok)
        if num_saved_ok == len(items.items()):
            ok += " (all)"
        s += f"Num lines saved: {COL_OK}{ok}{COL_DEFAULT}\n"
        col = COL_ACCENT if num_saved_with_error else ""
        if num_saved_with_error:
            s += f"Num lines saved, but flagged with potential errors: {col}{num_saved_with_error}{COL_DEFAULT}\n"
        if num_failed:
            s += f"Num lines failed to generate: {COL_ERROR}{num_failed}{COL_DEFAULT}"
        print(s)

        return did_interrupt

    @staticmethod
    def generate_sound_full_flow(
        project: Project,
        text_segment: TextSegment,
        stt_variant: SttVariant,
        stt_config: SttConfig,
        max_passes: int = 2,
    ) -> tuple[Sound, ValidationResult] | str:
        """
        All TTS inference should be done through here.

        Full program flow for generating sound for a single text segment, including retries.

        max_passes:
            0: skips validation
            1: does validation, does not retry on fail
            2+: does validation, retries up to n-1 times on fail

        stt_variant:
            If DISABLED, skips validation, ofc

        Prints: Generation speed info, and error feedback on non-final generation fail.

        Returns either (Sound, ValidationResult) or error string for model failure etc.
        """

        pass_num = 0
        while True:
            pass_num += 1

            start_time = time.time()




            # Generate
            print(COL_DIM, end="", flush=True) # Dim color during inference printouts
            result = GenerateUtil.generate_single(project, text_segment)
            printt() # Restore print color, print blank line

            if isinstance(result, str):
                err = result
                return err

            sound = result

            elapsed = time.time() - start_time or 1.0
            GenerateUtil.print_speed_info(result.duration, elapsed)

            if max_passes == 0 or stt_variant == SttVariant.DISABLED:
                return (sound, SkippedResult())

            # Transcribe
            result = WhisperUtil.transcribe_to_segments(sound, stt_variant, stt_config)
            if isinstance(result, str):
                err = result
                return err
            segments = result

            transcribed_words = WhisperUtil.get_words_from_segments(segments)

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
                    printt()
                    continue
                else:
                    # Returns sound even though identified as invalid
                    return sound, validation_result

    @staticmethod
    def generate_single(project: Project, text_segment: TextSegment) -> Sound | str:
        """
        Core audio generation function.
        Returns model-generated sound data (in model's native samplerate),
        or error string
        """

        text = text_segment.text
        language_code = project.chatterbox_language
        text = GenerateUtil.preprocess_text_common(text, language_code)
        text = Tts.get_instance().preprocess_text(text)

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
                    temperature = HiggsModelProtocol.DEFAULT_TEMPERATURE
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
                cfg_scale = VibeVoiceProtocol.DEFAULT_CFG if project.vibevoice_cfg == -1 else project.vibevoice_cfg
                num_steps = VibeVoiceProtocol.DEFAULT_NUM_STEPS if project.vibevoice_steps == -1 else project.vibevoice_steps

                result = Tts.get_vibevoice().generate(
                    text=text,
                    voice_path=voice_path,
                    cfg_scale=cfg_scale,
                    num_steps=num_steps
                )

            case TtsModelInfos.INDEXTTS2:

                voice_path = os.path.join(project.dir_path, project.indextts2_voice_file_name)

                if project.indextts2_emo_voice_file_name:
                    emo_voice_path = os.path.join(project.dir_path, project.indextts2_emo_voice_file_name)
                else:
                    emo_voice_path = ""

                result = Tts.get_indextts2().generate(
                    text=text,
                    voice_path=voice_path,
                    temperature=project.indextts2_temperature,
                    emo_alpha=project.indextts2_emo_alpha,
                    emo_voice_path=emo_voice_path,
                    emo_vector=project.indextts2_emo_vector
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

        return sound


    @staticmethod
    def preprocess_text_common(text: str, language_code: str = ChatterboxProtocol.DEFAULT_LANGUAGE) -> str:
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
        # text = TextUtil.expand_int_words_in_text(text)
        try:
            # This replaces the need for the old 'expand_int_words' entirely
            text = re.sub(r'\d+', lambda x: num2words(int(x.group()), lang=language_code), text)
        except NotImplementedError:
            # Fallback for languages num2words doesn't support
            pass

        return text

    @staticmethod
    def print_item_heading(is_regenerate: bool, text: str, index: int, count: int, total: int) -> None:
        verb = "Regenerating" if is_regenerate else "Generating"
        s  = f"{COL_ACCENT}[{COL_DEFAULT}{count+1}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
        s += f"{COL_ACCENT}{verb} audio for text segment {COL_DEFAULT}{index+1}{COL_ACCENT}:{COL_DEFAULT}"
        printt(s)
        printt(f"{COL_DIM}{Ansi.ITALICS}{text.strip()}")
        printt()

    @staticmethod
    def print_speed_info(sound_duration: float, elapsed: float) -> None:
        multi = sound_duration / elapsed
        s = f"Audio duration: {COL_ACCENT}{sound_duration:.1f}s{COL_DEFAULT}, inference time: {COL_ACCENT}{elapsed:.1f}s"
        s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
        printt(s)



# ---

def print_eta(saved_elapsed: list[float], num_left) -> None:

    MIN_SAMPLES = 5
    if len(saved_elapsed) < MIN_SAMPLES or num_left <= 0:
        return

    MAX_SAMPLES = 50
    i_start = len(saved_elapsed) - MAX_SAMPLES
    i_start = max(i_start, 0)
    i_end = len(saved_elapsed)
    sum = 0
    for i in range(i_start, i_end):
        elapsed = saved_elapsed[i]
        sum += elapsed
    num_samples = i_end - i_start
    avg = sum / num_samples
    eta = num_left * avg
    printt(f"Est. time remaining: {duration_string(eta)}")
