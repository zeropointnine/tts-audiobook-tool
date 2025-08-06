import os
import sys
import time
import copy

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_info import TtsType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_sound_segments import *
from tts_audiobook_tool.validate_util import ValidateResult, ValidateResultType, ValidateUtil

class GenerateUtil:

    @staticmethod
    def generate_items_to_files(
            project: Project,
            indices_to_generate: set[int],
            items_to_regenerate: dict[int, str]
    ) -> bool:
        """
        Subroutine for doing multiple audio generations

        indices_to_generate and items_to_regenerate are mutually exclusive

        "Regenerate" here simply means that the currently existing, failed file
        for the given index will be deleted first.

        Prints feedback at end of each item

        Returns True if ended because interrupted
        """

        # not great
        SoundFileUtil.debug_save_dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

        if indices_to_generate:
            is_regenerate = False
            # Convert list to dict for compatibility
            items = {}
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

            opt_sound, validate_result = GenerateUtil.generate_sound_full(project, text_segment, is_regen=is_regenerate)

            s = validate_result.message
            if validate_result.result == ValidateResultType.TRIMMABLE:
                s = f"Fixed: {s}"
            printt(s)

            if opt_sound:
                is_invalid = (validate_result.result == ValidateResultType.INVALID)
                if is_invalid:
                    printt(f"{COL_ERROR}Max fails reached{COL_DEFAULT}, tagging as failed but will save anyway")
                    num_saved_with_error += 1
                else:
                    num_saved_ok += 1

                path = SoundSegmentUtil.make_segment_file_path(i, project)
                if is_invalid:
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

            else:
                printt(f"{COL_ERROR}Skipped item")
                num_failed += 1

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
    def generate_sound_full(
        project: Project,
        text_segment: TextSegment,
        max_passes: int = 2,
        is_regen: bool = False
    ) -> tuple[Sound | None, ValidateResult]:
        """
        Prints error feedback only on non-final generation fail
        """

        def print_will_regenerate(error: str):
            printt(f"{error}")
            printt(f"{COL_ERROR}Will regenerate")

        pass_num = 0
        while True:
            pass_num += 1

            # Generate
            o = GenerateUtil.generate(project, text_segment, is_regen)
            if isinstance(o, str):
                # Failed to generate
                if pass_num < max_passes:
                    print_will_regenerate(o)
                    continue
                else:
                    return None, ValidateResult(ValidateResultType.INVALID, o)
            else:
                sound = o

            # Post process
            sound = GenerateUtil.post_process(sound)

            # Transcribe
            o = SoundUtil.transcribe(sound)
            if isinstance(o, str):
                if pass_num < max_passes:
                    print_will_regenerate(o)
                    continue
                else:
                    return None, ValidateResult(ValidateResultType.INVALID, o)
            else:
                whisper_data = o

            # Validate
            validate_result = ValidateUtil.validate_item(
                sound, text_segment.text, whisper_data,Tts.get_type().value
            )
            match validate_result.result:
                case ValidateResultType.VALID:
                    return sound, validate_result
                case ValidateResultType.TRIMMABLE:
                    start_time = validate_result.trim_start or 0
                    end_time = validate_result.trim_end or sound.duration
                    new_sound = SoundUtil.trim(sound, start_time, end_time)
                    return new_sound, validate_result
                case  ValidateResultType.INVALID:
                    if pass_num < max_passes:
                        print_will_regenerate(validate_result.message)
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
        sound = GenerateUtil.generate(project, text_segment)
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

        # Trim all silence from ends of audio clip
        sound = SilenceUtil.trim_silence(sound)
        SoundFileUtil.debug_save("after trim silence", sound)

        # Prevent 0-byte audio data as a failsafe
        if len(sound.data) == 0:
            sound = sound._replace(data=np.array([0], dtype=sound.data.dtype))

        return sound


    @staticmethod
    def generate(
        project: Project,
        text_segment: TextSegment,
        is_regen: bool=False,
        print_info: bool=True
    ) -> Sound | str:
        """
        Returns model-generated sound data, in model's native samplerate.
        """

        start_time = time.time()

        text = GenerateUtil._preprocess_text(text_segment.text)

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
                seed = DEFAULT_SEED if not is_regen else random.randint(1, sys.maxsize)
                sound = Tts.get_higgs().generate(
                    p_voice_path=os.path.join(project.dir_path, project.higgs_voice_file_name),
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
    def generate_kyutai(
        prompt: str,
        project: Project
    ) -> Sound | None:

        ...

        # kyutai = Tts.get_fish()

        # entries = kyutai.prepare_script([prompt], padding_between=1)

        # # male midwestern twang
        # # voice = "expresso/ex03-ex01_happy_001_channel1_334s.wav"
        # # female narration style somewhat
        # voice = "expresso/ex04-ex01_narration_001_channel1_605s.wav"

        # voice_path = kyutai.get_voice_path(voice)
        # condition_attributes = kyutai.make_condition_attributes(
        #     [voice_path], cfg_coef=2.0
        # )

        # pcms = []
        # def _on_frame(frame):
        #     print("Step", len(pcms), end="\r")
        #     if (frame != -1).all():
        #         pcm = kyutai.mimi.decode(frame[:, 1:, :]).cpu().numpy()
        #         pcms.append(np.clip(pcm[0, 0], -1, 1))

        # all_entries = [entries]
        # all_condition_attributes = [condition_attributes]
        # with kyutai.mimi.streaming(len(all_entries)):
        #     _ = kyutai.generate(all_entries, all_condition_attributes, on_frame=_on_frame)

        # audio = np.concatenate(pcms, axis=-1)

        # del kyutai
        # return Sound(audio, 24000)

    @staticmethod
    def _preprocess_text(text: str) -> str:
        """
        Transform text before passing it off as a prompt for the TTS model.
        """

        if Tts.get_type().value.em_dash_replace:
            text = text.replace("—", Tts.get_type().value.em_dash_replace)

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
