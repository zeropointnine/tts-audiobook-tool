import os
import time
import copy

from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.loudness_util import LoudnessUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_dir_util import *
from tts_audiobook_tool.validate_util import ValidateItem, ValidateResult, ValidateUtil

class GenerateUtil:

    @staticmethod
    def generate_validate_fix_items(
            state: State,
            indices: list[int],
            mode: str
    ) -> None:
        """
        mode:
            "generate"
            "generate-and-fix"
            "validate-and-fix": presupposes pre-existing files, skips the first of two potential generate passes
        """

        if not mode in ["generate", "generate-and-fix", "validate-and-fix"]:
            raise ValueError("Bad value for mode")

        # TODO: Duplicate work. Need to figure out best way to pass index-and-path through this whole feature, with all its permutations. Applies to entire app actually.
        index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)

        start_time = time.time()

        count = 1
        Shared.mode = "generating"

        for i in indices:

            print_item_heading(state.project.text_segments[i].text, i, count, len(indices))

            if mode == "generate":
                _ = GenerateUtil.generate_and_make_flac(
                        index=i, state=state, is_retry=False)
            elif mode == "generate-and-fix":
                _ = GenerateUtil.generate_validate_fix_item(
                        index=i, state=state, whisper_model=Shared.get_whisper())
            else: # == "validate-and-fix"
                path = index_to_path[i]
                _ = GenerateUtil.generate_validate_fix_item(
                        index=i, state=state, whisper_model=Shared.get_whisper(), skip_generate_file_path=path)

            count += 1
            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                break

        printt(f"Elapsed: {time_string(time.time() - start_time)}")
        printt()

        Shared.clear_whisper()

        ask("Press enter: ")

    @staticmethod
    def generate_validate_fix_item(
        index: int,
        state: State,
        whisper_model,
        skip_generate_file_path: str=""
    ) -> None:
        """
        skip_generate_file_path
            is used to "validate pre-existing item and fix" (as opposed to "generate, validate and fix")
        """

        pass_num = 1

        while True:

            # Make audio file
            if not skip_generate_file_path:
                file_path = GenerateUtil.generate_and_make_flac(index=index, state=state, is_retry=(pass_num > 1))
                if not file_path:
                    # Unexpected error making audio file
                    break
            else:
                # Validate-and-then-fix
                file_path = skip_generate_file_path
                skip_generate_file_path = ""

            # Validate audio file
            item = ValidateItem(index, file_path, state.project.text_segments[index].text)
            should_fix_delete = (pass_num == 1)
            result, message = ValidateUtil.validate_item(item, should_fix_delete, whisper_model)

            if pass_num == 1:

                match result:
                    case ValidateResult.VALIDATED_AND_TAGGED:
                        printt("Validated" + "\n")
                        break
                    case ValidateResult.FAILED_AND_CORRECTED:
                        printt(message + "\n" + "Corrected" + "\n")
                        break
                    case ValidateResult.FAILED_AND_DELETED:
                        printt(message + "\n" + "Will regenerate" + "\n")
                        pass_num = 2
                        continue
                    case _:
                        L.e("Shouldn't get here")
                        break

            else: # pass_num == 2:

                match result:
                    case ValidateResult.VALIDATED_AND_TAGGED:
                        printt("Validated on second try" + "\n")
                        break
                    case ValidateResult.FAILED_AND_CORRECTED:
                        printt(message + "\n" + "Corrected on second try" + "\n")
                        break
                    case ValidateResult.FAILED_ONLY:
                        printt(message + "\n" + "Voice line still fails test" + "\n")
                        break
                    case _:
                        L.e("Shouldn't get here")
                        break

        # ...

    @staticmethod
    def generate_and_make_flac(
        index: int,
        state: State,
        is_retry: bool=False
    ) -> str:
        """
        Generates temp wav file, converts to flac, deletes temp wav
        Prints info
        Returns file path on success, else empty string

        # TODO: should return (path, error), mutually exclusive
        """

        if not state.project.voice:
            return ""

        temp_wav_path = os.path.join(state.prefs.project_dir, make_random_hex_string() + ".wav")
        flac_path = HashFileUtil.make_segment_file_path(index, state)
        text_segment = state.project.text_segments[index]

        start_time = time.time()
        is_success = GenerateUtil.generate_wav_file(temp_wav_path, text_segment.text, state.project.voice, state.prefs.temperature)
        if not is_success:
            delete_temp_file(temp_wav_path)
            return ""

        # Print time info
        elapsed = time.time() - start_time
        audio_seconds = estimated_wav_seconds(temp_wav_path)
        s = f"File duration: {COL_ACCENT}{audio_seconds:.1f}s{COL_DEFAULT}, inference time: {COL_ACCENT}{elapsed:.1f}s"
        if elapsed > 0:
            multi = audio_seconds / elapsed
            s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
        printt(s)
        printt()

        # Normalize
        if state.prefs.should_normalize:
            printt("Performing loudness normalization")
            printt()
            temp_normalized_wav_path = os.path.join(state.prefs.project_dir, make_random_hex_string() + ".wav")
            err = LoudnessUtil.normalize(temp_wav_path, temp_normalized_wav_path)
            if err:
                delete_temp_file(temp_wav_path)
                delete_temp_file(temp_normalized_wav_path)
                return ""
        else:
            temp_normalized_wav_path = ""

        src_wav_path = temp_normalized_wav_path if temp_normalized_wav_path else temp_wav_path
        is_success = SoundUtil.encode_to_flac(src_wav_path, flac_path)

        if not is_success:
            delete_temp_file(temp_wav_path)
            if temp_normalized_wav_path:
                delete_temp_file(temp_normalized_wav_path)
            return ""

        if not is_retry:
            printt(f"Saved: {flac_path}")
            printt()

        delete_temp_file(temp_wav_path)
        if temp_normalized_wav_path:
            delete_temp_file(temp_normalized_wav_path)

        if state.prefs.play_on_generate:
            SoundUtil.play_flac_async(flac_path)

        return flac_path

    @staticmethod
    def generate_wav_file(
        dest_file_path: str,
        prompt: str,
        voice: dict,
        temperature: float
    ) -> bool:

        interface = Shared.get_oute_interface()

        # First, clone GENERATION_CONFIG from config file
        from outetts.models.config import GenerationConfig, SamplerConfig
        from outetts.models.info import GenerationType
        from tts_audiobook_tool.tts_config import GENERATION_CONFIG
        try:
            from .tts_config_dev import GENERATION_CONFIG
        except ImportError:
            pass
        gen_config = copy.deepcopy(GENERATION_CONFIG)

        gen_config.text = prompt
        gen_config.speaker = voice
        gen_config.sampler_config = SamplerConfig(temperature)

        try:
            output = interface.generate(config=gen_config)
            output.save(dest_file_path)
        except Exception as e:
            printt(str(e), "error")
            return False
        return True

# ---

def print_item_heading(text: str, index: int, count: int, total: int) -> None:
    s  = f"{COL_ACCENT}[{COL_DEFAULT}{count}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
    s += f"{COL_ACCENT}Generating audio for text segment {COL_DEFAULT}{index+1}{COL_ACCENT}:{COL_DEFAULT}"
    print(s)
    printt(f"{COL_DIM}{Ansi.ITALICS}{text.strip()}")
    printt()
