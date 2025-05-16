import gc
import os
import time
from typing import cast
from outetts.models.config import GenerationConfig
from outetts.models.config import GenerationConfig, SamplerConfig
from outetts.models.info import GenerationType

from outetts.version.interface import InterfaceHF
import torch
import whisper

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_dir_util import *
from tts_audiobook_tool.validate_util import ValidateItem, ValidateResult, ValidateUtil

class GenerateUtil:

    @staticmethod
    def generate_validate_fix(state: State, indices: list[int], mode: str) -> None:
        """
        "generate"
        "generate-and-fix"
        "validate-and-fix": skips the first generate pass
        """

        if not mode in ["generate", "generate-and-fix", "validate-and-fix"]:
            raise ValueError("Bad value for mode")

        if mode in ["generate-and-fix", "validate-and-fix"]:
            printt("Initializing whisper model...")
            printt()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            whisper_model = whisper.load_model("turbo", device=device)
        else:
            whisper_model = None

        config = GenerationConfig(
            text="", # type: ignore
            generation_type= GenerationType.CHUNKED, # type: ignore
            speaker=cast(dict, state.project.voice), # type: ignore
            sampler_config=SamplerConfig(temperature=state.prefs.temperature)  # type: ignore
        )

        # TODO: Duplicate work. Need to figure out best way to pass index-and-path through this whole feature, with all its permutations
        index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)

        start_time = time.time()

        count = 1
        Shared.mode = "generating"

        for i in indices:

            print_item_heading(state.project.text_segments[i], i, count, len(indices))

            if mode == "generate":
                _ = GenerateUtil.generate_and_make_flac(index=i, state=state, config=config)
            elif mode == "generate-and-fix":
                _ = GenerateUtil.generate_validate_and_fix_item(index=i, state=state, config=config, whisper_model=whisper_model)
            else: # == "validate-and-fix"
                path = index_to_path[i]
                _ = GenerateUtil.generate_validate_and_fix_item(index=i, state=state, config=config, whisper_model=whisper_model, skip_generate_file_path=path)

            count += 1
            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                break

        printt(f"Elapsed: {time_string(time.time() - start_time)}")
        printt()

        if whisper_model:
            printt("Unloading whisper...")
            printt()
            del whisper_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        ask("Press enter: ")

    @staticmethod
    def generate_validate_and_fix_item(
        index: int,
        state: State,
        config: GenerationConfig,
        whisper_model,
        skip_generate_file_path: str=""
    ) -> None:

        pass_num = 1

        while True:

            # Make audio file
            if not skip_generate_file_path:
                file_path = GenerateUtil.generate_and_make_flac(index=index, state=state, config=config, is_retry=(pass_num > 1))
                if not file_path:
                    # Unexpected error making audio file
                    break
            else:
                # Validate-and-then-fix
                file_path = skip_generate_file_path
                skip_generate_file_path = ""

            # Validate audio file
            item = ValidateItem(index, file_path, state.project.text_segments[index])
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
                        printt("Shouldn't get here")
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
                        printt(message + "\n" + "Voice line still has error" + "\n")
                        break
                    case _:
                        printt("Shouldn't get here")
                        break

        # ...

    @staticmethod
    def generate_and_make_flac(
        index: int,
        state: State,
        config: GenerationConfig,
        is_retry: bool=False
    ) -> str:
        """
        Generates temp wav file, converts to flac, deletes temp wav
        Prints info
        Returns file path on success, else empty string
        """
        temp_wav_path = os.path.join(state.prefs.project_dir, make_random_hex_string() + ".wav")
        flac_path = HashFileUtil.make_segment_file_path(index, state)
        text_segment = state.project.text_segments[index]

        start_time = time.time()
        is_success = GenerateUtil.generate_wav_file(temp_wav_path, text_segment, state.interface, config)
        if not is_success:
            delete_temp_file(temp_wav_path)
            return ""

        # Print time info
        elapsed = time.time() - start_time
        audio_seconds = estimated_wav_seconds(temp_wav_path)
        s = f"Duration: {COL_ACCENT}{audio_seconds:.1f}s{COL_DEFAULT}, elapsed {COL_ACCENT}{elapsed:.1f}s"
        if elapsed > 0:
            multi = audio_seconds / elapsed
            s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
        printt(s)

        is_success = encode_to_flac(temp_wav_path, flac_path)
        if not is_success:
            delete_temp_file(temp_wav_path)
            return ""

        if not is_retry:
            printt(f"Saved: {flac_path}\n")

        delete_temp_file(temp_wav_path)

        if state.prefs.play_on_generate:
            SoundUtil.play_flac_async(flac_path)

        return flac_path

    @staticmethod
    def generate_wav_file(
            dest_file_path: str,
            prompt: str,
            interface: InterfaceHF,
            config: GenerationConfig
    ) -> bool:
        try:
            config.text = prompt
            output = interface.generate(config=config)
            output.save(dest_file_path)
        except Exception as e:
            printt(str(e), "error")
            return False
        return True

def print_item_heading(text_segment: str, index: int, count: int, total: int) -> None:
    s  = f"{COL_ACCENT}[{COL_DEFAULT}{count}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
    s += f"{COL_ACCENT}Generating audio for text segment index {COL_DEFAULT}{index}{COL_ACCENT}:{COL_DEFAULT}"
    print(s)
    printt(f"{COL_DIM}{Ansi.ITALICS}{text_segment}")
    printt()
