import os
import time
from typing import cast
from outetts.models.config import GenerationConfig
from outetts.models.config import GenerationConfig, SamplerConfig
from outetts.models.info import GenerationType

from outetts.version.interface import InterfaceHF

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_dir_util import *

class GenerateUtil:

    @staticmethod
    def submenu_and_generate(state: State, indices: list[int]=[]) -> None:
        """
        Generates the audio files
        Considers all items if `indices` is empty
        Skips items that are already completed
        Terminates with prompt
        """

        if not indices:
            indices = list(range(len(state.project.text_segments)))
        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        already_complete = list(dic.keys())
        original_len = len(indices)
        indices = [item for item in indices if item not in already_complete]
        indices.sort()

        if not indices:
            printt(f"Already generated ({original_len})")
            ask("Press enter: ")
            return

        print_heading("Generate audio:")

        elapsed_message = f"Will generate {len(indices)} audio segments"
        num_completed = original_len - len(indices)
        if num_completed > 0:
            elapsed_message += f" (already completed {num_completed} items)"
        printt(elapsed_message)
        printt()

        printt(f"{make_hotkey_string("1")} Generate, and concatenate when finished")
        printt(f"{make_hotkey_string("2")} Generate only")
        printt()

        hotkey = ask_hotkey()
        if hotkey == "1":
            and_concat = True
        elif hotkey == "2":
            and_concat = False
        else:
            return

        # ---

        config = GenerationConfig(
            text="", # type: ignore
            generation_type= GenerationType.CHUNKED, # type: ignore
            speaker=cast(dict, state.project.voice), # type: ignore
            sampler_config=SamplerConfig(temperature=state.prefs.temperature)  # type: ignore
        )

        start_time = time.time()

        count = 1
        Shared.mode = "generating"
        was_interrupted = False

        for i in indices:
            _ = GenerateUtil.generate_and_convert_flac(
                index=i,
                state=state,
                config=config,
                batch_count=count,
                batch_total=len(indices),
                batch_start_time=start_time
            )
            count += 1
            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                was_interrupted = True
                break

        elapsed_message = f"Elapsed: {AppUtil.time_string(time.time() - start_time)}"
        if was_interrupted:
            printt(elapsed_message)
            printt()
            ask("Press enter: ")
            return

        printt(elapsed_message + "\a")

        if and_concat:
            ConcatUtil.concatenate_project_flacs(state)
        else:
            ask("Finished. Press enter: ")

    @staticmethod
    def generate_and_convert_flac(
        index: int,
        state: State,
        config: GenerationConfig,
        batch_count: int = 0,
        batch_total: int = 0,
        batch_start_time: float = 0.0
    ) -> bool:
        """
        Generates temp wav file, converts to flac, deletes temp wav
        """
        temp_wav_path = os.path.join(state.prefs.project_dir, make_random_hex_string() + ".wav")
        flac_path = HashFileUtil.make_segment_file_path(index, state)
        text_segment = state.project.text_segments[index]

        s = ""
        if batch_count > 0 and batch_total > 0:
            s = f"{COL_ACCENT}[{COL_DEFAULT}{batch_count}{COL_ACCENT}/{COL_DEFAULT}{batch_total}{COL_ACCENT}] "
        s += f"{COL_ACCENT}Generating audio for index {COL_DEFAULT}{index}{COL_ACCENT}:{COL_DEFAULT}\n"
        s += f"{COL_DIM}{Ansi.ITALICS}{text_segment}\n"
        printt(s)

        start_time = time.time()
        is_success = GenerateUtil.generate_wav_file(temp_wav_path, text_segment, state.interface, config)
        if not is_success:
            delete_temp_file(temp_wav_path)
            return False

        elapsed = time.time() - start_time
        audio_seconds = estimated_wav_seconds(temp_wav_path)
        multi_string = f"{(audio_seconds / elapsed):.2f}" if elapsed > 0.01 else "---"
        s = f"Duration: {COL_ACCENT}{audio_seconds:.1f}s{COL_DEFAULT}, elapsed {COL_ACCENT}{elapsed:.1f}s"
        s += f" = {COL_ACCENT}{multi_string}x"
        if batch_count >= 3 and batch_total > 0 and batch_start_time > 0:
            batch_elapsed = time.time() - batch_start_time
            time_per_line = batch_elapsed / batch_count
            time_left = time_per_line * (batch_total - batch_count)
            s += f" {COL_DEFAULT} (est. time left: {COL_ACCENT}{AppUtil.time_string(time_left)}{COL_DEFAULT})"
        printt(s)

        is_success = encode_to_flac(temp_wav_path, flac_path)
        if not is_success:
            delete_temp_file(temp_wav_path)
            return False

        printt(f"Saved: {flac_path}\n")
        delete_temp_file(temp_wav_path)

        if state.prefs.play_on_generate:
            SoundUtil.play_flac_async(flac_path)

        return True

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
