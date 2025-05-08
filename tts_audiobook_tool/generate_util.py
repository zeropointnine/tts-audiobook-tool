import os
import time
from typing import cast
from outetts.models.config import GenerationConfig
from outetts.models.config import GenerationConfig, SamplerConfig
from outetts.models.info import GenerationType

from outetts.version.interface import InterfaceHF

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from .util import *
from .constants import *
from .project_dir_util import *

class GenerateUtil:

    @staticmethod
    def ask_generate_all(state: State, skip_ask: bool=False) -> bool:
        """
        First asks for confirmation.  Returns True if aborted.
        Generates audio files for all text segments.
        Skips those that are already completed.
        """

        voice = cast(dict, state.voice)
        text_segments = state.text_segments

        config = GenerationConfig(
            text="", # type: ignore
            generation_type= GenerationType.CHUNKED, # type: ignore
            speaker=voice, # type: ignore
            sampler_config=SamplerConfig(temperature=state.temperature)  # type: ignore
        )

        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        indices_completed = list(dic.keys())
        indices_completed.sort()

        if len(indices_completed) == len(text_segments):
            printt(f"All audio files already generated ({len(text_segments)})")
            ask("Press enter: ")
            return False

        num_left = len(text_segments) - len(indices_completed)
        printt(f"Will generate {num_left} audio segments")
        if len(indices_completed) > 0:
            printt(f"(Already completed {len(indices_completed)} audio segments)")

        if skip_ask:
            printt()
        else:
            if not ask_confirm():
                return True

        start_time = time.time()
        count = 0
        for i in range(len(text_segments)):
            if i in indices_completed:
                continue
            count += 1
            _ = GenerateUtil.generate_and_convert_flac(
                index=i,
                state=state,
                config=config,
                batch_count=count,
                batch_total=num_left,
                batch_start_time=start_time
            )
        elapsed = time.time() - start_time
        printt(f"Elapsed: {AppUtil.time_string(elapsed)}\a\n")

        return False

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
        temp_wav_path = os.path.join(state.project_dir, make_random_hex_string() + ".wav")
        flac_path = HashFileUtil.make_segment_file_path(index, state)
        text_segment = state.text_segments[index]

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

        if state.play_on_generate:
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
