import os
import time
from typing import cast
from outetts.models.config import GenerationConfig
from outetts.models.config import GenerationConfig, SamplerConfig
from outetts.models.info import GenerationType

from outetts.version.interface import InterfaceHF

from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from .util import *
from .constants import *
from .project_dir_util import *

class GenerateUtil:

    @staticmethod
    def generate_all(state: State) -> bool:    
        """
        First asks for confirmation.  Returns True if aborted.
        Generates audio files for all text segments.
        Skips those that are already completed.
        """

        voice = cast(dict, state.voice)
        project_dir = state.project_dir
        text_segments = state.text_segments
        interface = state.interface

        config = GenerationConfig(
            text="",
            generation_type= GenerationType.CHUNKED, # type: ignore
            speaker=voice,
            sampler_config=SamplerConfig(temperature=state.temperature)  # type: ignore
        ) 

        dic = ProjectDirUtil.get_audio_segment_files(state)
        indices_completed = list(dic.keys())
        indices_completed.sort()

        if len(indices_completed) == len(text_segments):
            printt(f"All audio files already generated ({len(text_segments)})")
            ask("Press enter: ")
            return False

        # Will generate 5 audio segments 
        # (Already completed 3 audio segments)

        num_left = len(text_segments) - len(indices_completed)
        printt(f"Will generate {num_left} audio segments")
        if len(indices_completed) > 0:
            printt(f"(Already completed {len(indices_completed)} audio segments)")
        
        if not ask_confirm():
            return True

        for i in range(len(text_segments)):
            if i in indices_completed:
                continue
            _ = GenerateUtil.generate_and_convert_flac(i, state, config)
        
        return False
    
    @staticmethod
    def generate_and_convert_flac(index: int, state: State, config: GenerationConfig) -> bool:
        """
        Generates temp wav file, converts to flac, deletes temp wav
        """
        temp_wav_path = os.path.join(state.project_dir, make_random_hex_string() + ".wav")
        flac_path = HashFileUtil.make_segment_file_path(index, state)
        text_segment = state.text_segments[index]

        s = f"{COL_ACCENT}Generating audio for index {COL_DEFAULT}{index}{COL_ACCENT}:{COL_DEFAULT}\n"
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
        s = f"Duration {COL_ACCENT}{audio_seconds:.1f}{COL_DEFAULT}s, elapsed {COL_ACCENT}{elapsed:.1f}{COL_DEFAULT}s"
        s += f" = {COL_ACCENT}{multi_string}x{COL_DEFAULT}"
        printt(s)
        
        is_success = encode_to_flac(temp_wav_path, flac_path)
        if not is_success:
            delete_temp_file(temp_wav_path)
            return False

        printt(f"Saved {flac_path}\n")
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
