import time

import numpy as np
from tts_audiobook_tool.app_types import Sound, SttVariant
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.memory_util import MemoryUtil
from tts_audiobook_tool.models_util import ModelsUtil
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.phrase import PhraseGroup, Reason
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.validation_result import ValidationResult


class RealTimeUtil:

    @staticmethod
    def start(
            state: State,
            phrase_groups: list[PhraseGroup],
            line_range: tuple[int, int] | None
        ) -> None:
        """
        line_range is one-indexed
        """

        project = state.project

        if not line_range:
            line_range = (1, len(phrase_groups))

        start_index, end_index = line_range
        start_index -= 1
        end_index -= 1
        num_items = end_index - start_index + 1
        showed_vram_warning = False
        
        # Warm up models
        force_no_stt = ValidateUtil.is_unsupported_language_code(project.language_code)
        did_interrupt = ModelsUtil.warm_up_models(state)
        if did_interrupt:
            print_feedback("\nCancelled")
            return

        showed_vram_warning = MemoryUtil.show_vram_memory_warning_if_necessary()

        s = "Starting real-time playback..."
        if state.prefs.stt_variant == SttVariant.DISABLED:
            s += f" {COL_DIM}(speech-to-text validation disabled){COL_ACCENT}"
        print_heading(s, dont_clear=True, non_menu=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        # Outer loop

        SigIntHandler().set("generating")
        did_interrupt = False
        stream = None
        count = 0

        for index in range(start_index, end_index + 1):

            if did_interrupt:
                break

            if not showed_vram_warning:
                b = MemoryUtil.show_vram_memory_warning_if_necessary()
                if b:
                    print("\a", end="")
                    showed_vram_warning = True

            phrase_group = phrase_groups[index]
            phrase = phrase_group.as_flattened_phrase()

            printt()
            GenerateUtil.print_batch_heading(
                indices=[index],
                num_complete=index - start_index,
                num_remaining=end_index + 1 - index,
                num_total=end_index + 1 - start_index
            )
            printt(f"{COL_DIM}{Ansi.ITALICS}{phrase_group.presentable_text}")
            printt()

            # TODO: make dynamic - if "estimated gen time" < buffer duration x ~2 x max_retries...
            has_runway = (stream is not None and stream.buffer_duration >= (REQUIRED_SECONDS_PER_RETRY * state.project.max_retries))
            
            sound_opt, did_interrupt = RealTimeUtil.generate_full_flow(
                state, phrase_groups, index, has_runway=has_runway
            )
            if not sound_opt:
                printt(f"{COL_ERROR}Coun't generate sound{COL_DIM}, continuing to next segment")
                printt()
                continue
            else:
                sound = sound_opt

            # Add appended sound
            if index == end_index:
                appended_sound = None
            else:
                use_sound_effect = state.project.use_section_sound_effect and phrase.reason == Reason.SECTION
                if use_sound_effect:
                    path = SECTION_SOUND_EFFECT_PATH
                    result = SoundFileUtil.load(path, sound.sr)
                    if isinstance(result, str):
                        printt(f"{COL_ERROR}Error loading sound effect: {path}")
                        appended_sound = None
                    else:
                        appended_sound = result.data
                else:
                    silence_duration = phrase.reason.pause_duration
                    appended_sound = np.zeros(int(sound.sr * silence_duration), dtype=sound.data.dtype)

            # Start stream lazy
            if not stream:
                if Tts.get_type() == TtsModelInfos.GLM: # special case
                    sr = state.project.glm_sr
                else:
                    sr = Tts.get_type().value.sample_rate
                stream = SoundDeviceStream(sr)
                stream.start()

            # Add sound to the stream
            stream.add_data(sound.data)
            if appended_sound is not None:
                # Add page-turn sound
                stream.add_data(appended_sound)

            full_duration = sound.duration
            if appended_sound is not None:
                full_duration += len(appended_sound) / sound.sr

            # Print buffer duration
            value = stream.buffer_duration - full_duration
            if value <= 0.0:
                value = +0.0
            s = f"{COL_ERROR}" if value < 0.1 else f"{COL_OK}"
            s += f"{duration_string(value, include_tenth=True)}"
            printt(f"Buffer duration: {s}")

            # Sleep if necessary to prevent growing buffer beyond threshold
            if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS:
                sleep_duration = int(full_duration)
                printt(f"Sleeping for {sleep_duration}s")
                for index in range(0, sleep_duration):
                    time.sleep(1) # hah

            count += 1

        # Finished
        SigIntHandler().clear()
        if stream:
            if stream.buffer_duration > 0:
                # Gives opportunity for remaining buffer data to play through before killing stream
                printt()
                AskUtil.ask_enter_to_continue()
            stream.shut_down()
        printt()

    @staticmethod
    def generate_full_flow(  
            state: State,
            phrase_groups: list[PhraseGroup],
            index: int,
            has_runway: bool
    ) -> tuple[Sound | None, bool]:
        """
        Similar to `GenerateUtil.generate_full_flow()` but slightly different control flow, simpler.
        Returns tuple: Sound or no-sound if problem, and if user interrupted
        """

        SigIntHandler().set("generating")

        project = state.project
        phrase_group = phrase_groups[index]
        did_interrupt = False

        gen_result: ValidationResult | str  = ""
        num_attempts = 1 + project.max_retries if has_runway else 1
        
        for attempt in range(num_attempts):

            gen_result = GenerateUtil.generate_and_validate_batch(
                state=state,
                indices=[index],
                phrase_groups=phrase_groups,
                stt_variant=state.prefs.stt_variant,
                stt_config=state.prefs.stt_config,
                force_random_seed=(attempt > 0),
                is_realtime=True,
                is_skip_reason_buffer=not has_runway
            )[0]

            # Print result info
            if isinstance(gen_result, str):
                err = gen_result
                printt(f"{COL_ERROR}Model fail: {err}")
                printt()
            else:
                printt(f"Transcript validation: {gen_result.get_ui_message()}")

            if SigIntHandler().did_interrupt:
                did_interrupt = True

            if did_interrupt:
                break
            is_pass = isinstance(gen_result, ValidationResult) and not gen_result.is_fail
            if is_pass:
                break

        SigIntHandler().clear()        

        if isinstance(gen_result, str): 
            return None, did_interrupt # is error
        else:
            validation_result = gen_result
            if project.realtime_save:
                err, saved_path = GenerateUtil.save_sound_and_timing_json(
                    state, phrase_group, index, validation_result, is_real_time=True
                )
                if err:
                    printt(f"{COL_ERROR}Couldn't save file: {err} {saved_path}")
                else:
                    printt(f"{COL_DIM}Saved: {Path(saved_path).name}")
            return validation_result.sound, did_interrupt

# ---

REQUIRED_SECONDS_PER_RETRY = 60.0
