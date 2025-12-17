import time

import numpy as np
from tts_audiobook_tool.app_types import FailResult, Sound, SttConfig, SttVariant, ValidationResult
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.project import Project
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
        
        # Warm up models
        force_no_stt = ValidateUtil.is_unsupported_language_code(project.language_code)
        did_interrupt = Tts.warm_up_models(force_no_stt)
        if did_interrupt:
            print_feedback("\nCancelled")
            return

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

            phrase_group = phrase_groups[index]
            phrase = phrase_group.as_flattened_phrase()

            printt()
            GenerateUtil.print_item_heading(False, phrase.text, index, count, num_items)

            # TODO: make dynamic - if "estimated gen time" < buffer duration x ~2 ...
            has_runway = (stream is not None and stream.buffer_duration >= STREAM_DURATION_THRESHOLD)            
            
            sound_opt, did_interrupt = RealTimeUtil.generate_full_flow(
                project, phrase_groups, index, project.language_code, state.prefs.stt_variant, state.prefs.stt_config,
                has_runway=has_runway,
                should_save=state.project.realtime_save
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
            project: Project,
            phrase_groups: list[PhraseGroup],
            index: int,
            language_code: str,
            stt_variant: SttVariant,
            stt_config: SttConfig,
            has_runway: bool,
            should_save: bool
    ) -> tuple[Sound | None, bool]:
        """
        Similar to `GenerateUtil.generate_single_full_flow()` but slightly different logic flow, simpler
        Returns Sound or no-sound if problem, and if user interrupted
        """

        SigIntHandler().set("generating")

        phrase_group = phrase_groups[index]
        did_interrupt = False

        gen_result: tuple[Sound, ValidationResult] | str  = ""
        max_attempts = 2 if has_runway else 1
        for attempt in range(max_attempts):

            gen_result = GenerateUtil.generate_and_validate(
                project=project,
                phrase_group=phrase_group,
                index=index,
                language_code=language_code,
                stt_variant=stt_variant,
                stt_config=stt_config,
                gen_info_path=project.realtime_path,
                is_retry=(attempt > 0),
                is_realtime=True,
                is_skip_reason_buffer=not has_runway
            )

            # Print result feedback
            if isinstance(gen_result, str):
                err = gen_result
                printt(f"{COL_ERROR}Model fail: {err}")
                printt()
            else:
                GenerateUtil.print_validation_result(
                    gen_result[1], is_last_attempt=(attempt == max_attempts - 1), is_real_time=True
                )                

            if SigIntHandler().did_interrupt:
                did_interrupt = True

            success = isinstance(gen_result, tuple) and not isinstance(gen_result[1], FailResult)
            if success or did_interrupt:
                break

        if isinstance(gen_result, str): # is error
            result = None, did_interrupt
        else:
            sound, validation_result = gen_result
            if should_save:
                err, saved_path = GenerateUtil.save_gen(project, phrase_group, index, sound, validation_result, is_real_time=True)
                if err:
                    printt(f"{COL_ERROR}Couldn't save file: {err} {saved_path}")
                else:
                    printt(f"Saved: {saved_path}")
            result = sound, did_interrupt

        SigIntHandler().clear()        
        return result

# ---

STREAM_DURATION_THRESHOLD = 60.0
