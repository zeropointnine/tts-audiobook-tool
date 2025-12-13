import math
import time

import numpy as np
from tts_audiobook_tool.app_types import FailResult, PassResult, SkippedResult, Sound, SttConfig, SttVariant, TrimmableResult, ValidationResult
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.phrase import PhraseGroup, Reason
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants import *
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

        SoundFileUtil.debug_save_dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)
        
        # Warm up models
        force_no_stt = ValidateUtil.is_unsupported_language_code(project.language_code)
        did_cancel = Tts.warm_up_models(force_no_stt)
        if did_cancel:
            print_feedback("\nCancelled")
            return

        s = "Starting real-time playback..."
        if state.prefs.stt_variant == SttVariant.DISABLED:
            s += f" {COL_DIM}(speech-to-text validation disabled){COL_ACCENT}"
        print_heading(s, dont_clear=True, non_menu=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        # Start loop

        SigIntHandler().set("generating")

        stream = None
        count = 0

        for i in range(start_index, end_index + 1):

            if SigIntHandler().did_interrupt:
                break

            phrase_group = phrase_groups[i]
            phrase = phrase_group.as_flattened_phrase()

            printt()
            GenerateUtil.print_item_heading(False, phrase.text, i, count, num_items)

            # TODO: make dynamic - if "estimated gen time" < buffer duration x ~2 ...
            has_enough_runway = (stream and stream.buffer_duration >= STREAM_DURATION_THRESHOLD)
            max_passes = 2 if has_enough_runway else 0

            result = GenerateUtil.generate_single_full_flow(
                project=project,
                phrase_group=phrase_group,
                stt_variant=state.prefs.stt_variant,
                stt_config=state.prefs.stt_config,
                max_passes=max_passes,
                is_realtime=True,
                skip_reason_buffer=(state.prefs.stt_variant != SttVariant.DISABLED and not has_enough_runway)
            )
            if isinstance(result, str):
                err = result
                printt(f"{COL_ERROR}Model failure, skipping item:")
                printt(f"{COL_ERROR}{err}")
                printt()
                continue

            sound, _, __ = result

            if i == end_index:
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
                stream = SoundDeviceStream(Tts.get_type().value.sample_rate)
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

            SoundFileUtil.debug_save("realtime", sound)

            # Sleep if necessary to prevent growing buffer beyond threshold
            if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS:
                sleep_duration = int(full_duration)
                printt(f"Sleeping for {sleep_duration}s")
                for i in range(0, sleep_duration):
                    time.sleep(1) # hah

            count += 1

        if not stream:
            printt()
            return

        if stream.buffer_duration > 0:
            # Gives opportunity for remaining buffer data to play through before killing stream
            printt()
            AskUtil.ask_enter_to_continue()

        SigIntHandler().clear()

        if stream:
            stream.shut_down()
        printt()

# ---

STREAM_DURATION_THRESHOLD = 60.0