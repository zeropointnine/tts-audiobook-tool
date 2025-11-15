import time

import numpy as np
from tts_audiobook_tool.app_types import FailResult, PassResult, SkippedResult, Sound, SttConfig, SttVariant, TrimmableResult
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_segment import TextSegment, TextSegmentReason
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *


class RealTimeUtil:

    @staticmethod
    def start(
            state: State,
            text_segments: list[TextSegment],
            line_range: tuple[int, int] | None
        ) -> None:
        """
        line_range is one-indexed
        """

        project = state.project

        if not line_range:
            line_range = (1, len(text_segments))

        start_index, end_index = line_range
        start_index -= 1
        end_index -= 1
        num_items = end_index - start_index + 1

        SoundFileUtil.debug_save_dir = os.path.join(project.dir_path, PROJECT_SOUND_SEGMENTS_SUBDIR)

        # Warm up models
        Tts.warm_up_models()

        s = "Starting real-time playback..."
        if state.prefs.stt_variant == SttVariant.DISABLED:
            s += f" {COL_DIM}(speech-to-text validation disabled){COL_ACCENT}"
        print_heading(s, dont_clear=True)
        printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
        printt()

        # Start loop

        SigIntHandler().set("generating")

        stream = None
        last_text_segment: TextSegment | None = None
        current_text_segment: TextSegment | None = None
        count = 0

        for i in range(start_index, end_index + 1):

            if SigIntHandler().did_interrupt:
                break

            current_text_segment = text_segments[i]

            printt()
            GenerateUtil.print_item_heading(
                False, current_text_segment.text, i, count, num_items)

            # TODO: make dynamic - if "estimated gen time" < buffer duration * 2 ...
            has_enough_runway = (stream and stream.buffer_duration >= 60.0)
            max_passes = 2 if has_enough_runway else 0

            result = GenerateUtil.generate_sound_full_flow(
                project=project,
                text_segment=current_text_segment,
                stt_variant=state.prefs.stt_variant,
                stt_config=state.prefs.stt_config,
                max_passes=max_passes
            )
            if isinstance(result, str):
                err = result
                printt(f"{COL_ERROR}Model failure, skipping item:")
                printt(f"{COL_ERROR}{err}")
                printt()
                last_text_segment = None
                continue

            sound, validation_result = result

            # Print validation info
            s = "Speech-to-text validation: "
            if isinstance(validation_result, SkippedResult):
                s += "Skipped"
                if state.prefs.stt_variant == SttVariant.DISABLED:
                    s += " (Whisper disabled)"
                elif max_passes == 0:
                    s += " (requires 60s buffered audio)"
            elif isinstance(validation_result, PassResult):
                s += "Passed"
            elif isinstance(validation_result, TrimmableResult):
                s += f"{COL_OK}Fixed{COL_DEFAULT} - {validation_result.get_ui_message()}"
            elif isinstance(validation_result, FailResult):
                s += f"{COL_ERROR}Max fails reached{COL_DEFAULT}, playing anyway"
            printt(s)

            # "Pre-sound" is some audio inserted before the real generated audio
            # (ie, some duration of silence, or special sound effect)
            if i == start_index:
                pre_sound = None
            else:
                use_sound_effect = state.prefs.use_section_sound_effect and \
                        last_text_segment and last_text_segment.reason == TextSegmentReason.SECTION
                if use_sound_effect:
                    path = SECTION_SOUND_EFFECT_PATH
                    result = SoundFileUtil.load(path, sound.sr)
                    if isinstance(result, str):
                        printt(f"{COL_ERROR}Error loading sound effect: {path}")
                        pre_sound = None
                    else:
                        pre_sound = result.data
                else:
                    if last_text_segment:
                        silence_duration = last_text_segment.reason.pause_duration
                    else:
                        silence_duration = 1 # fallback
                    if silence_duration:
                        pre_sound = np.zeros(int(sound.sr * silence_duration), dtype=sound.data.dtype)
                    else:
                        pre_sound = None

            # Start stream lazy
            if not stream:
                stream = SoundDeviceStream(Tts.get_type().value.sample_rate)
                stream.start()

            # Add to the stream
            if pre_sound is not None:
                stream.add_data(pre_sound)
            stream.add_data(sound.data)

            duration = sound.duration
            if pre_sound is not None:
                duration += len(pre_sound) / sound.sr

            # Print buffer duration
            value = stream.buffer_duration - duration
            if value <= 0.0:
                value = +0.0
            s = f"{COL_ERROR}" if value < 0.1 else f"{COL_OK}"
            s += f"{duration_string(value, include_tenth=True)}"
            printt(f"Buffer duration: {s}")

            SoundFileUtil.debug_save("realtime", sound)

            # Sleep if necessary to prevent growing buffer beyond threshold
            if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS:
                sleep_duration = int(duration)
                print(f"Sleeping for {sleep_duration}s")
                for i in range(0, sleep_duration):
                    time.sleep(1) # hah

            last_text_segment = current_text_segment
            count += 1

        if not stream:
            print()
            return

        if stream.buffer_duration > 0:
            # Gives opportunity for remaining buffer data to play through before killing stream
            print()
            AskUtil.ask_enter_to_continue()

        SigIntHandler().clear()

        if stream:
            stream.shut_down()
        printt()
