import time

import numpy as np
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.util import *


class RealTime:

    @staticmethod
    def start(project: Project, text_segments: list[TextSegment]) -> None:
        """
        """

        # Ask for starting line number
        inp = ask("Line number to start at: ")
        if not inp:
            return
        try:
            line_number = int(inp)
        except:
            ask_continue("Bad value.")
            return
        if line_number < 1 or line_number > len(text_segments):
            ask_continue("Out of range")
            return

        # Warm up models
        Shared.warm_up_models()

        # Start stream
        stream_sr = Shared.get_model_samplerate()
        if not stream_sr:
            ask_continue("No tts model samplerate, exiting. ")
            return
        stream = SoundDeviceStream(stream_sr)
        stream.start()

        # Start loop

        print_heading("Starting real-time playback...", dont_clear=True)
        printt(f"{COL_DIM}Press {make_hotkey_string("Control-C")}{COL_DIM} to interrupt")
        printt()

        Shared.mode = "generating"

        start_index = line_number - 1
        end_index = len(text_segments) - 1

        last_text_segment: TextSegment | None = None
        current_text_segment: TextSegment | None = None

        last_sound: Sound | None = None
        current_sound: Sound | None = None

        for i in range(start_index, end_index + 1):

            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                break

            current_text_segment = text_segments[i]

            has_enough_runway = (stream.buffer_duration >= 45.0) # TODO: if estimated gen time < buffer duration * 2 ...
            max_passes = 2 if has_enough_runway else 1

            current_sound, validate_result = GenerateUtil.generate_sound_full(project, current_text_segment, max_passes=max_passes)

            # TODO: printing similar feedback as with regular gen, or at least for fails

            if current_sound:

                # Calc pause duration
                if not last_text_segment:
                    pause_duration = 0
                else:
                    if not last_sound:
                        pause_duration = 1
                    else:
                        pause_duration = last_text_segment.reason.pause_duration

                # Add pause
                if pause_duration:
                    silence = np.zeros(int(current_sound.sr * pause_duration), dtype=current_sound.data.dtype)
                    stream.add_data(silence)

                # Add sound
                stream.add_data(current_sound.data)

                # Print buffer duration
                value = stream.buffer_duration - current_sound.duration - pause_duration
                if value <= 0.0:
                    value = +0.0
                s = f"{COL_ERROR}" if value < 0.1 else f"{COL_ACCENT}"
                s += f"{duration_string(value, include_tenth=True)}"
                printt(f"Buffer duration: {s}")

                SoundFileUtil.debug_save("realtime", current_sound)

                # Prevent growing the buffer beyond threshold
                if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS:
                    sleep_duration = int(current_sound.duration + pause_duration)
                    print(f"Sleeping for {int(sleep_duration)}s")
                    for i in range(0, sleep_duration):
                        time.sleep(1) # hah

            last_text_segment = current_text_segment
            last_sound = current_sound

        stream.shut_down()
        printt()
