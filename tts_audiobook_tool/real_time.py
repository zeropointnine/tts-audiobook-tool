import time

import numpy as np
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.shared import Shared
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.util import *


class RealTime:

    @staticmethod
    def go(project: Project) -> None:
        """
        """

        # Get start line number
        inp = ask("Line number to start at: ")
        if not inp:
            return
        try:
            line_number = int(inp)
        except:
            ask_continue("Bad value.")
            return
        if line_number < 1 or line_number > len(project.text_segments):
            ask_continue("Out of range")
            return

        # Warm up models
        printt("Warming up models...")
        printt()
        if Shared.is_oute():
            _ = Shared.get_oute()
        elif Shared.is_chatterbox():
            _ = Shared.get_chatterbox()
        whisper_model = Shared.get_whisper()

        # Start stream
        stream_sr = Shared.get_model_samplerate()
        if not stream_sr:
            ask_continue("No tts model samplerate, exiting. ")
            return
        stream = SoundDeviceStream(stream_sr)
        stream.start()

        # Start loop

        printt("Starting playback")
        printt(f"Press {make_hotkey_string("Control-C")} to interrupt")
        printt()

        Shared.mode = "generating"

        start_index = line_number - 1
        end_index = len(project.text_segments) - 1

        last_text_segment: TextSegment | None = None
        current_text_segment: TextSegment | None = None

        last_sound: Sound | None = None
        current_sound: Sound | None = None

        for i in range(start_index, end_index + 1):

            if Shared.stop_flag:
                Shared.stop_flag = False
                Shared.mode = ""
                did_interrupt = True
                break

            printt(f"Buffer duration: {duration_string(stream.buffer_duration)}")

            current_text_segment = project.text_segments[i]

            has_enough_runway = (stream.buffer_duration >= 45.0) # TODO: if estimated gen time < buffer duration * 2 ...
            max_passes = 2 if has_enough_runway else 1

            current_sound, validate_result = GenerateUtil.generate_sound_full(i, project, whisper_model, max_passes=max_passes)
            # TODO: consider printing similar feedback as with regular gen (or maybe not to keep the noise down)

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

                # Prevent growing the buffer beyond threshold
                if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS:
                    sleep_duration = int(current_sound.duration + pause_duration)
                    print(f"Sleeping for {int(sleep_duration)}s")
                    for i in range(0, sleep_duration):
                        time.sleep(1) # hah

            last_text_segment = current_text_segment
            last_sound = current_sound

        print("Closing stream")
        stream.shut_down()

        ask_continue()
