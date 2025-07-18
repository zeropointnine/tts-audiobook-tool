import time

import numpy as np
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.generate_util import GenerateUtil
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_segment import TextSegment
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.util import *


class RealTime:

    @staticmethod
    def start(project: Project, text_segments: list[TextSegment], start_index: int) -> None:
        """
        """

        SoundFileUtil.debug_save_dir = AppUtil.get_app_temp_dir()

        # Warm up models
        Tts.warm_up_models()

        # Start stream
        stream = SoundDeviceStream(Tts.get_type().value.sample_rate)
        stream.start()

        # Start loop

        print_heading("Starting real-time playback...", dont_clear=True)
        printt(f"{COL_DIM}Press {make_hotkey_string("Control-C")}{COL_DIM} to interrupt")
        printt()

        SigIntHandler().set("generating")

        end_index = len(text_segments) - 1

        last_text_segment: TextSegment | None = None
        current_text_segment: TextSegment | None = None

        last_sound: Sound | None = None
        current_sound: Sound | None = None

        did_interrupt = False

        for i in range(start_index, end_index + 1):

            if SigIntHandler().did_interrupt:
                did_interrupt = True
                break

            current_text_segment = text_segments[i]

            has_enough_runway = (stream.buffer_duration >= 45.0) # TODO: make dynamic - if estimated gen time < buffer duration * 2 ...
            max_passes = 2 if has_enough_runway else 1

            current_sound, _ = GenerateUtil.generate_sound_full(project, current_text_segment, max_passes=max_passes)

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
                s = f"{COL_ERROR}" if value < 0.1 else f"{COL_OK}"
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

        if not did_interrupt:
            ask_continue("\n")

        SigIntHandler().clear()

        stream.shut_down()
        printt()
