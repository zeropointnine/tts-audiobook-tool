import os
import subprocess
import sounddevice as sd
import wave
import numpy as np
import threading
import soundfile as sf

from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *

class SoundFileUtil:

    @staticmethod
    def encode_to_flac(wav_path: str, flac_path: str) -> bool:

        # TODO: use "make_file" here

        try:
            cmd = [
                'ffmpeg',
                "-y", "-hide_banner", "-loglevel", "error",
                "-i", wav_path
            ]
            cmd.extend(FLAC_OUTPUT_FFMPEG_ARGUMENTS)
            cmd.append(flac_path)

            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            return True

        except subprocess.CalledProcessError as e:
            printt(str(e.stderr), "error")
            return False
        except Exception as e:
            printt(str(e), "error")
            return False

    @staticmethod
    def trim_flac_file(
            source_flac_path: str,
            dest_file_path: str,
            start_time_seconds: float,
            end_time_seconds: float
    ) -> bool:
        """
        Trims a source FLAC file from start_time_seconds to end_time_seconds
        and saves it to dest_file_path using ffmpeg.
        Returns True for success
        """

        source_flac_path = os.path.abspath(source_flac_path)
        dest_file_path = os.path.abspath(dest_file_path)

        if not os.path.exists(source_flac_path):
            L.w(f"Doesn't exist: {source_flac_path}")
            return False

        duration = end_time_seconds - start_time_seconds
        if duration <= 0:
            L.w(f"Bad start/end times {start_time_seconds} {end_time_seconds}")
            return False

        # TODO: replace with FfmpegUtil.make_file()

        partial_command = [
            "-hide_banner", "-loglevel", "error", "-y",
            "-i", source_flac_path,
            "-ss", str(start_time_seconds),
            "-to", str(end_time_seconds),
        ]
        partial_command.extend(FLAC_OUTPUT_FFMPEG_ARGUMENTS)
        err = FfmpegUtil.make_file(partial_command, dest_file_path, use_temp_file=False)
        if err:
            L.e(err)
        return not bool(err)

    @staticmethod
    def concatenate_flacs(
        file_paths: list[str],
        dest_flac_path: str
    ) -> str:
        """
        Concatenates a list of flac files into one.
        Empty file paths in list are ignored.
        Returns error message or empty string.
        """

        dest_base_path = str( Path(dest_flac_path).parent )
        os.makedirs(dest_base_path, exist_ok=True)

        # [1] Make temp text file with list of input WAVs for ffmpeg's concat demuxer

        temp_text_path = os.path.join(dest_base_path, PROJECT_CONCAT_TEMP_TEXT_FILE_NAME)
        try:
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                for path in file_paths:
                    # Must be absolute paths
                    # Important: Escape backslashes for ffmpeg on all platforms within the list file
                    # And wrap the path in single quotes for robustness (handles spaces etc.)
                    if not path:
                        continue
                    escaped_path = path.replace('\\', '\\\\')
                    f.write(f"file '{escaped_path}'\n")
        except Exception as e:
            delete_temp_file(temp_text_path)
            return str(e)

        # [2] Do concat

        printt(f"Concatenating {len(file_paths)} files...")
        printt()

        # -y: Overwrite output file without asking
        # -f concat: Use the concatenation demuxer
        # -safe 0: Allow unsafe file paths (useful if paths are complex, though quoting helps)
        # -i list_filename: Input file is the list we created
        partial_command = [
            "-hide_banner", "-loglevel", "error", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_text_path
        ]
        partial_command.extend(FLAC_OUTPUT_FFMPEG_ARGUMENTS)

        err = FfmpegUtil.make_file(partial_command, dest_flac_path, use_temp_file=True)
        delete_temp_file(temp_text_path)

        return err

    @staticmethod
    def add_silence_flac(src_path: str, dest_path: str, duration: float) -> str:
        """
        Returns error message on fail or empty string
        """
        data, samplerate = sf.read(src_path)
        silence = np.zeros(int(samplerate * duration), dtype=data.dtype) # Match dtype for concatenation
        new_data = np.concatenate([data, silence])
        try:
            sf.write(dest_path, new_data, samplerate, format='FLAC', subtype='PCM_16')
            data, samplerate = sf.read(dest_path)

        except Exception as e:
            return str(e)
        return ""

    @staticmethod
    def transcode_to_aac(source_file_path: str, kbps=96) -> tuple[str, str]:
        """
        Transcodes an audio file to AAC/MP4 format.
        Returns saved file name and error string, mutually exclusive
        """

        path = Path(source_file_path)
        if path.suffix in [".mp4", ".m4a", ".m4b"]:
            return "", "Is already an mp4 file"

        dest_file_path = str(path.with_suffix(".mp4"))
        dest_file_path = get_unique_file_path(dest_file_path)

        partial_command = [
            "-i", source_file_path,
            "-c:a", "aac",
            "-b:a", f"{kbps}k",
            "-v", "warning", "-progress",
            "-",
        ]
        err = FfmpegUtil.make_file(partial_command, dest_file_path, False)
        if err:
            return "", err

        return dest_file_path, ""

    @staticmethod
    def play_wav(file_path: str):
        with wave.open(file_path, 'rb') as wf:
            sample_rate = wf.getframerate()
            num_channels = wf.getnchannels()
            dtype = 'int16' if wf.getsampwidth() == 2 else 'int32'

            def callback(outdata, frames, time, status):
                if status:
                    print(f"Audio callback status: {status}")
                data = wf.readframes(frames)
                if len(data) == 0:
                    raise sd.CallbackStop()
                outdata[:len(data)] = np.frombuffer(data, dtype=dtype).reshape(-1, num_channels)

            with sd.OutputStream(samplerate=sample_rate, channels=num_channels, callback=callback, dtype=dtype):
                sd.sleep(int(wf.getnframes() / sample_rate * 1000))

    @staticmethod
    def play_flac_async(file_path: str):
        """
        Plays a FLAC file asynchronously using a streaming approach.
        Eats exceptions and prints them.
        """
        def _play_stream():
            try:
                with sf.SoundFile(file_path, 'r') as flac_file:
                    samplerate = flac_file.samplerate
                    channels = flac_file.channels

                    stream_finished_event = threading.Event()

                    def callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
                        if status:
                            print(f"Audio callback status (FLAC): {status}")

                        # Read 'frames' number of frames from the FLAC file
                        # always_2d=True ensures data is (n_frames, n_channels)
                        data_read = flac_file.read(frames, dtype='float32', always_2d=True)

                        if data_read.shape[0] == 0: # No more data
                            raise sd.CallbackStop()

                        # Fill outdata
                        if data_read.shape[0] < frames:
                            outdata[:data_read.shape[0]] = data_read
                            # Zero out the rest of the buffer
                            outdata[data_read.shape[0]:] = 0
                        else:
                            outdata[:] = data_read

                    def set_event_on_finish():
                        stream_finished_event.set()

                    with sd.OutputStream(
                        samplerate=samplerate,
                        channels=channels,
                        callback=callback,
                        dtype='float32', # Ensure this matches data_read dtype
                        finished_callback=set_event_on_finish
                    ):
                        stream_finished_event.wait() # Wait for playback to complete

            except Exception as e:
                printt(f"Couldn't play FLAC audio file via stream: {file_path} - {e}")

        thread = threading.Thread(target=_play_stream, daemon=True)
        thread.start()

    @staticmethod
    def print_samplerates(dir_path: str):
        """ Dev"""

        for item in os.listdir(dir_path):
            path = os.path.join(dir_path, item)
            if path.endswith(".flac"):
                data, samplerate = sf.read(path)
                print(item)
                print(samplerate, data.dtype)
                print()
