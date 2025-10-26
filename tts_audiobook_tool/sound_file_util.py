import os
import time
from typing import Any
import librosa
import sounddevice as sd
import numpy as np
import threading
import soundfile

from tts_audiobook_tool.app_types import FailResult, Sound, ValidationResult
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.util import *

class SoundFileUtil:

    _current_playback_thread = None
    _stop_playback_event = threading.Event()
    debug_save_dir: str = AppUtil.get_app_temp_dir()

    @staticmethod
    def load(path: str, target_sr: int=0) -> Sound | str:
        """
        Returns data as normalized np32 floats, mono
        """
        try:
            data, sr = librosa.load(path)
            sr = int(sr)
        except Exception as e:
            return make_error_string(e)

        if target_sr != 0 and target_sr != sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
            return Sound(data, target_sr)
        else:
            return Sound(data, sr)

    @staticmethod
    def save_flac(sound: Sound, flac_path: str) -> str:
        """ Return error string on fail """
        try:
            soundfile.write(
                flac_path,
                sound.data,
                sound.sr,
                format="FLAC",
                subtype="PCM_16"
            )
            return ""
        except Exception as e:
            return f"Error saving sound file {flac_path}: {type(e)} {e}"

    @staticmethod
    def debug_save(label: str, sound: Any): # sound = Sound

        if not DEV_SAVE_INTERMEDIATE_FILES:
            return

        fn = f"{int(time.time()*1000)} {label}.flac"
        path = os.path.join(SoundFileUtil.debug_save_dir, fn)
        SoundFileUtil.save_flac(sound, path)

    @staticmethod
    def debug_save_result_info(result: ValidationResult, source_text: str, transcribed_text: str):

        if not DEV_SAVE_INTERMEDIATE_FILES:
            return

        s = type(result).__name__
        if isinstance(result, FailResult):
            s = "[fail] " + s
        fn = f"{int(time.time()*1000)} {s}.txt"
        path = os.path.join(SoundFileUtil.debug_save_dir, fn)

        text = type(result).__name__ + "\n" + result.get_ui_message() + "\n\n\n"
        text += "source text:" + "\n" + source_text + "\n\n"
        text += "transcribed_text:" + "\n" + transcribed_text + "\n\n"
        source_text_massaged = massage_for_text_comparison(source_text)
        text += "source_text, massaged:" + "\n" + source_text_massaged + "\n\n"
        transcribed_text_massaged = massage_for_text_comparison(transcribed_text)
        text += "transcribed_text, massaged:" + "\n" + transcribed_text_massaged + "\n\n"

        with open(path, 'w', encoding='utf-8') as f:
            f.write(text)

    @staticmethod
    def is_valid_sound_file(path: str) -> str:
        """ Returns error string or empty string"""
        from mutagen._file import File
        try:
            File(path)
        except Exception as e:
            return "Not a valid audio file: {e}"
        return ""

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
            delete_silently(temp_text_path)
            return make_error_string(e)

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
        partial_command.extend(FFMPEG_ARGUMENTS_OUTPUT_FLAC)

        err = FfmpegUtil.make_file(partial_command, dest_flac_path, use_temp_file=True)
        delete_silently(temp_text_path)

        return err

    @staticmethod
    def transcode_to_aac(source_file_path: str) -> tuple[str, str]:
        """
        Transcodes an audio file to AAC/MP4 format.
        Returns saved file name and error string, mutually exclusive
        """

        # TODO i refactored things, re-verify

        path = Path(source_file_path)
        if path.suffix in AAC_SUFFIXES:
            return "", "Is already an mp4/m4a file"

        dest_file_path = str(path.with_suffix(".m4a"))
        dest_file_path = get_unique_file_path(dest_file_path)

        partial_command = FFMPEG_TYPICAL_OPTIONS[:]
        partial_command.extend(["-i", source_file_path])
        partial_command.extend(FFMPEG_ARGUMENTS_OUTPUT_AAC[:])
        partial_command.append("-")

        err = FfmpegUtil.make_file(partial_command, dest_file_path, False)
        if err:
            return "", err

        return dest_file_path, ""

    @staticmethod
    def play_flac_async(file_path: str):
        """
        Plays a FLAC file asynchronously using a streaming approach.
        A new playback will cancel the previous one.
        Eats exceptions and prints them.

        # TODO play in-memory sound data
        """
        # Signal previous thread to stop if it exists and is alive
        if SoundFileUtil._current_playback_thread and SoundFileUtil._current_playback_thread.is_alive():
            SoundFileUtil._stop_playback_event.set() # Set the event to signal stopping
            SoundFileUtil._current_playback_thread.join(timeout=0.5) # Wait a short time for it to stop

        # Clear the event for the new playback
        SoundFileUtil._stop_playback_event.clear()

        def _play_stream(stop_event: threading.Event):
            try:
                with soundfile.SoundFile(file_path, 'r') as flac_file:
                    stream_finished_event = threading.Event()

                    def callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
                        if stop_event.is_set():
                            raise sd.CallbackStop

                        data_read = flac_file.read(frames, dtype='float32', always_2d=True)

                        if data_read.shape[0] > 0:
                            outdata[:data_read.shape[0]] = data_read
                            if data_read.shape[0] < frames:
                                outdata[data_read.shape[0]:] = 0
                        else:
                            raise sd.CallbackStop

                    def set_event_on_finish():
                        stream_finished_event.set()

                    with sd.OutputStream(
                        samplerate=flac_file.samplerate,
                        channels=flac_file.channels,
                        callback=callback,
                        dtype='float32',
                        finished_callback=set_event_on_finish
                    ):
                        stream_finished_event.wait() # Wait for playback to complete or be cancelled

            except sd.CallbackStop:
                pass # Normal exit from the stream
            except Exception as e:
                printt(f"Couldn't play FLAC audio file via stream: {file_path} - {e}")

        new_thread = threading.Thread(target=_play_stream, args=(SoundFileUtil._stop_playback_event,), daemon=True)
        new_thread.start()
        SoundFileUtil._current_playback_thread = new_thread

    @staticmethod
    def stop_sound_async() -> bool:
        """
        Stops a sound that is playing using `play_sound_async()`.
        Returns True if there was a sound to be stopped.
        """

        # Signal previous thread to stop if it exists and is alive
        if SoundFileUtil._current_playback_thread and SoundFileUtil._current_playback_thread.is_alive():
            SoundFileUtil._stop_playback_event.set() # Set the event to signal stopping
            SoundFileUtil._current_playback_thread.join(timeout=0.5) # Wait a short time for it to stop
            return True
        else:
            return False

    @staticmethod
    def play_sound_async(sound: Sound):
        """
        Plays in-memory sound data asynchronously.
        A new playback will cancel the previous one.
        Eats exceptions and prints them.
        """
        SoundFileUtil.stop_sound_async()

        # Clear the event for the new playback
        SoundFileUtil._stop_playback_event.clear()

        def _play_stream_from_data(sound_data: np.ndarray, samplerate: int, channels: int, stop_event: threading.Event):
            try:
                stream_finished_event = threading.Event()
                current_frame = 0

                def callback(outdata: np.ndarray, frames: int, time, status: sd.CallbackFlags):
                    nonlocal current_frame
                    if stop_event.is_set():
                        raise sd.CallbackStop

                    # Calculate how many frames are left in the sound_data
                    frames_to_read = min(frames, len(sound_data) - current_frame)

                    if frames_to_read > 0:
                        # Copy data from sound_data to outdata
                        outdata[:frames_to_read] = sound_data[current_frame : current_frame + frames_to_read]
                        # Fill remaining part of outdata with zeros if less data was read than requested
                        if frames_to_read < frames:
                            outdata[frames_to_read:] = 0
                        current_frame += frames_to_read
                    else:
                        # No more data to play
                        raise sd.CallbackStop

                def set_event_on_finish():
                    stream_finished_event.set()

                with sd.OutputStream(
                    samplerate=samplerate,
                    channels=channels,
                    callback=callback,
                    dtype='float32',
                    finished_callback=set_event_on_finish
                ):
                    stream_finished_event.wait() # Wait for playback to complete or be cancelled

            except sd.CallbackStop:
                pass # Normal exit from the stream
            except Exception as e:
                printt(f"Couldn't play in-memory audio data: {e}")

        # Ensure sound_data is 2D for sounddevice (frames, channels)
        # If it's 1D (mono), convert it to 2D
        sound_data_2d = sound.data.reshape(-1, 1) if sound.data.ndim == 1 else sound.data
        channels = sound_data_2d.shape[1]

        new_thread = threading.Thread(target=_play_stream_from_data, args=(sound_data_2d, sound.sr, channels, SoundFileUtil._stop_playback_event,), daemon=True)
        new_thread.start()
        SoundFileUtil._current_playback_thread = new_thread


    @staticmethod
    def print_samplerates(dir_path: str):
        """ Dev"""

        for item in os.listdir(dir_path):
            path = os.path.join(dir_path, item)
            if path.endswith(".flac"):
                data, samplerate = soundfile.read(path)
                print(item)
                print(samplerate, data.dtype)
                print()
