import glob
import os
import subprocess
from typing import cast

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.hash_file_util import HashFileUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def concatenate_project_flacs(state: State):

        dic = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        sorted_indices = list(dic.keys())
        sorted_indices.sort()
        file_paths = [dic[index] for index in sorted_indices] # TODO need filenames probably
        file_names = [os.path.basename(file_path) for file_path in file_paths]

        num_missing = len(state.text_segments) - len(file_paths)
        if num_missing > 0:
            printt(f"Warning: Missing {num_missing} files.")
            # TODO enumerate missing indices and text fragments
            if not ask_confirm(f"Press {make_hotkey_string("Y")} to continue anyway: "):
                return

        voice = cast(dict, state.voice)
        dest_file_name = HashFileUtil.make_concat_file_name(state.text_segments, voice)

        ConcatUtil.concatenate_flacs(state.project_dir, file_names, dest_file_name)

    @staticmethod
    def concatenate_flacs(
        base_dir: str,
        source_file_names: list[str],
        dest_file_name: str,
        emphasize_finished: bool=True,
        ffmpeg_path: str="ffmpeg"
    ):
        """ Assumes valid dir path """

        base_dir = os.path.abspath(base_dir)

        dest_flac_path = os.path.join(base_dir, dest_file_name)

        printt(f"Combining {len(source_file_names)} audio segment files\n")

        file_paths = [os.path.join(base_dir, file_name) for file_name in source_file_names]

        # Make temp file with list of input WAVs for ffmpeg's concat demuxer
        temp_text_path = os.path.join(base_dir, PROJECT_FFMPEG_TEMP_FILE_NAME)
        try:
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                for path in file_paths:
                    # Must be absolute paths
                    # Important: Escape backslashes for ffmpeg on all platforms within the list file
                    # And wrap the path in single quotes for robustness (handles spaces etc.)
                    escaped_path = path.replace('\\', '\\\\')
                    f.write(f"file '{escaped_path}'\n")

        except Exception as e:
            delete_temp_file(temp_text_path)
            printt(str(e), "error")
            return

        # Construct the ffmpeg command
        # -y: Overwrite output file without asking
        # -f concat: Use the concatenation demuxer
        # -safe 0: Allow unsafe file paths (useful if paths are complex, though quoting helps)
        # -i list_filename: Input file is the list we created
        # -c:a flac: Specify the audio codec for the output as FLAC
        temp_flac_path = os.path.join(base_dir, make_random_hex_string() + ".flac")
        command = [
            ffmpeg_path,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_text_path,
            "-c:a", "flac",
            temp_flac_path
        ]

        # Concat:
        # printt(f"Running FFmpeg command:\n{' '.join(command)}")
        try:
            # capture_output=True hides ffmpeg's verbose output from the console by default
            # text=True decodes stdout/stderr as text
            completed_process = subprocess.run(
                command,
                check=True,  # Raise CalledProcessError if ffmpeg returns non-zero exit code
                capture_output=True,
                text=True,
                encoding='utf-8' # Explicitly set encoding might help on some systems
            )
            if completed_process.returncode != 0:
                printt(f"Bad return code: {completed_process.returncode}", type="error")
                return

        except subprocess.CalledProcessError as e:
            delete_temp_file(temp_text_path)
            printt(f"{COL_ERROR}Aborting. FFmpeg command failed with exit code {e.returncode}")
            printt(str(e.stderr), type="error")
            return
        except Exception as e:
            delete_temp_file(temp_text_path)
            printt(f"Aborting. An unexpected error occurred running FFmpeg: {e}", type="error")
            return

        try:
            if os.path.exists(dest_flac_path):
                # Delete any pre-existing dest file
                os.unlink(dest_flac_path)
            os.rename(temp_flac_path, dest_flac_path)
        except Exception as e:
            printt(str(e), type="error")
            # Don't delete temp file in this case
            return

        delete_temp_file(temp_text_path)
        s = f"Finished: {COL_ACCENT}{dest_flac_path}"
        if emphasize_finished:
            text_len = len(strip_ansi_codes(s))
            printt("-" * text_len)
            printt(s)
            printt("-" * text_len + "\n")
        else:
            printt(s + "\n")

    @staticmethod
    def ask_do_dir_concat():
        path = ask("Enter directory path to concatenate FLAC files:\n")
        if not path:
            return
        elif not os.path.exists(path):
            ask("Bad directory. Press enter: ")
            return
        flac_file_names = list(glob.iglob(f"{path}/*.[fF][lL][aA][cC]"))
        if len(flac_file_names) <= 1:
            ask("Two or more FLAC files required. Press enter: ")
            return

        dest_fn = f"combined {AppUtil.make_timestamp_string()}.flac"
        ConcatUtil.concatenate_flacs(path, flac_file_names, dest_fn, emphasize_finished=False)


    @staticmethod
    def trim_flac_file(
            source_flac_path: str,
            dest_file_path: str,
            start_time_seconds: float,
            end_time_seconds: float,
            ffmpeg_path: str="ffmpeg"
        ) -> bool:
            """
            Trims a source FLAC file from start_time_seconds to end_time_seconds
            and saves it to dest_file_path using ffmpeg.
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

            # Construct the ffmpeg command
            # -y: Overwrite output file without asking
            # -i: Input file
            # -ss: Start time
            # -to: End time (alternatively, -t for duration)
            # -c:a flac: Specify the audio codec for the output as FLAC
            command = [
                ffmpeg_path,
                "-hide_banner",
                "-loglevel", "error",
                "-y",
                "-i", source_flac_path,
                "-ss", str(start_time_seconds),
                "-to", str(end_time_seconds),
                "-c:a", "flac",
                dest_file_path
            ]

            try:
                completed_process = subprocess.run(
                    command,
                    check=True,  # Raise CalledProcessError if ffmpeg returns non-zero exit code
                    capture_output=True,
                    text=True,
                    encoding='utf-8'
                )
                if completed_process.returncode != 0:
                    L.w(f"ffmpeg fail, returncode - {completed_process.returncode}")
                    return False
                return True
            except subprocess.CalledProcessError as e:
                L.w(f"ffmpeg fail, returncode - {e.returncode} - {e.stderr}")
                return False
            except Exception as e:
                L.w(f"subprocess fail, ffmpeg - {e}")
                return False
