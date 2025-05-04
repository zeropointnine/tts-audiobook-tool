import os
import subprocess
from pathlib import Path

from tts_audiobook_tool.hash_file_util import HashFileUtil
from .constants import *
from .generate_util import *

class ConcatUtil:

    @staticmethod
    def concatenate_flacs(state: State, ffmpeg_path: str="ffmpeg"):

        project_dir = state.project_dir
        voice = cast(dict, state.voice)

        fn = HashFileUtil.make_concat_file_name(state.text_segments, voice)
        dest_flac_path = os.path.join(project_dir, fn)

        printt(f"Combining audio segment files\n")
        
        dic = ProjectDirUtil.get_audio_segment_files(state)
        sorted_indices = list(dic.keys())
        sorted_indices.sort()
        wav_paths = [dic[index] for index in sorted_indices]

        num_missing = len(state.text_segments) - len(wav_paths)
        if num_missing > 0:
            printt(f"Warning: Missing {num_missing} files.") 
            # TODO enumerate missing indices and text fragments
            if not ask_confirm(f"Press {make_hotkey_string("Y")} to continue anyway: "):
                return

        # Make temp file with list of input WAVs for ffmpeg's concat demuxer
        temp_text_path = os.path.join(project_dir, PROJECT_FFMPEG_TEMP_FILE_NAME)
        try:
            with open(temp_text_path, 'w', encoding='utf-8') as f:
                for path in wav_paths:
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
        temp_flac_path = os.path.join(project_dir, make_random_hex_string() + ".flac")
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

        # Concat/compress:
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
            printt(f"{COL_ERROR}FFmpeg command failed with exit code {e.returncode}") 
            printt(str(e.stderr), type="error") 
            return
        except Exception as e: 
            delete_temp_file(temp_text_path)
            printt(f"An unexpected error occurred running FFmpeg: {e}", type="error")
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
        printt("-" * len(s))
        printt(s)
        printt("-" * len(s) + "\n")
        ask("Press enter: ")
