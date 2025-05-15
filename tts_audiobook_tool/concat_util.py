import os
import subprocess
from typing import cast
from pathlib import Path

from tts_audiobook_tool.l import L
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def concatenate_project_flacs(state: State):
        """
        Concats project audio segments into final flac file/s in the `concat` subdir
        If files missing for whatever reason, prompts to continue
        Prompts at end.
        """

        print_heading(f"Concatenate audio segments:")

        segment_index_to_file_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        segment_index_ranges = make_section_ranges(state.project.section_dividers, len(state.project.text_segments))

        # Parallel liss
        sections = []
        section_filenames = []
        warning = ""

        for i, rang in enumerate(segment_index_ranges):

            start_index = rang[0]
            end_index = rang[1]

            segment_file_paths = []
            num_missing = 0
            for segment_index in range(start_index, end_index + 1):
                segment_file_path = segment_index_to_file_path.get(segment_index, "")
                if segment_file_path:
                    segment_file_paths.append(segment_file_path)
                else:
                    num_missing += 1

            if num_missing > 0:
                if len(segment_index_ranges) > 1:
                    warning += f"Chapter file {i+1} is missing {num_missing} audio segments\n"
                else:
                    warning += f"Missing {num_missing} audio segments\n"

            if len(segment_file_paths) == 0:
                continue

            sections.append(segment_file_paths)

            # Make Filename
            # project dir base name
            filename = sanitize_for_filename( Path(state.prefs.project_dir).name[:20] ) + " "
            # file number
            filename += f"[{ i+1 } of {len(segment_index_ranges)}]" + " "
            # text segment range
            filename += f"[{start_index}-{end_index}]" + " "
            # num missing
            if num_missing > 0:
                filename += f"[{num_missing} missing]" + " "
            # voice
            voice = cast(dict, state.project.voice)
            filename += f"[{voice.get("identifier", "voice")}]"
            filename += ".flac"
            section_filenames.append(filename)

        if warning:
            printt(warning)
            b = ask_confirm("Continue anyway? ")
            if not b:
                return

        timestamp_subdir = timestamp_string()

        for i in range(len(sections)):
            printt(f"Creating file {i+1} of {len(sections)}\n")
            filename = section_filenames[i]
            segment_file_paths = sections[i]
            dest_file_path = os.path.join(state.project.dir_path, CONCAT_SUBDIR, timestamp_subdir, filename)
            ConcatUtil.concatenate_flacs(file_paths=segment_file_paths, dest_flac_path=dest_file_path)

        ask("Finished. Press enter:")

    @staticmethod
    def concatenate_flacs(
        file_paths: list[str],
        dest_flac_path: str,
        emphasize_finished: bool=True,
        ffmpeg_path: str="ffmpeg"
    ):
        printt(f"Concatenating {len(file_paths)} audio segment files\n")

        dest_base_path = str( Path(dest_flac_path).parent )
        os.makedirs(dest_base_path, exist_ok=True)

        # Make temp file with list of input WAVs for ffmpeg's concat demuxer
        temp_text_path = os.path.join(dest_base_path, PROJECT_FFMPEG_TEMP_FILE_NAME)
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
        temp_flac_path = os.path.join(dest_base_path, make_random_hex_string() + ".flac")
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

        s = "Finished concatenating audio file:\n"
        s += f"{COL_ACCENT}{dest_flac_path}"
        if emphasize_finished:
            length = len(dest_flac_path)
            printt("-" * length)
            printt(s)
            printt("-" * length)
        else:
            printt(s)
        printt()

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


        # @staticmethod
        # def make_concat_file_path(state: State) -> str:
        #     fn = HashFileUtil.make_concat_file_name(state.project.text_segments, cast(dict, state.project.voice))
        #     return os.path.join(state.prefs.project_dir, fn)

        # @staticmethod
        # def make_concat_file_name(text_segments: list[str], voice: dict) -> str:
        #     hash = HashFileUtil.calc_full_hash(text_segments, voice)
        #     return  f"combined [{voice.get("identifier", "")}] [{hash}] {timestamp_string()}.flac"

        # @staticmethod
        # def does_concat_file_exist(state: State) -> bool:
        #     fn = HashFileUtil.make_concat_file_name(state.project.text_segments, cast(dict, state.project.voice))
        #     file_path = os.path.join(state.prefs.project_dir, fn)
        #     path = Path(file_path)
        #     if path.exists:
        #         if path.stat().st_size > 0:
        #             return True
        #     return False

