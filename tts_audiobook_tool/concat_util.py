import os
from typing import cast
from pathlib import Path

from tts_audiobook_tool.ffmpeg_util import FfmpegUtil
from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment, TimedTextSegment
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def ask_concat(state: State) -> None:
        """
        Asks which chapters to create and concats project audio segments into final flac files.
        """

        print_heading(f"Combine audio segments:")

        segment_index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        segment_index_ranges = make_section_ranges(state.project.section_dividers, len(state.project.text_segments))
        section_indices = []

        for section_index, rang in enumerate(segment_index_ranges):

            segment_index_start = rang[0]
            segment_index_end = rang[1]

            segment_indices_and_paths: list[tuple[int, str]] = []
            num_missing = 0

            for segment_index in range(segment_index_start, segment_index_end + 1):
                segment_file_path = segment_index_to_path.get(segment_index, "")
                if segment_file_path:
                    segment_indices_and_paths.append( (segment_index, segment_file_path) )
                else:
                    num_missing += 1

            # Print info
            are_all_missing = (num_missing == segment_index_end - segment_index_start + 1)
            index_string = str(section_index+1) if not are_all_missing else "-"
            if num_missing == 0:
                desc = "ready"
            elif are_all_missing:
                desc = "no audio files generated yet"
            else:
                desc = f"{num_missing} file/s missing"
            label_col = COL_DIM if are_all_missing else COL_DEFAULT
            printt(f"{COL_DEFAULT}[{COL_ACCENT}{index_string}{COL_DEFAULT}] {label_col}segments {segment_index_start}-{segment_index_end} {COL_DIM}({desc})")

            if not segment_indices_and_paths:
                continue

            section_indices.append(section_index)

        printt()

        if len(section_indices) > 1:
            inp = ask("Enter chapter numbers (eg, \"1, 2, 5-10\", or \"all\"): ")
            if inp == "all" or inp == "a":
                selected_section_indices = section_indices.copy()
            else:
                input_indices, warnings = ParseUtil.parse_int_list(inp)
                if warnings:
                    for warning in warnings:
                        printt(warning)
                    printt()
                    return
                if not input_indices:
                    return
                input_indices = [item - 1 for item in input_indices] # make zero-indexed
                selected_section_indices = [item for item in input_indices if item in section_indices]

                if not selected_section_indices:
                    ask("No valid chapters numbers entered. Press enter: ")
                    return

        else:
            selected_section_indices = [0]

        strings = [str(index+1) for index in selected_section_indices]
        string = ", ".join(strings)
        printt(f"Will create the following chapter files: {string}")
        printt()
        b = ask_confirm()
        if not b:
            return

        ConcatUtil.concatenate_sections(selected_section_indices, state)

        if not state.prefs.has_shown_player_reminder:
            show_player_reminder(state)

        ask("Finished. Press enter:")

    @staticmethod
    def concatenate_sections(indices: list[int], state: State) -> None:

        raw_text = state.project.load_raw_text()
        if not raw_text:
            printt("Error loading text", "error")
            return

        ranges = make_section_ranges(state.project.section_dividers, len(state.project.text_segments))
        segment_index_to_path = ProjectDirUtil.get_project_audio_segment_file_paths(state)
        counter = 1
        timestamp_subdir = timestamp_string()

        for section_index in indices:

            segment_index_start, segment_index_end = ranges[section_index]
            num_missing = 0

            # Parallel lists
            file_paths = []
            text_segments = []

            for segment_index in range(segment_index_start, segment_index_end + 1):

                if not segment_index in segment_index_to_path:
                    num_missing += 1
                    continue
                file_path = segment_index_to_path[segment_index]
                file_paths.append(file_path)

                text_segment = state.project.text_segments[segment_index]
                text_segments.append(text_segment)

            dest_file_path = ConcatUtil._make_file_path(
                state,
                section_index,
                len(ranges),
                segment_index_start,
                segment_index_end,
                num_missing,
                timestamp_subdir
            )
            printt(f"Creating combined audio file ({counter}/{len(indices)}\n")
            error = ConcatUtil.make_app_flac(raw_text, file_paths, text_segments, dest_file_path)
            if error:
                printt(error, "error")
                return

            printt(f"Saved: {COL_ACCENT}{dest_file_path}")
            printt()

            counter += 1

    @staticmethod
    def _make_file_path(
        state: State,
        section_index: int,
        num_sections: int,
        segment_index_start: int,
        segment_index_end: int,
        num_missing: int,
        subdir: str
    ) -> str:

        file_name = sanitize_for_filename( Path(state.prefs.project_dir).name[:20] ) + " "
        file_name += f"[{ section_index+1 } of {num_sections}]" + " "
        file_name += f"[{segment_index_start+1}-{segment_index_end+1}]" + " "
        if num_missing > 0:
            file_name += f"[{num_missing} missing]" + " "
        voice = cast(dict, state.project.voice)
        file_name += f"[{voice.get("identifier", "voice")}]" + ".flac"

        file_path = os.path.join(state.project.dir_path, CONCAT_SUBDIR, subdir, file_name)
        return file_path

    @staticmethod
    def make_app_flac(
        raw_text: str,
        file_paths: list[str],
        text_segments: list[TextSegment],
        dest_file_path: str
    ) -> str:
        """
        param file_paths and text_segments - parallel lists

        Make "app flac" (concatenated text segment audio files plus app metadata)
        Returns error message or empty string.
        """
        if len(file_paths) != len(text_segments):
            raise ValueError("file_paths and text_segments don't match")

        # First make 'timed text segments' by getting durations of all files
        timed_text_segments = []
        seconds = 0.0
        for i in range(0, len(file_paths)):
            file_path = file_paths[i]
            text_segment = text_segments[i]
            duration = AppMetaUtil.get_flac_duration(file_path)
            if duration is None:
                return f"Couldn't get duration for {file_path}"
            timed_text_segment = TimedTextSegment.make_using(text_segment, seconds, seconds + duration)
            timed_text_segments.append(timed_text_segment)
            seconds += duration

        error = ConcatUtil.concatenate_flacs(file_paths, dest_file_path)
        if error:
            return error

        error = AppMetaUtil.set_flac_app_metadata(
            flac_path=dest_file_path,
            raw_text=raw_text,
            timed_text_segments=timed_text_segments
        )
        if error:
            return error

        return "" # success

    @staticmethod
    def concatenate_flacs(
        file_paths: list[str],
        dest_flac_path: str
    ) -> str:
        """
        Concatenates a list of flac files into one.
        Returns error message or empty string.
        """

        dest_base_path = str( Path(dest_flac_path).parent )
        os.makedirs(dest_base_path, exist_ok=True)

        # [1] Make temp text file with list of input WAVs for ffmpeg's concat demuxer

        temp_text_path = os.path.join(dest_base_path, PROJECT_FFMPEG_CONCAT_TEMP_FILE_NAME)
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
            return str(e)

        # [2] Do concat

        # -y: Overwrite output file without asking
        # -f concat: Use the concatenation demuxer
        # -safe 0: Allow unsafe file paths (useful if paths are complex, though quoting helps)
        # -i list_filename: Input file is the list we created
        # -c:a flac: Specify the audio codec for the output as FLAC
        partial_command = [
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", temp_text_path,
            "-c:a", "flac",
        ]
        err = FfmpegUtil.make_file(partial_command, dest_flac_path, use_temp_file=True)
        delete_temp_file(temp_text_path)
        return err

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
            "-hide_banner",
            "-loglevel", "error",
            "-y",
            "-i", source_flac_path,
            "-ss", str(start_time_seconds),
            "-to", str(end_time_seconds),
            "-c:a", "flac",
        ]
        err = FfmpegUtil.make_file(partial_command, dest_file_path, use_temp_file=False)
        if err:
            L.e(err)
        return not bool(err)

    @staticmethod
    def print_concat_info(section_dividers: list[int], num_items: int) -> None:
        section_index_strings = [str(index+1) for index in section_dividers]
        section_indices_string = ", ".join(section_index_strings)
        ranges = make_section_ranges(section_dividers, num_items)
        range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
        ranges_string = ", ".join(range_strings)
        printt(f"Current chapter dividers: {section_indices_string} {COL_DIM}({ranges_string})")
        printt()

    # TODO: reimplement
    # @staticmethod
    # def does_concat_file_exist(state: State) -> bool:
    #     fn = HashFileUtil.make_concat_file_name(state.project.text_segments, cast(dict, state.project.voice))
    #     file_path = os.path.join(state.prefs.project_dir, fn)
    #     path = Path(file_path)
    #     if path.exists:
    #         if path.stat().st_size > 0:
    #             return True
    #     return False

# ---

def show_player_reminder(state: State) -> None:
    printt(f"🔔 {COL_ACCENT}Reminder:")
    printt("You can use audio files with the interactive player/reader here:")
    package_dir = get_package_dir()
    if package_dir:
        browser_path = str( Path(package_dir).parent / "browser_player" / "index.html" )
    else:
        browser_path = "browser_player" + os.path.sep + "index.html"
    printt(browser_path)
    printt(f"or on the web here: {PLAYER_URL}")
    state.prefs.has_shown_player_reminder = True
