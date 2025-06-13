import os
from typing import cast
from pathlib import Path

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment, TextSegmentReason
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class ConcatSubmenu:

    @staticmethod
    def concat_submenu(state: State) -> None:

        print_heading(f"Combine audio segments:")

        infos = ChapterInfo.make_chapter_infos(state)
        if len(infos) > 1:
            print_chapter_segment_info(infos)

        if not state.project.section_dividers:
            chapter_dividers_string = "none"
        else:
            strings = [str(item) for item in state.project.section_dividers]
            chapter_dividers_string = ", ".join(strings)

        printt(f"{make_hotkey_string('1')} Combine to FLAC, and transcode to MP4")
        printt(f"{make_hotkey_string('2')} Combine to FLAC only")
        printt(f"{make_hotkey_string('3')} Define file cut points {COL_DIM}(current: {chapter_dividers_string})")
        printt(f"{make_hotkey_string('4')} Trim sentence continuation pauses {COL_DIM}(current: {state.prefs.optimize_segment_silence})")
        printt()
        hotkey = ask_hotkey()

        if hotkey == "1":
            ConcatSubmenu.ask_chapters(infos, state, and_transcode=True)
        elif hotkey == "2":
            ConcatSubmenu.ask_chapters(infos, state, and_transcode=False)
        elif hotkey == "3":
            ConcatSubmenu.ask_cut_points(state)
            ConcatSubmenu.concat_submenu(state)
        elif hotkey == "4":
            state.prefs.optimize_segment_silence = not state.prefs.optimize_segment_silence
            printt(f"Optimize silence duration between segments: {state.prefs.optimize_segment_silence}") # TODO needs some explanation
            printt()
            ConcatSubmenu.concat_submenu(state)


    @staticmethod
    def ask_cut_points(state: State) -> None:

        print_heading("Combine > File cut points:")

        num_text_segments = len(state.project.text_segments)

        section_dividers = state.project.section_dividers
        if section_dividers:
            print_concat_info(section_dividers, num_text_segments)

        printt("Enter the line numbers where new files will begin.")
        printt("For example, if there are 1000 lines of text and you enter \"250, 700\",")
        printt("three audio files will be created spanning lines 1-249, 250-699, and 700-1000.")
        printt("Enter \"1\" for no cut points.")
        printt()
        inp = ask()
        printt()
        if not inp:
            return

        string_items = inp.split(",")
        one_indexed_items = []
        for string_item in string_items:
            try:
                index = int(string_item)
                one_indexed_items.append(index)
            except:
                printt(f"Parse error: {string_item}", "error")
                return
        one_indexed_items = list(set(one_indexed_items))
        one_indexed_items.sort()
        for item in one_indexed_items:
            if item < 1 or item > len(state.project.text_segments):
                printt(f"Index out of range: {item}", "error")
                return
        zero_indexed_items = [item - 1 for item in one_indexed_items]
        if 0 in zero_indexed_items:
            del zero_indexed_items[0]
        state.project.section_dividers = zero_indexed_items
        state.project.save()



    @staticmethod
    def ask_chapters(infos: list[ChapterInfo], state: State, and_transcode: bool) -> None:

        # Chapter indices that have any files
        chapter_indices = []
        for i, info in enumerate(infos):
            if info.num_files_exist > 0:
                chapter_indices.append(i)

        if len(chapter_indices) > 1:
            printt("Enter file numbers for which you want to combine audio segments")
            printt("(For example: \"1, 2, 4\" or  \"2-5\", or \"all\")")
            inp = ask()
            if inp == "all" or inp == "a":
                selected_chapter_indices = chapter_indices.copy()
            else:
                input_indices, warnings = ParseUtil.parse_one_indexed_ranges_string(inp, len(state.project.text_segments))
                if warnings:
                    for warning in warnings:
                        printt(warning)
                    printt()
                    return
                if not input_indices:
                    return
                input_indices = [item - 1 for item in input_indices] # make zero-indexed
                selected_chapter_indices = [item for item in input_indices if item in chapter_indices]

                if not selected_chapter_indices:
                    ask("No valid chapters numbers entered. Press enter: ")
                    return

        else:
            selected_chapter_indices = [0]

        strings = [str(index+1) for index in selected_chapter_indices]
        string = ", ".join(strings)
        printt(f"Will concatenate audio segments to create the following chapters: {string}")
        printt()
        b = ask_confirm()
        if not b:
            return

        timestamp_subdir = timestamp_string()
        dest_subdir = os.path.join(state.project.dir_path, CONCAT_SUBDIR, timestamp_subdir)
        try:
            os.makedirs(dest_subdir, exist_ok=True)
        except:
            printt(f"Couldn't make directory {dest_subdir}", "error")
            return

        ok = ConcatUtil.concatenate_chapters(
            chapter_indices=selected_chapter_indices,
            state=state,
            and_transcode=and_transcode,
            base_dir=dest_subdir
        )

        if ok and not state.prefs.has_shown_player_reminder:
            AppUtil.show_player_reminder(state.prefs)

        if not ok or not has_gui():
            ask_continue("Finished.")
        else:
            printt("Finished.")
            printt()
            hotkey = ask_hotkey(f"Press {make_hotkey_string("Enter")}, or press {make_hotkey_string("O")} to open output directory: ")
            if hotkey == "o":
                err = open_directory_gui(dest_subdir)
                if err:
                    printt(err, "error")

# ---

def print_concat_info(section_dividers: list[int], num_items: int) -> None:
    section_index_strings = [str(index+1) for index in section_dividers]
    section_indices_string = ", ".join(section_index_strings)
    ranges = make_section_ranges(section_dividers, num_items)
    range_strings = [ str(range[0]+1) + "-" + str(range[1]+1) for range in ranges]
    ranges_string = ", ".join(range_strings)
    printt(f"Current cut points: {section_indices_string} {COL_DIM}({ranges_string})")
    printt()


def print_chapter_segment_info(infos: list[ChapterInfo]) -> None:
    for i, info in enumerate(infos):
        if info.num_files_missing == 0:
            desc = ""
        else:
            desc = f"{info.num_files_missing} of {info.num_segments} files missing"
        s = f"File {i+1}: lines {info.segment_index_start + 1}-{info.segment_index_end + 1}"
        if desc:
            s += f" {COL_DIM}({desc})"
        printt(s)
    printt()

