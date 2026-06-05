from datetime import datetime
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy import ndarray

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import hints
from tts_audiobook_tool.app_types import SectionMarkerMode, ExportType, HighShelfEq, NormalizationType
from tts_audiobook_tool import ask
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.sound.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.sound import m4b_chapter_util
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound.sidon_util import SidonUtil
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil, get_segment_stt_info_path
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_types.app_metadata import AppMetadata, AppMetadataSection
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.state import State
from tts_audiobook_tool.app_types.phrase import Phrase
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.app_types.output_range_info import OutputRangeInfo
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def make_files(
        state: State, 
        file_cut_indices: list[int], 
        bookmark_indices: list[int]
    ) -> None:
        """
        Creates a final, concatenated audio file for each cut point index given.
        Else, creates a single concatenated audio file for the entire book.

        `file_cut_indices` and `bookmark_indices` are mutually exclusive.
        """

        if file_cut_indices and bookmark_indices:
            raise ValueError(f"file_cut_indices and bookmark_indices are mutually exclusive: {file_cut_indices} vs {bookmark_indices}")

        start_time = time.time()
        
        # Preflight checks
        if state.project.use_upsampler:
            import torch
            if not torch.cuda.is_available():
                printt(f"{COL_DIM_ITALICS}Warning: Sidon enabled but CUDA not available")
                printt()
            if not SidonUtil.has_sidon():
                printt(f"{COL_DIM_ITALICS}Warning: Sidon enabled but Sidon library not installed")
                printt()
        if state.project.use_upsampler and ModelManager.is_any_model_loaded():
            printt(f"{COL_DIM_ITALICS}Attempting to unload models to free up VRAM for generative upsampling...{COL_DEFAULT}")
            printt()
            ModelManager.clear_all_models(except_sidon=True)

        # Make subdir
        subdir = datetime.now().strftime("%y%m%d_%H%M%S") # eg, 260518_120811
        dest_dir = os.path.join(state.project.concat_path, subdir)
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except:
            ask.ask_error(f"Couldn't make directory {dest_dir}")
            return

        if not file_cut_indices:
            file_cut_indices = [0]

        for i, file_cut_index in enumerate(file_cut_indices):

            message = "Creating concatenated audiobook file"
            if len(file_cut_indices) > 1:
                message += f" {i+1} of {len(file_cut_indices)} - output file {file_cut_index+1}"
            message += "..."
            dash_line = "-" * len(message)
            printt(f"{COL_ACCENT}{dash_line}")
            printt(f"{COL_ACCENT}{message}")
            printt()

            if state.project.chapter_mode == SectionMarkerMode.FILES:
                ranges = make_file_line_ranges(state.project.markers, len(state.project.phrase_groups))
                index_start, index_end = ranges[file_cut_index]
                num_chapters = len(ranges)
            else:
                index_start, index_end = 0, len(state.project.phrase_groups) - 1
                num_chapters = 1

            stem = make_stem(
                project=state.project, 
                index_start=index_start, index_end=index_end, 
                file_cut_index=file_cut_index, num_chapters=num_chapters
            )
            dest_stem_path = os.path.join(dest_dir, stem)

            dest_path, err = ConcatUtil.make_file(
                state=state,
                index_start=index_start,
                index_end=index_end,
                bookmark_indices=bookmark_indices,
                stem_path=dest_stem_path
            )
            if err:
                printt()
                ask.ask_error(err)
                return
            
            printt(f"{COL_ACCENT}Saved {COL_DEFAULT}{make_terminal_hyperlink(dest_path)}") 
            printt()

        # Post-concat feedback, prompt
        
        app_support.play_done_sound()
        elapsed = duration_string(time.time() - start_time)
        printt(f"Finished{COL_DIM} (elapsed: {elapsed})")
        printt()

        ModelManager.clear_sidon_upsampler()

        hints.show_player_hint_if_necessary(state.prefs)

        hotkey = ask.ask_hotkey(f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open output directory in system file explorer: ")
        printt()
        if hotkey == "o":
            err = open_directory_in_gui(dest_dir)
            if err:
                ask.ask_error(err)

    @staticmethod
    def make_file(
        state: State,
        index_start: int,
        index_end: int,
        bookmark_indices: list[int],
        stem_path: str
    ) -> tuple[str, str]:
        """
        Creates final output file, full feature flow.

        Params:
            index_end: Inclusive
            stem_path: Destination file path w/o file suffix
        
        Prints to console along the way
        Returns successful file path, error message if any
        """

        # Load raw text for app metadata. If unavailable, continue with a
        # phrase-group fallback rather than aborting concat.
        raw_text = ProjectTextIOUtil.load_raw_text(state.project)
        if not raw_text:
            raw_text = "\n".join(
                group.as_flattened_phrase().text for group in state.project.phrase_groups
            )
            if not raw_text:
                raw_text = ""

        # Intermediate file paths etc
        
        is_aac = (state.project.export_type == ExportType.AAC)
        should_normalize = (state.project.normalization_type != NormalizationType.DISABLED)

        concat_suffix = ".m4b" if is_aac and not should_normalize else ".flac" # tricky
        concat_path = stem_path + " [concat]" + concat_suffix
        
        suffix = ".m4b" if is_aac else ".flac"
        if should_normalize:
            norm_path = stem_path + " [norm]" + suffix
        else:
            norm_path = ""
        if is_aac and m4b_chapter_util.has_multiple_chapters(state.project, index_start, index_end):
            chapter_meta_path = stem_path + " [chaptermeta]" + suffix
        else:
            chapter_meta_path = ""
        final_path = stem_path + ".abr" + suffix
        last_path = ""

        def delete_intermediate_files(keep_final: bool=False) -> None:
            # TODO: add delete-as-you-go logic instead
            if state.prefs.save_debug_files:
                return
            paths = [ concat_path, norm_path, chapter_meta_path]
            if not keep_final:
                paths.append(final_path)
            for path in paths:
                if path:
                    delete_silently(path)

        # [0] Prep data

        high_shelf = HighShelfEq.get_by_id(state.project.high_shelf) or HighShelfEq.DISABLED

        # Make phrase/path list # TODO: Duplicated logic; refactor OutputRangeInfo and add phrase info etc
        phrases_and_paths = ConcatUtil.make_phrases_and_paths(
            state.project, index_start, index_end
        )
                    
        # [1] Concatenated audio file

        result = ConcatUtil.concatenate_sound_segments(
            concat_path,
            phrases_and_paths,
            print_progress=True,
            use_break_sound_effect=state.project.use_break_sound_effect,
            high_shelf=high_shelf,
            aac_bitrate=state.prefs.aac_bitrate,
            use_upsampler=state.project.use_upsampler
        )
        if isinstance(result, str): # is error
            delete_intermediate_files()
            return "", result
        else:
            durations = result
            last_path = concat_path

        # [2] Loudness-normalized file
        #     Ideally, this would go at the very end due to taking the most time but can't be helped

        if norm_path:
            err = LoudnessNormalizationUtil.normalize_file(
                source_flac=last_path, 
                specs=state.project.normalization_type.value, 
                dest_path=norm_path,
                aac_bitrate=state.prefs.aac_bitrate
            )
            if err:
                delete_intermediate_files()
                return "", err
            else:
                last_path = norm_path

        # [3] Chapter (M4B) metadata
        #     Must be done before custom metadata
        #     Also, can't be easily combined into other save operation
        #     Both due to ffmpeg limitations

        if chapter_meta_path:
            chapter_metadata = m4b_chapter_util.make_metadata(
                state.project,
                durations,
                file_title=Path(stem_path).name,
                index_start=index_start,
                index_end=index_end,
            )
            err = m4b_chapter_util.make_copy_with_metadata(
                source_path=last_path, dest_path=chapter_meta_path, metadata=chapter_metadata
            )
            if err:
                delete_intermediate_files()
                return "", f"Error making file with chapter metadata: {err}"
            else:
                last_path = chapter_meta_path

        # [4] App metadata (final file)

        phrases = [item[0] for item in phrases_and_paths]
        timed_phrases = TimedPhrase.make_list_using(phrases, durations)        
        phrase_to_text_segment_start_indices = list(range(len(timed_phrases)))
        if state.project.subdivide_phrases:
            file_paths = [item[1] for item in phrases_and_paths]
            timed_phrases, bookmark_indices, phrase_to_text_segment_start_indices = make_subdivided_timed_phrases(
                timed_phrases=timed_phrases, 
                sound_paths=file_paths, 
                sound_durations=durations, 
                bookmark_indices=bookmark_indices
            )
        sections = make_app_metadata_sections(
            project=state.project,
            index_start=index_start,
            index_end=index_end,
            phrase_to_text_segment_start_indices=phrase_to_text_segment_start_indices,
            text_segment_count=len(timed_phrases),
        )
        app_meta = AppMetadata(
            timed_phrases=timed_phrases,
            title=state.project.book.title,
            version=ABR_VERSION,
            raw_text=raw_text, 
            bookmark_indices=bookmark_indices,
            has_break_audio=state.project.use_break_sound_effect,
            project_snapshot=ProjectSerializationUtil.to_snapshot_dict(state.project),
            sections=sections,
        )
        if DEV or state.prefs.save_debug_files:
            debug_json_path = stem_path + ".abr.metadata.json"
            err = save_abr_metadata_debug_json(app_meta, debug_json_path)
            if err:
                L.w(f"Couldn't save ABR metadata debug JSON: {err}")
        if is_aac:
            err = AppMetadata.save_to_mp4(app_meta, last_path, final_path)
        else:
            err = AppMetadata.save_to_flac(app_meta, last_path, final_path)
        if err:
            delete_intermediate_files()
            return "", err
        else:
            delete_intermediate_files(keep_final=True)
            return final_path, "" # success

    @staticmethod
    def make_phrases_and_paths(
        project: Project, 
        index_start: int = -1, 
        index_end: int = -1
    ) -> list[tuple[Phrase, str, bool]]:
        
        if index_start == -1:
            index_start = 0
        if index_end == -1:
            index_end = len(project.phrase_groups) - 1

        phrases_and_paths: list[tuple[Phrase, str, bool]] = []
        section_start_indices = {0, *project.markers}

        for group_index, group in enumerate(project.phrase_groups):            
            phrase = group.as_flattened_phrase()
            is_first_in_section = group_index in section_start_indices
            out_of_range = (group_index < index_start or group_index > index_end)
            if out_of_range:
                file_path = ""
            else:
                file_name = project.sound_segments.get_best_file_for(group_index)
                if file_name:
                    file_path = os.path.join(project.sound_segments_path, file_name)
                else:
                    file_path = ""
            phrases_and_paths.append((phrase, file_path, is_first_in_section))

        return phrases_and_paths

    @staticmethod
    def num_missing_in(
        project: Project, 
        index_start: int, 
        index_end: int
    ) -> int:
        num_missing = 0
        for group_index in range(len(project.phrase_groups)):
            out_of_range = (group_index < index_start or group_index > index_end)
            if out_of_range:
                num_missing += 1
            else:
                file_name = project.sound_segments.get_best_file_for(group_index)
                if not file_name:
                    num_missing += 1        
        return num_missing

    @staticmethod
    def concatenate_sound_segments(
        dest_path: str,
        phrases_and_paths: list[ tuple[Phrase, str, bool] ],
        use_break_sound_effect: bool,
        high_shelf: HighShelfEq,
        print_progress: bool,
        aac_bitrate: str=AAC_BITRATE_DEFAULT,
        use_upsampler: bool = False
    ) -> list[float] | str:
        """
        Concatenates a list of files to a destination file using ffmpeg streaming process.
        Adds silence or sound effect between adjacent segments based on phrase "reason".

        :param aac_bitrate: 
            Only relevant if dest_path suffix is .m4a/.m4b; ignored otherwise. 
            Must be a valid AAC bitrate string like "128k".

        Returns list of float durations of each added segment to be used for app metadata .

        On error, returns error string
        """

        duration_sum = 0
        durations = []

        to_aac_not_flac = dest_path.lower().endswith(tuple(AAC_SUFFIXES))
        process = ConcatUtil.init_ffmpeg_stream(dest_path, to_aac_not_flac, aac_bitrate)

        Interrupts().set("concat")

        for (phrase, path, is_first_in_section) in phrases_and_paths:

            if not path:
                durations.append(0)
                continue
            if Interrupts().did_interrupt:
                Interrupts().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            result = SoundPipeline.make_concat_rendered_sound_segment(
                phrase, path, use_break_sound_effect, high_shelf,
                is_first_in_section=is_first_in_section,
                use_upsampler=use_upsampler
            )
            if isinstance(result, str): # error
                ConcatUtil.close_ffmpeg_stream(process) # TODO clean up more and message user
                return result
            
            sound = result
            durations.append(sound.duration)
            duration_sum += sound.duration
            if Interrupts().did_interrupt:
                Interrupts().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            ConcatUtil.add_audio_to_ffmpeg_stream(process, sound.data)

            if print_progress:
                s = f"{time_stamp(duration_sum, with_tenth=False)} {Path(path).stem[:80]} ... "
                print("\x1b[1G" + s, end="\033[K", flush=True)

        if print_progress:
            printt()
        printt()

        Interrupts().clear()
        ConcatUtil.close_ffmpeg_stream(process)
        return durations

    @staticmethod
    def init_ffmpeg_stream(
            dest_path: str,
            is_aac_not_flac: bool,
            aac_bitrate: str=AAC_BITRATE_DEFAULT
    ) -> subprocess.Popen:
        """
        Initializes and returns an ffmpeg process for streaming FLAC encoding.
        """
        args = [
            "ffmpeg",
            "-y"  # Overwrite output file if it exists
        ]
        # Input stream related
        args.extend([
            "-f", "s16le",
            "-ar", f"{APP_SAMPLE_RATE}",
            "-ac", "1",
            "-i", "-"
        ])

        # Output stream related
        if is_aac_not_flac:
            args.extend(make_ffmpeg_arguments_output_aac(aac_bitrate))
        else:
            args.extend(FFMPEG_ARGUMENTS_OUTPUT_FLAC)
        args.append(dest_path)

        return subprocess.Popen(
            args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )

    @staticmethod
    def add_audio_to_ffmpeg_stream(process: subprocess.Popen, audio_data: ndarray):
        """
        Writes a chunk of audio data to the ffmpeg process's stdin.
        Converts float audio data to 16-bit PCM format.
        """
        # Assuming audio_data is float, convert to 16-bit PCM
        # This is a standard conversion for float arrays in range [-1.0, 1.0]
        # if audio_data.dtype is not float, raise ValueError("Expecting float")
        pcm_data = (audio_data * 32767).astype(np.int16) # type: ignore

        try:
            process.stdin.write(pcm_data.tobytes()) # type: ignore
        except BrokenPipeError:
            pass

    @staticmethod
    def close_ffmpeg_stream(process: subprocess.Popen):
        """
        Closes the stdin of the ffmpeg process and waits for it to terminate.
        """
        process.stdin.close()  # type: ignore
        process.wait()

    @staticmethod
    def auto_concat_after_generation(state: State) -> None:
        if state.project.chapter_mode == SectionMarkerMode.FILES and len(state.project.markers) > 0:
            infos = OutputRangeInfo.make_output_range_infos(state.project)
            output_indices = [info.output_index for info in infos if info.num_files_exist > 0]
            if not output_indices:
                print_feedback("No chapter files have generated audio", is_error=True)
                return

            ConcatUtil.make_files(
                state=state,
                file_cut_indices=output_indices,
                bookmark_indices=[]
            )
            return

        if state.project.can_use_bookmark_section_markers():
            bookmark_indices = state.project.markers
        else:
            bookmark_indices = []

        MenuUtil.print_heading(state, "Concatenating file", dont_clear=True, non_menu=True)
        ConcatUtil.make_files(
            state=state,
            file_cut_indices=[],
            bookmark_indices=bookmark_indices
        )

# ---

def make_stem(
    project: Project,
    index_start: int, 
    index_end: int,
    file_cut_index: int,
    num_chapters: int
) -> str:

    phrases_and_paths: list[tuple[Phrase, str, bool]] = []
    num_missing = 0

    for group_index, group in enumerate(project.phrase_groups):            
        phrase = group.as_flattened_phrase()
        is_first_in_section = group_index == 0 or group_index in project.markers
        out_of_range = (group_index < index_start or group_index > index_end)
        if out_of_range:
            file_path = ""
        else:
            stem = project.sound_segments.get_best_file_for(group_index)
            if stem:
                file_path = os.path.join(project.sound_segments_path, stem)
            else:
                file_path = ""
        phrases_and_paths.append((phrase, file_path, is_first_in_section))
        if file_path == "":
            num_missing += 1

    # Make Filename
    extant_file_names = [file_name for _, file_name, _ in phrases_and_paths if file_name]
    # [1] project name
    stem = app_text.sanitize_for_filename(Path(project.dir_path).name[:20]) + " "
    # [2] file number
    if num_chapters > 1:
        stem += f"[{ file_cut_index+1 } of {num_chapters}]" + " "
    # [3] line range
    if num_chapters > 1:
        stem += f"[{index_start+1}-{index_end+1}]" + " "
    # [4] num lines missing within range
    if num_missing > 0:
        stem += f"[{num_missing} missing]" + " "
    # [5] model tag
    common_model_tag = SoundSegmentUtil.get_common_model_tag(extant_file_names)
    if common_model_tag:
        stem += f"[{common_model_tag}]" + " "
    # [5] voice tag
    common_voice_tag = SoundSegmentUtil.get_common_voice_tag(extant_file_names)
    if common_voice_tag:
        stem += f"[{common_voice_tag}]" + " "

    stem = stem.strip()
    return stem

def make_subdivided_timed_phrases(
        timed_phrases: list[TimedPhrase], 
        sound_paths: list[str],
        sound_durations: list[float],
        bookmark_indices: list[int]
    ) -> tuple[ list[TimedPhrase], list[int], list[int] ]:
    """
    Uses the "forced alignment" metadata in the json files which is saved alongside the sound_paths 
    to break up the timed_phrases into smaller parts.

    The three arguments are parallel lists.

    Returns updated timed_phrases and bookmark_indices.
    """

    if not (len(timed_phrases) == len(sound_paths) == len(sound_durations)):
        raise ValueError("lists must have same lengths")

    new_timed_phrases: list[TimedPhrase] = []
    new_bookmark_indices: list[int] = []
    phrase_to_text_segment_start_indices: list[int] = []

    for i in range(0, len(timed_phrases)):
        phrase_to_text_segment_start_indices.append(len(new_timed_phrases))
        
        def add_to_new_bookmark_indices(debug_reason: str, debug_text: str) -> None:
            if i in bookmark_indices:
                new_bookmark_index = len(new_timed_phrases)
                new_bookmark_indices.append(new_bookmark_index)
                if False:
                    print("added", new_bookmark_index, "new len", len(new_bookmark_indices), debug_reason, debug_text)

        original_timed_phrase = timed_phrases[i]
        sound_path = sound_paths[i]

        if not sound_path:
            add_to_new_bookmark_indices("no-time-segment", original_timed_phrase.presentable_text)
            new_timed_phrases.append(original_timed_phrase)
            continue

        subdivided_items_json_path = get_segment_stt_info_path(sound_path)
        if not subdivided_items_json_path.exists():
            L.w(f"Missing segment timing/STT sidecar JSON: {subdivided_items_json_path}")
            add_to_new_bookmark_indices("no-subdivided-json", original_timed_phrase.presentable_text)
            new_timed_phrases.append(original_timed_phrase)
            continue

        parse_result = SegmentTranscriptUtil.load_timed_phrases(subdivided_items_json_path)
        if isinstance(parse_result, str): 
            # File/parse error; use original item
            add_to_new_bookmark_indices("parse-error", original_timed_phrase.presentable_text)
            new_timed_phrases.append(original_timed_phrase)
            continue
        else:
            subdivided_timed_phrases = parse_result

        # Finally, do subdivision action
        offset = original_timed_phrase.time_start
        for subdivided_index, item in enumerate(subdivided_timed_phrases):
            
            if subdivided_index == 0:
                add_to_new_bookmark_indices("first-subdivision", item.presentable_text)

            if subdivided_index == 0:
                time_start = offset
            else:
                time_start = offset + subdivided_timed_phrases[subdivided_index - 1].time_end
            time_end = offset + item.time_end

            updated_item = TimedPhrase(item.text, time_start, time_end)
            new_timed_phrases.append(updated_item)
        
        # Set last item's time_end using the duration of the source audio clip
        # rather than the transcription end word timestamp
        # (prevents discontinuities in segment selectedness across boundaries, which looks distracting)
        new_timed_phrases[-1].time_end = sound_durations[i] + offset

    return new_timed_phrases, new_bookmark_indices, phrase_to_text_segment_start_indices


def make_app_metadata_sections(
    project: Project,
    index_start: int,
    index_end: int,
    phrase_to_text_segment_start_indices: list[int],
    text_segment_count: int,
) -> list[AppMetadataSection]:
    sections: list[AppMetadataSection] = []
    book_sections = project.book.sections
    section_ranges = ProjectBookUtil.get_section_ranges(project)

    if len(book_sections) != len(section_ranges):
        return sections

    for section, (section_start, section_end) in zip(book_sections, section_ranges):
        start_index = phrase_to_text_segment_start_indices[section_start]
        if section_end < len(phrase_to_text_segment_start_indices):
            end_index = phrase_to_text_segment_start_indices[section_end]
        else:
            end_index = text_segment_count

        sections.append(AppMetadataSection(
            title=section.title,
            start_index=start_index,
            end_index=end_index,
        ))

    return sections


def save_abr_metadata_debug_json(app_meta: AppMetadata, path: str) -> str:
    try:
        payload = json.loads(app_meta.to_json_string())
        with open(path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=4, ensure_ascii=False)
        return ""
    except Exception as e:
        return make_error_string(e)
