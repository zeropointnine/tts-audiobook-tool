import os
from pathlib import Path

import numpy as np
from numpy import ndarray

from tts_audiobook_tool.app_types import ChapterMode, ExportType, HighShelfEq, NormalizationType, Sound
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.ask_util import AskUtil
from tts_audiobook_tool.models_util import ModelsUtil
from tts_audiobook_tool.loudness_normalization_util import LoudnessNormalizationUtil
from tts_audiobook_tool.chapter_metadata import ChapterMetadata
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.phrase import Phrase, Reason
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def make_files(
        state: State, 
        chapter_indices: list[int], 
        bookmark_indices: list[int]
    ) -> None:
        """
        Creates a final, concatenated audio file for each chapter index given.
        Else, creates a single concatenated audio file for the entire book.
        `chapter_indices` and `bookmark_indices` are mutually exclusive.
        """

        if chapter_indices and bookmark_indices:
            raise ValueError(f"chapter_indices and bookmark_indices are mutually exclusive: {chapter_indices} vs {bookmark_indices}")
        
        if state.project.use_upscaler and ModelsUtil.is_any_model_loaded():
            printt(f"{COL_DIM}{Ansi.ITALICS}Attempting to unload models to free up VRAM for generative upscaling...{COL_DEFAULT}")
            printt()
            ModelsUtil.clear_all_models(except_sidon=True)

        # Make subdir
        dest_dir = os.path.join(state.project.concat_path, timestamp_string())
        try:
            os.makedirs(dest_dir, exist_ok=True)
        except:
            AskUtil.ask_error(f"Couldn't make directory {dest_dir}")
            return

        if not chapter_indices:
            chapter_indices = [0]

        for i, chapter_index in enumerate(chapter_indices):

            message = "Creating concatenated audiobook file"
            if len(chapter_indices) > 1:
                message += f" {COL_ACCENT}{i+1}{COL_DEFAULT}/{COL_ACCENT}{len(chapter_indices)}{COL_DEFAULT} - chapter file {COL_ACCENT}{chapter_index+1}{COL_DEFAULT}"
            message += "..."
            print_heading(message, dont_clear=True, non_menu=True)

            if state.project.chapter_mode == ChapterMode.FILES:
                ranges = make_chapter_ranges(state.project.section_dividers, len(state.project.phrase_groups))
                index_start, index_end = ranges[chapter_index]
                num_chapters = len(ranges)
            else:
                index_start, index_end = 0, len(state.project.phrase_groups) - 1
                num_chapters = 1

            stem = make_stem(
                project=state.project, 
                index_start=index_start, index_end=index_end, 
                chapter_index=chapter_index, num_chapters=num_chapters
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
                AskUtil.ask_error(err)
                return
            
            printt(f"Saved {COL_ACCENT}{dest_path}") 
            printt()

        # Post-concat feedback, prompt
        
        AppUtil.play_done_sound()
        printt("Finished.")
        printt()

        ModelsUtil.clear_sidon_upscaler()

        Hint.show_player_hint_if_necessary(state.prefs)

        hotkey = AskUtil.ask_hotkey(f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} to open output directory in system file explorer: ")
        printt()
        if hotkey == "o":
            err = open_directory_in_gui(dest_dir)
            if err:
                AskUtil.ask_error(err)

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
        Returns successfull file path, error message if any
        """

        # Load raw text (player no longer requires this but is still part of the 'spec')
        raw_text = state.project.load_raw_text()
        if not raw_text:
            return "", "Error loading text"

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
        if is_aac and bookmark_indices:
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

        # Make phrase/path list # TODO: Duplicated logic; refactor ChapterInfo and add phrase info etc
        phrases_and_paths = ConcatUtil.make_phrases_and_paths(
            state.project, index_start, index_end
        )
                    
        # [1] Concatenated audio file

        result = ConcatUtil.concatenate_sound_segments(
            concat_path,
            phrases_and_paths,
            print_progress=True,
            use_section_sound_effect=state.project.use_section_sound_effect,
            high_shelf=high_shelf,
            aac_bitrate=state.prefs.aac_bitrate,
            use_upscaler=state.project.use_upscaler,
            limit_silence_gaps=state.project.limit_silence_gaps
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
            chapter_metadata = ChapterMetadata.make_metadata(
                state.project, durations, file_title=Path(stem_path).name
            )
            err = ChapterMetadata.make_copy_with_metadata(
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
        if state.project.subdivide_phrases:
            file_paths = [item[1] for item in phrases_and_paths]
            timed_phrases, bookmark_indices = make_subdivided_timed_phrases(
                timed_phrases=timed_phrases, 
                sound_paths=file_paths, 
                sound_durations=durations, 
                bookmark_indices=bookmark_indices
            )
        app_meta = AppMetadata(
            timed_phrases=timed_phrases,
            raw_text=raw_text, 
            bookmark_indices=bookmark_indices,
            has_section_break_audio=state.project.use_section_sound_effect
        )
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
    ) -> list[tuple[Phrase, str]]:
        
        if index_start == -1:
            index_start = 0
        if index_end == -1:
            index_end = len(project.phrase_groups) - 1

        phrases_and_paths: list[tuple[Phrase, str]] = []

        for group_index, group in enumerate(project.phrase_groups):            
            phrase = group.as_flattened_phrase()
            out_of_range = (group_index < index_start or group_index > index_end)
            if out_of_range:
                file_path = ""
            else:
                file_name = project.sound_segments.get_best_file_for(group_index)
                if file_name:
                    file_path = os.path.join(project.sound_segments_path, file_name)
                else:
                    file_path = ""
            phrases_and_paths.append((phrase, file_path))

        return phrases_and_paths

    @staticmethod
    def num_missing_in(
        project: Project, 
        chapter_index_start: int, 
        chapter_index_end: int
    ) -> int:
        num_missing = 0
        for group_index in range(len(project.phrase_groups)):
            out_of_range = (group_index < chapter_index_start or group_index > chapter_index_end)
            if out_of_range:
                num_missing += 1
            else:
                file_name = project.sound_segments.get_best_file_for(group_index)
                if not file_name:
                    num_missing += 1        
        return num_missing

    @staticmethod
    def make_rendered_sound_segment(
        phrase: Phrase,
        path: str,
        use_section_sound_effect: bool,
        high_shelf: HighShelfEq,
        use_upscaler: bool = False,
        limit_silence_gaps: bool = False
    ) -> Sound | str:
        """
        Renders a sound segment by applying necessary transformations to the source audio file at `path`
        :param path: cannot be empty
        """

        result = SoundFileUtil.load(path)
        if isinstance(result, str): # error
            return result
        sound = result

        result = ConcatUtil.apply_segment_post_processing(
            sound,
            high_shelf=high_shelf,
            limit_silence_gaps=limit_silence_gaps,
            use_upscaler=use_upscaler,
        )
        if isinstance(result, str):
            return result
        sound = result

        sound = SoundUtil.append_pause_or_section_effect(
            sound, reason=phrase.reason, use_section_sound_effect=use_section_sound_effect
        )

        return sound

    @staticmethod
    def apply_segment_post_processing(
        sound: Sound,
        high_shelf: HighShelfEq,
        limit_silence_gaps: bool = False,
        use_upscaler: bool = False,
    ) -> Sound | str:
        """
        Standard audio post-processing chain shared by the audiobook concat
        flow and the conversation TTS flow. Applied to a TTS-generated Sound
        already in memory (no file IO).

        Order:
          1. Optional silence-gap limiting
          2. Optional generative upscaling (sidon)
          3. Resample to APP_SAMPLE_RATE
          4. High-shelf EQ (no-op if `high_shelf` is DISABLED)

        Returns the processed Sound, or an error string if upscaling was
        requested but failed.
        """
        if limit_silence_gaps:
            new_sound = SilenceUtil.limit_silence_gaps(sound, LIMIT_SILENCE_GAPS_DURATION)
            if abs(new_sound.duration - sound.duration) > 0.01: # 'epsilon'
                sound = new_sound

        if use_upscaler:
            upscaler = ModelsUtil.get_sidon_upscaler()
            if not upscaler:
                return "\"Generative upscaling\" enabled but not available"
            result = upscaler.process(sound)
            if isinstance(result, str):
                return result
            sound = result # rem, this is 48khz

        sound = SoundUtil.resample_if_necessary(sound, APP_SAMPLE_RATE)

        sound = SoundUtil.high_shelf_eq(
            sound,
            strength=high_shelf.strength,
            boost_start_hz=high_shelf.boost_start_hz,
            q_like=high_shelf.q_like,
        )

        return sound

    @staticmethod
    def get_rendered_sound_segment_durations(
        phrases_and_paths: list[ tuple[Phrase, str] ],
        use_section_sound_effect: bool,
        high_shelf: HighShelfEq,
        use_upscaler: bool = False
    ) -> list[float] | str:
        """
        Calculates the final durations of the sound segments to be concatenated.

        Currently uses the same concrete sound transform operations as the main concat function
        to ensure same duration results.
        TODO: Optimize that 
        """

        durations = []

        for (phrase, path) in phrases_and_paths:

            if not path:
                durations.append(0)
                continue

            result = ConcatUtil.make_rendered_sound_segment(phrase, path, use_section_sound_effect, high_shelf, use_upscaler)
            if isinstance(result, str): # error
                return result

            sound = result
            durations.append(sound.duration)

        return durations

    @staticmethod
    def concatenate_sound_segments(
        dest_path: str,
        phrases_and_paths: list[ tuple[Phrase, str] ],
        use_section_sound_effect: bool,
        high_shelf: HighShelfEq,
        print_progress: bool,
        aac_bitrate: str=AAC_BITRATE_DEFAULT,
        use_upscaler: bool = False,
        limit_silence_gaps: bool = False
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

        SigIntHandler().set("concat")

        for (phrase, path) in phrases_and_paths:

            if not path:
                durations.append(0)
                continue

            if SigIntHandler().did_interrupt:
                SigIntHandler().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            result = ConcatUtil.make_rendered_sound_segment(
                phrase, path, use_section_sound_effect, high_shelf, use_upscaler, limit_silence_gaps
            )
            if isinstance(result, str): # error
                ConcatUtil.close_ffmpeg_stream(process) # TODO clean up more and message user
                return result
            
            sound = result
            durations.append(sound.duration)
            duration_sum += sound.duration

            if SigIntHandler().did_interrupt:
                SigIntHandler().clear()
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

        SigIntHandler().clear()
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
        pcm_data = (audio_data * 32767).astype(np.int16)

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

# ---

def make_stem(
    project: Project,
    index_start: int, 
    index_end: int,
    chapter_index: int,
    num_chapters: int
) -> str:

    phrases_and_paths: list[tuple[Phrase, str]] = []
    num_missing = 0

    for group_index, group in enumerate(project.phrase_groups):            
        phrase = group.as_flattened_phrase()
        out_of_range = (group_index < index_start or group_index > index_end)
        if out_of_range:
            file_path = ""
        else:
            stem = project.sound_segments.get_best_file_for(group_index)
            if stem:
                file_path = os.path.join(project.sound_segments_path, stem)
            else:
                file_path = ""
        phrases_and_paths.append((phrase, file_path))
        if file_path == "":
            num_missing += 1

    # Make Filename
    extant_file_names = [file_name for _, file_name in phrases_and_paths if file_name]
    # [1] project name
    stem = TextUtil.sanitize_for_filename( Path(project.dir_path).name[:20] ) + " "
    # [2] file number
    if num_chapters > 1:
        stem += f"[{ chapter_index+1 } of {num_chapters}]" + " "
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
    ) -> tuple[ list[TimedPhrase], list[int] ]:
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

    for i in range(0, len(timed_phrases)):
        
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

        subdivided_items_json_path = Path(sound_path).with_suffix(".json") # TODO rename this to .json
        if not subdivided_items_json_path.exists():
            add_to_new_bookmark_indices("no-subdivided-json", original_timed_phrase.presentable_text)
            new_timed_phrases.append(original_timed_phrase)
            continue

        try:
            with open(subdivided_items_json_path, 'r', encoding='utf-8') as file:
                json_dicts = json.load(file)
        except Exception as e:
            # File error; use original item
            add_to_new_bookmark_indices("file-error", original_timed_phrase.presentable_text)            
            new_timed_phrases.append(original_timed_phrase)
            continue

        parse_result = TimedPhrase.dicts_to_timed_phrases(json_dicts)
        if isinstance(parse_result, str): 
            # Parse error; use original item
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

    return new_timed_phrases, new_bookmark_indices
