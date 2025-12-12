import os
from pathlib import Path

import numpy as np
from numpy import ndarray

from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.phrase import Phrase, Reason
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def concatenate_chapter_file(
        state: State,
        chapter_index: int,
        to_aac_not_flac: bool,
        base_dir: str
    ) -> tuple[str, str]:
        """
        Returns saved path, error message
        """

        raw_text = state.project.load_raw_text()
        if not raw_text:
            return "", "Error loading text"

        # Make tuples list
        # This list is the full length of the project.text_segments,
        # but uses empty strings for file paths outside the range of the chapter

        ranges = make_section_ranges(state.project.section_dividers, len(state.project.phrase_groups))
        sound_segments = state.project.sound_segments.sound_segments

        chapter_index_start, chapter_index_end = ranges[chapter_index]
        num_missing = 0

        # Make phrase + file path list
        phrases_and_paths: list[ tuple[Phrase, str]] = []
        for group_index, group in enumerate(state.project.phrase_groups):
            segment = group.as_flattened_phrase()
            if group_index < chapter_index_start or group_index > chapter_index_end:
                # Out of range
                phrases_and_paths.append((segment, ""))
                continue
            if not group_index in sound_segments:
                # Audio segment file missing / not yet generated
                phrases_and_paths.append((segment, ""))
                num_missing += 1
                continue
            file_path = sound_segments[group_index]
            phrases_and_paths.append((segment, file_path))

        # Filename
        extant_paths = [item[1] for item in phrases_and_paths if item[1]]
        # [1] project name
        file_name = sanitize_for_filename( Path(state.prefs.project_dir).name[:20] ) + " "
        # [2] file number
        if len(ranges) > 1:
            file_name += f"[{ chapter_index+1 } of {len(ranges)}]" + " "
        # [3] line range
        if len(ranges) > 1:
            file_name += f"[{chapter_index_start+1}-{chapter_index_end+1}]" + " "
        # [4] num lines missing within that range
        if num_missing > 0:
            file_name += f"[{num_missing} missing]" + " "
        # [5] model tag
        common_model_tag = SoundSegmentUtil.get_common_model_tag(extant_paths)
        if common_model_tag:
            file_name += f"[{common_model_tag}]" + " "
        # [5] voice tag
        common_voice_tag = SoundSegmentUtil.get_common_voice_tag(extant_paths)
        if common_voice_tag:
            file_name += f"[{common_voice_tag}]" + " "
        file_name = file_name.strip() + ".abr" + (".m4a" if to_aac_not_flac else ".flac")
        dest_path = os.path.join(base_dir, file_name)

        # Concat
        durations = ConcatUtil.concatenate_files_plus_silence(
            dest_path,
            phrases_and_paths,
            print_progress=True,
            use_section_sound_effect=state.project.use_section_sound_effect,
            to_aac_not_flac=to_aac_not_flac
        )
        if isinstance(durations, str):
            return "", durations

        # Add the app metadata
        phrases = [item[0] for item in phrases_and_paths]
        sound_paths = [item[1] for item in phrases_and_paths]
        timed_phrases = TimedPhrase.make_list_using(phrases, durations)
        if state.project.subdivide_phrases:
            timed_phrases = make_granular_timed_phrases(timed_phrases, sound_paths, durations)
        
        meta = AppMetadata(
            raw_text=raw_text, 
            timed_phrases=timed_phrases,
            has_section_break_audio=state.project.use_section_sound_effect
        )

        if to_aac_not_flac:
            err = AppMetadata.save_to_mp4(meta, dest_path)
        else:
            err = AppMetadata.save_to_flac(meta, dest_path)
        if err:
            return "", err

        return dest_path, ""


    @staticmethod
    def concatenate_files_plus_silence(
        dest_path: str,
        phrases_and_paths: list[ tuple[Phrase, str] ],
        use_section_sound_effect: bool,
        print_progress: bool,
        to_aac_not_flac: bool
    ) -> list[float] | str:
        """
        Concatenates a list of files to a destination file using ffmpeg streaming process.
        Dynamically adds silence between adjacent segments based on phrase "reason".

        Returns list of float durations of each added segment to be used for app metadata .

        On error, returns error string
        """

        total_duration = 0
        durations = []

        process = ConcatUtil.init_ffmpeg_stream(dest_path, to_aac_not_flac)

        SigIntHandler().set("concat")

        for i, (phrase, sound_path) in enumerate(phrases_and_paths):

            if not sound_path:
                durations.append(0)
                continue

            if SigIntHandler().did_interrupt:
                SigIntHandler().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            if i < len(phrases_and_paths) - 1:
                if phrase.reason == Reason.SECTION and use_section_sound_effect:
                    appended_silence_duration = 0
                    appended_sound_effect_path = SECTION_SOUND_EFFECT_PATH
                else:
                    appended_silence_duration = phrase.reason.pause_duration
                    appended_sound_effect_path = ""
            else: # Last phrase
                appended_silence_duration = 0
                appended_sound_effect_path = ""

            if print_progress:
                s = f"{time_stamp(total_duration, with_tenth=False)} {Path(sound_path).stem[:80]} ... "
                print("\x1b[1G" + s, end="\033[K", flush=True)

            result = SoundFileUtil.load(sound_path)
            if isinstance(result, str): # error
                ConcatUtil.close_ffmpeg_stream(process) # TODO clean up more and message user
                return result
            else:
                sound = result

            sound = SoundUtil.resample_if_necessary(sound, APP_SAMPLE_RATE)

            # Append sound effect or silence or not
            if appended_sound_effect_path:
                sound = SoundUtil.append_sound_using_path(sound, appended_sound_effect_path)
            elif appended_silence_duration:
                sound = SoundUtil.add_silence(sound, appended_silence_duration)

            durations.append(sound.duration)

            total_duration += sound.duration

            ConcatUtil.add_audio_to_ffmpeg_stream(process, sound.data)

        printt()
        printt()
        ConcatUtil.close_ffmpeg_stream(process)

        return durations

    @staticmethod
    def init_ffmpeg_stream(dest_path: str, is_aac_not_flac: bool) -> subprocess.Popen:
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
            args.extend(FFMPEG_ARGUMENTS_OUTPUT_AAC)
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

        process.stdin.write(pcm_data.tobytes()) # type: ignore

    @staticmethod
    def close_ffmpeg_stream(process: subprocess.Popen):
        """
        Closes the stdin of the ffmpeg process and waits for it to terminate.
        """
        process.stdin.close()  # type: ignore
        process.wait()

# ---

def make_granular_timed_phrases(
        timed_phrases: list[TimedPhrase], 
        sound_paths: list[str],
        sound_durations: list[float]
    ) -> list[TimedPhrase]:
    """
    Uses the "forced alignment" metadata in the json files saved alongside the sound_paths 
    to break up the timed_phrases into smaller parts.

    The argments are parallel lists.
    """

    assert(len(timed_phrases) == len(sound_paths) == len(sound_durations)) # xxx

    results: list[TimedPhrase] = []

    for i in range(0, len(timed_phrases)):
        
        print()

        original_timed_phrase = timed_phrases[i]
        sound_path = sound_paths[i]

        if not sound_path:
            results.append(original_timed_phrase)
            continue

        subdivided_items_json_path = Path(sound_path).with_suffix(".json") # TODO rename this to .json
        if not subdivided_items_json_path.exists():
            results.append(original_timed_phrase)
            continue

        try:
            with open(subdivided_items_json_path, 'r', encoding='utf-8') as file:
                json_dicts = json.load(file)
        except Exception as e:
            # File error; use original item
            results.append(original_timed_phrase)
            continue

        parse_result = TimedPhrase.dicts_to_timed_phrases(json_dicts)
        if isinstance(parse_result, str): 
            # Parse error; use original item
            results.append(original_timed_phrase)
            continue
        subdivided_timed_phrases = parse_result

        offset = original_timed_phrase.time_start

        for subdivided_index, item in enumerate(subdivided_timed_phrases):
            
            if subdivided_index == 0:
                time_start = offset
            else:
                time_start = offset + subdivided_timed_phrases[subdivided_index - 1].time_end
            time_end = offset + item.time_end

            updated_item = TimedPhrase(item.text, time_start, time_end)
            # print("xxx new timed phrase", updated_item)
            results.append(updated_item)
        
        # Set last item's time_end using the duration of the source audio clip
        # rather than the transcription end word timestamp
        # (prevents discontinuities in segment selectedness across boundaries, which looks distracting)
        results[-1].time_end = sound_durations[i] + offset

    return results
