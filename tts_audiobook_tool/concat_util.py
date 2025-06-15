import os
from typing import cast
from pathlib import Path

from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.chapter_info import ChapterInfo
from tts_audiobook_tool.app_meta_util import AppMetaUtil
from tts_audiobook_tool.l import L
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.parse_util import ParseUtil
from tts_audiobook_tool.project_dir_util import ProjectDirUtil
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment, TextSegmentReason
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
from tts_audiobook_tool.transcode_util import TranscodeUtil
from tts_audiobook_tool.util import *

class ConcatUtil:

    @staticmethod
    def concatenate_chapters(
        chapter_indices: list[int],
        state: State,
        and_transcode: bool,
        base_dir: str
    ) -> bool:
        """
        Returns False on (any) fail
        """

        # TODO: make interruptable after concat and after compress

        raw_text = state.project.load_raw_text()
        if not raw_text:
            printt("Error loading text", "error")
            return False

        ranges = make_section_ranges(state.project.section_dividers, len(state.project.text_segments))
        segment_index_to_path = ProjectDirUtil.get_indices_and_paths(state)
        counter = 1

        for chapter_index in chapter_indices:

            # Make tuples list
            # This list is the full length of the project.text_segments,
            # but uses empty strings for file paths outside the range of the chapter

            chapter_index_start, chapter_index_end = ranges[chapter_index]
            num_missing = 0

            # Make text segment + file path list
            segments_and_paths: list[ tuple[TextSegment, str]] = []
            for segment_index, segment in enumerate(state.project.text_segments):
                if segment_index < chapter_index_start or segment_index > chapter_index_end:
                    # Out of range
                    segments_and_paths.append((segment, ""))
                    continue
                if not segment_index in segment_index_to_path:
                    # Audio segment file not yet generated
                    segments_and_paths.append((segment, ""))
                    num_missing += 1
                    continue
                file_path = segment_index_to_path[segment_index]
                segments_and_paths.append((segment, file_path))

            # Trim silence pass
            # Note how this is being done at 'concatenation time' ATM, not 'generation time'
            if state.prefs.optimize_segment_silence:
                printt("Trimming sentence continuation pauses...")
                printt()
                ConcatUtil.trim_sentence_continuation_pauses(
                    segments_and_paths, SENTENCE_CONTINUATION_MAX_DURATION
                )
                printt()

            # Append minimum duration of silence at paragraph breaks
            # Note how this is being done at 'concatenation time' ATM, not 'generation time'
            if state.prefs.optimize_segment_silence: # TODO rename pref or add independent pref value
                printt("Enforcing minimum silence duration at paragraph breaks...")
                printt()
                ConcatUtil.add_silence_at_paragraph_breaks(
                    segments_and_paths, PARAGRAPH_SILENCE_MIN_DURATION
                )
                printt()

            # Make final concatenated audio file
            printt(f"Creating finalized, concatenated audio file ({counter}/{len(chapter_indices)})\n")

            # project name
            file_name = sanitize_for_filename( Path(state.prefs.project_dir).name[:20] ) + " "
            # file number
            file_name += f"[{ chapter_index+1 } of {len(ranges)}]" + " "
            # line range
            file_name += f"[{chapter_index_start+1}-{chapter_index_end+1}]" + " "
            # num segments missing
            if num_missing > 0:
                file_name += f"[{num_missing} missing]" + " "
            # voice label
            extant_paths = [item[1] for item in segments_and_paths if item[1]]
            common_voice_label = ProjectDirUtil.get_common_voice_label(extant_paths)
            if common_voice_label:
                file_name += f"[{state.project.get_voice_label()}]"
            file_name += ".abr.flac"

            dest_file_path = os.path.join(base_dir, file_name)

            error = ConcatUtil.make_app_flac(raw_text, segments_and_paths, dest_file_path)
            if error:
                printt(error, "error")
                return False

            col = COL_DEFAULT if and_transcode else COL_ACCENT
            printt(f"Saved: {col}{dest_file_path}")
            printt()

            if and_transcode:
                printt("Transcoding to MP4...")
                printt()
                transcode_path, err = TranscodeUtil.transcode_abr_flac_to_aac(dest_file_path)
                printt()
                if err:
                    printt(err, "error")
                    return False
                printt(f"Saved: {COL_ACCENT}{transcode_path}")
                printt()

            counter += 1

        return True

    @staticmethod
    def trim_sentence_continuation_pauses(
        segments_and_paths: list[ tuple[TextSegment, str] ],
        max_duration: float=0.5
    ):
        """
        Trims excessive silence between adjacent segments when segment is a sentence continuation.
        """

        for i in range( 1, len(segments_and_paths) ):

            segment_a, path_a = segments_and_paths[i - 1]
            segment_b, path_b = segments_and_paths[i]

            if segment_b.reason != TextSegmentReason.INSIDE_SENTENCE:
                continue
            if not path_b or not path_a:
                continue


            result = SilenceUtil.trim_silence_if_necessary(path_a, path_b, max_duration)
            if isinstance(result, str):
                printt(f"{COL_ERROR}{result}")

    @staticmethod
    def add_silence_at_paragraph_breaks(
        segments_and_paths: list[ tuple[TextSegment, str] ],
        min_duration: float=0.5
    ):
        """
        Enforces a minimum duration of silence at paragraph breaks
        """
        for i in range( 1, len(segments_and_paths) ):

            segment_a, path_a = segments_and_paths[i - 1]
            segment_b, path_b = segments_and_paths[i]

            if not path_a or not path_b:
                continue

            if segment_b.reason == TextSegmentReason.PARAGRAPH:

                result = SilenceUtil.add_silence_if_necessary(path_a, path_b, min_duration)
                if isinstance(result, str):
                    printt(f"{COL_ERROR}{result}")

    @staticmethod
    def make_app_flac(
        raw_text: str,
        segments_and_paths: list[ tuple[TextSegment, str] ],
        dest_file_path: str
    ) -> str:
        """
        Makes ".abr.flac" file (concatenated audio segments plus custom app metadata)
        Returns error message or empty string.

        :param segments_and_paths:
            List of tuples of TextSegment and corresponding audio file path
            Should contain all the text segments of the project
            File paths which are empty will be skipped.
        """

        # Make app metadata ("timed text segments")
        timed_text_segments = []
        seconds = 0.0

        for text_segment, audio_file_path in segments_and_paths:

            if not audio_file_path:
                timed_text_segment = TimedTextSegment.make_using(text_segment, 0, 0) # has no start/end times
                timed_text_segments.append(timed_text_segment)
                continue

            duration = AudioMetaUtil.get_audio_duration(audio_file_path)
            if duration is None:
                return f"Couldn't get duration for {audio_file_path}"
            timed_text_segment = TimedTextSegment.make_using(text_segment, seconds, seconds + duration)
            timed_text_segments.append(timed_text_segment)
            seconds += duration

        file_paths = [path for _, path in segments_and_paths if path]

        # Make concatenated file
        error = SoundFileUtil.concatenate_flacs(file_paths, dest_file_path)
        if error:
            return error

        # Add the app metadata
        error = AppMetaUtil.set_flac_app_metadata(
            flac_path=dest_file_path,
            raw_text=raw_text,
            timed_text_segments=timed_text_segments
        )
        if error:
            return error

        return "" # success


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

