import os
from pathlib import Path

import numpy as np
from numpy import ndarray

from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.audio_meta_util import AudioMetaUtil
from tts_audiobook_tool.app_metadata import AppMetadata
from tts_audiobook_tool.l import L # type: ignore
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.text_segment import TextSegment, TextSegmentReason
from tts_audiobook_tool.timed_text_segment import TimedTextSegment
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

        ranges = make_section_ranges(state.project.section_dividers, len(state.project.text_segments))
        sound_segments = state.project.sound_segments.sound_segments

        chapter_index_start, chapter_index_end = ranges[chapter_index]
        num_missing = 0

        # Make text segment + file path list
        segments_and_paths: list[ tuple[TextSegment, str]] = []
        for segment_index, segment in enumerate(state.project.text_segments):
            if segment_index < chapter_index_start or segment_index > chapter_index_end:
                # Out of range
                segments_and_paths.append((segment, ""))
                continue
            if not segment_index in sound_segments:
                # Audio segment file not yet generated
                segments_and_paths.append((segment, ""))
                num_missing += 1
                continue
            file_path = sound_segments[segment_index]
            segments_and_paths.append((segment, file_path))

        # Filename
        extant_paths = [item[1] for item in segments_and_paths if item[1]]
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
            segments_and_paths,
            print_progress=True,
            section_sound_effect=state.prefs.section_sound_effect,
            to_aac_not_flac=to_aac_not_flac
        )
        if isinstance(durations, str):
            return "", durations

        # Add the app metadata
        text_segments = [item[0] for item in segments_and_paths]
        timed_text_segments = TimedTextSegment.make_list_using(text_segments, durations)
        meta = AppMetadata(raw_text=raw_text, timed_text_segments=timed_text_segments)

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
        segments_and_paths: list[ tuple[TextSegment, str] ],
        section_sound_effect: bool,
        print_progress: bool,
        to_aac_not_flac: bool
    ) -> list[float] | str:
        """
        Concatenates a list of files to a destination file using ffmpeg streaming process.
        Dynamically adds silence between adjacent segments based on text segment "reason".

        Returns timestamps of added segments to be used for app metadata
        On error, returns error string
        """

        total_duration = 0
        durations = []

        process = ConcatUtil.init_ffmpeg_stream(dest_path, to_aac_not_flac)

        # Add dummy item to ensure real last item is processed in the loop
        segments_and_paths = segments_and_paths.copy()
        segments_and_paths.append((TextSegment("", -1, -1, TextSegmentReason.UNDEFINED), ""))

        SigIntHandler().set("concat")

        for i in range(0, len(segments_and_paths) - 1):

            if SigIntHandler().did_interrupt:
                SigIntHandler().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            _, path_a = segments_and_paths[i]
            segment_b, path_b = segments_and_paths[i + 1]

            if not path_a:
                durations.append(0)
                continue

            if not path_b:
                # Gap in the sequence, use some default silence padding value
                appended_silence_duration = TextSegmentReason.UNDEFINED.pause_duration
            else:
                appended_silence_duration = segment_b.reason.pause_duration

            if segment_b.reason == TextSegmentReason.SECTION and section_sound_effect:
                appended_sound_effect_path = SECTION_SOUND_EFFECT_PATH
                appended_silence_duration = 0
            else:
                appended_sound_effect_path = ""

            is_last = i == len(segments_and_paths) - 2
            if is_last:
                appended_silence_duration = 0

            if print_progress:
                s = f"{time_stamp(total_duration, with_tenth=False)} {Path(path_a).stem[:80]} ..."
                print("\x1b[1G" + s, end="\033[K", flush=True)

            sound_a = SoundFileUtil.load(path_a)
            if isinstance(sound_a, str): # error
                ConcatUtil.close_ffmpeg_stream(process) # TODO clean up more and message user
                return sound_a

            sound_a = SoundUtil.resample_if_necessary(sound_a, APP_SAMPLE_RATE)

            # Append sound effect or silence or not
            if appended_sound_effect_path:
                sound_a = SoundUtil.append_sound_using_path(sound_a, appended_sound_effect_path)
            elif appended_silence_duration:
                sound_a = SoundUtil.add_silence(sound_a, appended_silence_duration)

            durations.append(sound_a.duration)

            total_duration += sound_a.duration

            ConcatUtil.add_audio_to_ffmpeg_stream(process, sound_a.data)

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


    @staticmethod
    def make_app_flac_using_files(
        raw_text: str,
        segments_and_paths: list[ tuple[TextSegment, str] ],
        dest_file_path: str
    ) -> str:

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
        err = SoundFileUtil.concatenate_flacs(file_paths, dest_file_path)
        if err:
            return err

        # Add the app metadata
        meta = AppMetadata(raw_text=raw_text, timed_text_segments=timed_text_segments)
        err = AppMetadata.save_to_flac(meta, dest_file_path)
        if err:
            return err

        return "" # success
