from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from numpy import ndarray

from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import app_hint_util
from tts_audiobook_tool.app_support import app_paths
from tts_audiobook_tool.app_support import dev_hooks_loader
from tts_audiobook_tool.system_support.browser import get_chromium_info, launch_player_with_chromium
from tts_audiobook_tool.app_types import SectionMarkerMode, ExportType, HighShelfEq, NormalizationSpecs, RoomReverb, RoomReverbStereo, Sound
from tts_audiobook_tool import ask
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.project_support.voice_spatial_policy import VoiceSpatialPolicy
from tts_audiobook_tool.project_support.project_serialization_util import ProjectSerializationUtil
from tts_audiobook_tool.project_support.project_text_io_util import ProjectTextIOUtil
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.sound.loudness_leveling import LevelingStats, LoudnessLeveling
from tts_audiobook_tool.sound import m4b_chapter_util
from tts_audiobook_tool.l import L
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.reason_pauses import ReasonPauses
from tts_audiobook_tool.sound.noise_bed import NoiseBed
from tts_audiobook_tool.sound.voice_spatial import VoiceSpatialProcessor
from tts_audiobook_tool.sound.sentence_gap_util import SentenceGapInsertion
from tts_audiobook_tool.sound.silence_util import GapTrimMap, SilenceUtil
from tts_audiobook_tool.sound.sound_pipeline import BreakEffectTracker, SoundPipeline
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil, get_segment_stt_info_path
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.app_support.system_sleep import prevent_system_sleep
from tts_audiobook_tool.app_types.app_metadata import (
    AppMetadata,
    AppMetadataSection,
    AppMetadataTextSegment,
    count_app_metadata_text_segment_leaves,
)
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION
from tts_audiobook_tool.state import State
from tts_audiobook_tool.app_types.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.app_types.output_range_info import OutputRangeInfo
from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.util import *


def _validate_sentence_gap_timings(
    timed_phrases: list[TimedPhrase], transcript_words: list[dict],
) -> None:
    """Validate sidecar fields consumed by gap extension, without rewriting timings."""
    def check_time(value: object) -> None:
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value)):
            raise ValueError("Sentence gap timings must be finite numbers")

    for phrase in timed_phrases:
        if not isinstance(phrase.text, str):
            raise ValueError("Invalid timed phrase text")
        if phrase.reason is not None and not isinstance(phrase.reason, Reason):
            raise ValueError("Invalid timed phrase reason")
        check_time(phrase.time_start)
        check_time(phrase.time_end)

    for word in transcript_words:
        if not isinstance(word, dict):
            raise ValueError("Invalid transcript word")
        if not isinstance(word["word"], str):
            raise ValueError("Invalid transcript word text")
        check_time(word["start"])
        check_time(word["end"])


@dataclass
class ConcatRenderResult:
    """
    What ConcatUtil.concatenate_sound_segments() produces.

    durations is aligned with phrases_and_paths (0.0 for groups without
    generated audio) and includes each segment's trailing pause or break effect.

    gap_trim_maps is aligned the same way and holds, for every segment whose
    internal silences were edited (sentence gaps extended and/or over-long gaps
    shortened), the map from the saved segment to the edited one (None where
    nothing was edited). STT word timings are recorded against the unedited
    saved audio, so subdivision has to push them through this map.
    """

    durations: list[float]
    gap_trim_maps: list[GapTrimMap | None] = field(default_factory=list)

    @property
    def gap_trim_count(self) -> int:
        return sum(len(gap_map.trims) for gap_map in self.gap_trim_maps if gap_map)

    @property
    def gap_removed_duration(self) -> float:
        return sum(
            sum(trim.removed_duration for trim in gap_map.trims)
            for gap_map in self.gap_trim_maps
            if gap_map
        )


class ConcatUtil:

    @staticmethod
    def unload_models_for_upsampling() -> str:
        if not ModelWorker.is_alive():
            return ""

        printt(f"{COL_DIM_ITALICS}Attempting to unload models to free up VRAM for generative upsampling...{COL_DEFAULT}")
        printt()
        error = ModelWorker.clear_models_if_running_blocking()
        if error:
            return f"Couldn't unload models before generative upsampling: {error}"
        return ""

    # Pseudo-silence compensation for inter-segment pauses during concatenation.
    #
    # Segments are already trimmed of true silence, but may still contain
    # low-volume but natural-sounding non-verbal audio at their ends/starts
    # (e.g. trailing breaths). When enabled, the silence inserted between two
    # adjacent segments is reduced by the combined duration of this "pseudo-
    # silence" present at the end of the previous segment and the start of the
    # next. If the combined pseudo-silence meets or exceeds the hardcoded pause
    # duration, no silence is inserted. Break sound effects are compensated the
    # same way: only the silence padding around the effect shrinks (the effect
    # itself is never trimmed), keeping it centered in the whole gap.
    PSEUDO_SILENCE_COMPENSATION_ENABLED = True
    PSEUDO_SILENCE_MAX_SECONDS = 1.0
    PSEUDO_SILENCE_THRESHOLD_DB = -20.0 # very permissive

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

        app_support.log_unload_memory_snapshot("right before concatenation")

        if file_cut_indices and bookmark_indices:
            raise ValueError(f"file_cut_indices and bookmark_indices are mutually exclusive: {file_cut_indices} vs {bookmark_indices}")

        start_time = time.time()

        # Preflight checks
        lava_sr_available, lava_probe_error = ModelWorker.probe_lava_sr_blocking()
        if lava_probe_error:
            lava_sr_available = False
        use_upsampler = state.project.use_upsampler and lava_sr_available
        if state.project.use_upsampler and not lava_sr_available:
            printt(
                f"{COL_DIM_ITALICS}Warning: LavaSR v2 upsampling is enabled, "
                "but LavaSR is not installed. Continuing without upsampling."
            )
            printt()
        if use_upsampler:
            unload_error = ConcatUtil.unload_models_for_upsampling()
            if unload_error:
                ask.ask_error(unload_error)
                return

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

        last_dest_path = ""
        dest_paths: list[str] = []
        for i, file_cut_index in enumerate(file_cut_indices):

            message = "Creating concatenated audiobook file"
            if len(file_cut_indices) > 1:
                message += f" {i+1} of {len(file_cut_indices)} - output file {file_cut_index+1}"
            message += "..."
            print_accent_separator(message)

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
                stem_path=dest_stem_path,
                use_upsampler=use_upsampler,
            )
            if err:
                printt()
                ask.ask_error(err)
                return
            last_dest_path = dest_path
            dest_paths.append(dest_path)

            printt(f"{COL_ACCENT}Saved {COL_DEFAULT}{make_terminal_hyperlink(dest_path)}")
            printt()

        # Post-concat feedback, prompt

        app_support.play_done_sound()
        elapsed = duration_string(time.time() - start_time)
        printt(f"Finished{COL_DIM} (elapsed: {elapsed})")
        printt()

        dev_hooks_loader.call(
            "on_concat_done",
            state=state, dest_dir=dest_dir, dest_paths=dest_paths, last_dest_path=last_dest_path
        )

        app_hint_util.show_player_hint(state.prefs)

        chromium_info = get_chromium_info()
        if chromium_info:
            prompt = (
                f"Press {make_hotkey_string('Enter')} to continue, "
                f"{make_hotkey_string('O')} to open output dir, or "
                f"{make_hotkey_string('P')} to play file in the browser app: "
            )
        else:
            prompt = (
                f"Press {make_hotkey_string('Enter')}, or press {make_hotkey_string('O')} "
                "to open output directory in system file explorer: "
            )
        hotkey = ask.ask_hotkey(prompt)
        printt()
        if hotkey == "o":
            err = open_directory_in_gui(dest_dir)
            if err:
                ask.ask_error(err)
        elif hotkey == "p" and chromium_info is not None:
            launch_player_with_chromium(
                chromium_path=chromium_info[1],
                audio_file_path=last_dest_path,
                user_data_dir=app_paths.get_chromium_user_data_dir(),
            )

    @staticmethod
    @prevent_system_sleep()
    def make_file(
        state: State,
        index_start: int,
        index_end: int,
        bookmark_indices: list[int],
        stem_path: str,
        use_upsampler: bool | None = None,
    ) -> tuple[str, str]:
        """
        Creates final output file, full feature flow.

        Params:
            index_end: Inclusive
            stem_path: Destination file path w/o file suffix
            use_upsampler: Whether to upsample concatenated segments. None uses
                the project's setting; an explicit bool overrides it (for
                example, after make_files resolves upsampler availability).

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

        # Loudness is leveled per segment during concatenation, so the
        # concatenated file is already in the final codec
        suffix = ".m4b" if is_aac else ".flac"
        concat_path = stem_path + " [concat]" + suffix

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
            paths = [ concat_path, chapter_meta_path]
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
        section_start_indices = ProjectBookUtil.get_section_start_indices(state.project)
        spatial_policy = VoiceSpatialPolicy.from_project(state.project)
        voice_pans = [
            spatial_policy.pan_for_voice_index(group.voice_index)
            for group in state.project.phrase_groups
        ]
        voice_reverbs = [
            spatial_policy.room_reverb_for_voice_index(group.voice_index)
            for group in state.project.phrase_groups
        ]
        voice_volume_dbs = [
            spatial_policy.volume_db_for_voice_index(group.voice_index)
            for group in state.project.phrase_groups
        ]

        # [1] Concatenated audio file

        use_upsampler = (
            state.project.use_upsampler
            if use_upsampler is None
            else use_upsampler
        )
        upsampler_unload_error = ""
        try:
            result = ConcatUtil.concatenate_sound_segments(
                concat_path,
                phrases_and_paths,
                print_progress=True,
                use_break_sound_effect=state.project.use_break_sound_effect,
                high_shelf=high_shelf,
                reason_pauses=state.project.reason_pauses,
                section_start_indices=section_start_indices,
                start_group_index=index_start,
                aac_bitrate=state.prefs.aac_bitrate,
                use_upsampler=use_upsampler,
                limit_silence_gaps=state.project.limit_silence_gaps,
                limit_silence_gaps_duration=state.project.limit_silence_gaps_duration,
                min_sentence_silence=state.project.min_segment_sentence_silence.duration,
                room_tone_db=state.project.room_tone.level_db,
                room_reverb_stereo=state.project.room_reverb_stereo,
                voice_pans=voice_pans,
                voice_reverbs=voice_reverbs,
                voice_volume_dbs=voice_volume_dbs,
                stereo_required=spatial_policy.requested_channels == 2,
                loudness=state.project.normalization_type.value,
            )
        finally:
            # LavaSR lives in the model worker. Always explicitly unload it at
            # the end of the upsampling phase—even on interruption or failure—
            # before any later output-processing step begins.
            if use_upsampler:
                app_support.log_unload_memory_snapshot("upscale operation finished")
                upsampler_unload_error = ModelWorker.clear_models_if_running_blocking()
                app_support.log_unload_memory_snapshot("clear_lava_sr_upsampler finished")

        if upsampler_unload_error:
            delete_intermediate_files()
            return "", f"Couldn't unload LavaSR after generative upsampling: {upsampler_unload_error}"
        if isinstance(result, str): # is error
            delete_intermediate_files()
            return "", result
        else:
            durations = result.durations
            gap_trim_maps = result.gap_trim_maps
            last_path = concat_path

        # [2] Chapter (M4B) metadata
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

        # [3] App metadata (final file)

        phrases = [item[0] for item in phrases_and_paths]
        base_timed_phrases = TimedPhrase.make_list_using(phrases, durations)
        timed_phrases: list[AppMetadataTextSegment] = list(base_timed_phrases)
        phrase_to_text_segment_start_indices = list(range(len(timed_phrases)))
        if state.project.subdivide_phrases:
            file_paths = [item[1] for item in phrases_and_paths]
            timed_phrases, bookmark_indices, phrase_to_text_segment_start_indices = make_subdivided_timed_phrases(
                timed_phrases=base_timed_phrases,
                phrase_groups=state.project.phrase_groups,
                sound_paths=file_paths,
                sound_durations=durations,
                bookmark_indices=bookmark_indices,
                gap_trim_maps=gap_trim_maps,
            )
        sections = make_app_metadata_sections(
            project=state.project,
            index_start=index_start,
            index_end=index_end,
            phrase_to_text_segment_start_indices=phrase_to_text_segment_start_indices,
            text_segment_count=count_app_metadata_text_segment_leaves(timed_phrases),
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
            type="generated",
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
        index_start: int,
        index_end: int
    ) -> int:
        num_missing = 0
        for group_index in range(len(project.phrase_groups)):
            out_of_range = (group_index < index_start or group_index > index_end)
            if out_of_range:
                continue
            file_name = project.sound_segments.get_best_file_for(group_index)
            if not file_name:
                num_missing += 1
        return num_missing

    @staticmethod
    def concatenate_sound_segments(
        dest_path: str,
        phrases_and_paths: list[ tuple[Phrase, str] ],
        use_break_sound_effect: bool,
        high_shelf: HighShelfEq,
        reason_pauses: ReasonPauses,
        print_progress: bool,
        section_start_indices: list[int] | None = None,
        start_group_index: int = 0,
        aac_bitrate: str=AAC_BITRATE_DEFAULT,
        use_upsampler: bool = False,
        limit_silence_gaps: bool = False,
        limit_silence_gaps_duration: float = PROJECT_DEFAULT_LIMIT_SILENCE_GAPS_DURATION,
        min_sentence_silence: float = 0.0,
        room_tone_db: float | None = None,
        room_reverb: RoomReverb = RoomReverb.OFF,
        room_reverb_stereo: RoomReverbStereo = RoomReverbStereo.MONO,
        voice_pans: list[float] | None = None,
        voice_reverbs: list[RoomReverb] | None = None,
        voice_volume_dbs: list[float] | None = None,
        stereo_required: bool = False,
        loudness: NormalizationSpecs | None = None,
    ) -> ConcatRenderResult | str:
        """
        Concatenates a list of files to a destination file using ffmpeg streaming process.
        Adds silence or sound effect between adjacent segments based on phrase "reason".

        :param section_start_indices:
            Group indices where sections begin (eg from
            `ProjectBookUtil.get_section_start_indices`). Feeds the
            `BreakEffectTracker` rules for break sound effects.
        :param start_group_index:
            The group index the output begins at, used to seed the tracker.
        :param aac_bitrate:
            Only relevant if dest_path suffix is .m4a/.m4b; ignored otherwise.
            Must be a valid AAC bitrate string like "128k".
        :param limit_silence_gaps:
            Shorten silence gaps inside each rendered segment. Done here rather
            than at generation time so re-concatenating saved audio with a
            different threshold doesn't need a regeneration pass.
        :param limit_silence_gaps_duration:
            Maximum duration of an internal silence gap, in seconds.
        :param min_sentence_silence:
            Minimum duration, in seconds, of the silence at sentence boundaries
            inside each segment (0 disables). Short gaps are extended using the
            segment sidecar's timings; segments without a usable sidecar are
            left alone. Applied before limit_silence_gaps, and capped at
            limit_silence_gaps_duration when that is enabled. Gaps are extended
            with digital silence.
        :param room_tone_db:
            Level (dBFS RMS) of the constant noise bed run under the whole
            output, including pauses and break effects (see NoiseBed). None
            disables it.
        :param room_reverb:
            Room reverb preset for every entry, when `voice_reverbs` is None.
        :param room_reverb_stereo:
            Stereo spread of the room reverb. When any entry's reverb is on and
            this isn't MONO, the output is stereo, independently of voice
            positioning.
        :param voice_pans:
            Per-entry direct narration pan, aligned with phrases_and_paths,
            including entries with missing files. None keeps every voice centered.
        :param voice_reverbs:
            Per-entry room reverb preset (the speaking voice's), aligned like
            `voice_pans`. Streamed over the whole output before the noise bed,
            one room per preset (see VoiceSpatialProcessor): only an entry's
            narration feeds its room; pauses and break effects stay dry, and
            every room's tail keeps decaying under whatever follows.
        :param voice_volume_dbs:
            Per-entry voice volume as a gain in dB (<= 0), aligned like
            `voice_pans`. Applied on top of loudness leveling, so only when
            `loudness` is given. None keeps every voice at full volume.
        :param stereo_required:
            Project-level stereo requirement, even if this range has no panned
            speech. Nonzero voice_pans also require stereo for low-level callers.
        :param loudness:
            Loudness target. Each segment is leveled to it (one gain, then a
            peak limiter) before panning, and the room tone and break effects
            are shifted to keep their level relative to the speech; a final
            safety limiter runs on the mixed output (see LoudnessLeveling).
            None leaves levels untouched.

        Returns a ConcatRenderResult: the duration of each added segment (for app
        metadata) plus the gap-cut maps needed to remap STT word timings.

        On error, returns error string
        """

        min_sentence_silence = SoundPipeline.effective_min_sentence_silence(
            min_sentence_silence, limit_silence_gaps, limit_silence_gaps_duration
        )

        duration_sum = 0.0
        # Pre-allocate so durations stays aligned with phrases_and_paths indices
        # even though each segment is flushed one iteration late (the look-ahead
        # buffer holds a segment until the next one is rendered).
        durations: list[float] = [0.0] * len(phrases_and_paths)
        gap_trim_maps: list[GapTrimMap | None] = [None] * len(phrases_and_paths)

        if voice_pans is None:
            voice_pans = [0.0] * len(phrases_and_paths)
        elif len(voice_pans) != len(phrases_and_paths):
            raise ValueError("voice_pans must align with phrases_and_paths")
        if voice_reverbs is None:
            voice_reverbs = [room_reverb] * len(phrases_and_paths)
        elif len(voice_reverbs) != len(phrases_and_paths):
            raise ValueError("voice_reverbs must align with phrases_and_paths")
        if voice_volume_dbs is None:
            voice_volume_dbs = [0.0] * len(phrases_and_paths)
        elif len(voice_volume_dbs) != len(phrases_and_paths):
            raise ValueError("voice_volume_dbs must align with phrases_and_paths")
        has_reverb = any(reverb.design is not None and reverb.drr_db is not None for reverb in voice_reverbs)
        channels = 2 if (stereo_required or any(pan != 0 for pan in voice_pans)
                         or (has_reverb and room_reverb_stereo.is_stereo)) else 1

        # Levels authored against un-leveled speech (room tone, break effects)
        # move with the loudness target
        level_offset_db = LoudnessLeveling.reference_offset_db(loudness, channels) if loudness is not None else 0.0
        if room_tone_db is not None:
            room_tone_db += level_offset_db
        leveling_stats = LevelingStats()

        to_aac_not_flac = dest_path.lower().endswith(tuple(AAC_SUFFIXES))
        process = ConcatUtil.init_ffmpeg_stream(dest_path, to_aac_not_flac, aac_bitrate, channels=channels)

        Interrupts().set("concat")

        # Render-time break effect rules. The tracker is fed one entry per
        # flushed (present) segment, in emission order.
        tracker = BreakEffectTracker(
            section_start_indices if section_start_indices is not None else [0],
            start_group_index=start_group_index,
        )

        # Look-ahead buffer. The previous present segment is held (rendered, but
        # without its trailing pause) until the next present segment is rendered,
        # so the inserted pause can be adjusted using the pseudo-silence measured
        # at the end of the held segment and the start of the next one.
        # pending = (sound_no_pause, phrase, index, path)
        pending: tuple[Sound, Phrase, int, str] | None = None

        # Room tone setting: a constant noise bed under the whole output.
        # Created on first use, at the rendered audio's sample rate.
        noise_bed: NoiseBed | None = None
        # One shared spatial processor; its room tails span voice changes.
        spatial = VoiceSpatialProcessor(channels=channels, room_reverb_stereo=room_reverb_stereo)

        def flush_pending(next_sound: Sound | None) -> None:
            """
            Appends the (possibly compensated) trailing pause to the held
            segment and streams it to ffmpeg. `next_sound` is the following
            segment's rendered sound (without pause), or None for the final
            segment (no compensation applied).
            """
            nonlocal duration_sum, noise_bed
            assert pending is not None
            sound, phrase, idx, path = pending
            narration_end = len(sound.data)
            is_final_segment = next_sound is None

            break_effect = tracker.next_break_effect(
                idx,
                phrase.reason,
                effects_enabled=use_break_sound_effect,
                is_final=is_final_segment,
            )

            # Pseudo-silence at the boundary counts toward the pause or break
            # effect gap (see PSEUDO_SILENCE_COMPENSATION_ENABLED)
            end_pseudo = start_pseudo = 0.0
            if ConcatUtil.PSEUDO_SILENCE_COMPENSATION_ENABLED and next_sound is not None:
                # FYI, start/end silence measurements not idempotent wrt upsampling vs not
                end_silence_start = SilenceUtil.get_end_silence(
                    sound,
                    max_seconds=ConcatUtil.PSEUDO_SILENCE_MAX_SECONDS,
                    threshold_db_relative_to_peak=ConcatUtil.PSEUDO_SILENCE_THRESHOLD_DB,
                )
                end_pseudo = (sound.duration - end_silence_start) if end_silence_start is not None else 0.0
                end_pseudo = max(0.0, min(end_pseudo, sound.duration))
                start_pseudo = SilenceUtil.get_start_silence(
                    next_sound,
                    max_seconds=ConcatUtil.PSEUDO_SILENCE_MAX_SECONDS,
                    threshold_db_relative_to_peak=ConcatUtil.PSEUDO_SILENCE_THRESHOLD_DB,
                ) or 0.0

                if end_pseudo + start_pseudo > 0:
                    base = reason_pauses.get_pause_for(break_effect if break_effect is not None else phrase.reason)
                    L.d(f"\n\npseudosilence - effect={break_effect} base={base:.3f} end={end_pseudo:.3f} start={start_pseudo:.3f} AMOUNT={end_pseudo + start_pseudo}\n")

            sound = SoundPipeline.append_pause_or_section_effect(
                sound,
                reason=phrase.reason,
                reason_pauses=reason_pauses,
                break_effect=break_effect,
                preceding_silence=end_pseudo,
                following_silence=start_pseudo,
                effect_level_offset_db=level_offset_db,
            )

            # Pan only direct narration; room sound stays independent of voice
            # position. The narration feeds its voice's room only. Appended
            # effects stay dry and centered.
            sound = spatial.apply(
                sound, pan=voice_pans[idx], send_end=narration_end,
                is_last=is_final_segment, room_reverb=voice_reverbs[idx],
            )

            # Constant noise bed under the segment and its pause or break
            # effect alike. Same duration.
            if room_tone_db is not None:
                if noise_bed is None:
                    noise_bed = NoiseBed(sound.sr, room_tone_db)
                sound = noise_bed.apply(sound, is_last=is_final_segment)

            # Room reverb adds to the leveled narration's peaks
            if loudness is not None:
                sound, mix_reduction_db = LoudnessLeveling.limit_mix(sound, loudness)
                leveling_stats.add_mix(mix_reduction_db)

            durations[idx] = sound.duration
            duration_sum += sound.duration
            ConcatUtil.add_audio_to_ffmpeg_stream(process, sound.data)

            if print_progress:
                s = f"{time_stamp(duration_sum, with_tenth=False)} {Path(path).stem[:80]} ... "
                print("\x1b[1G" + s, end="\033[K", flush=True)

        for i, (phrase, path) in enumerate(phrases_and_paths):

            if not path:
                # durations[i] already 0.0
                continue
            if Interrupts().did_interrupt:
                Interrupts().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            render_path = path
            temporary_upscaled_path = ""
            if use_upsampler:
                temporary_upscaled_path = f"{dest_path}.lava-{i}.flac"
                upsample_error = ModelWorker.upsample_file_blocking(
                    path,
                    temporary_upscaled_path,
                )
                if upsample_error:
                    ConcatUtil.close_ffmpeg_stream(process)
                    delete_silently(temporary_upscaled_path)
                    return upsample_error
                render_path = temporary_upscaled_path

            try:
                result = SoundPipeline.make_concat_rendered_sound_segment(
                    phrase, render_path, high_shelf,
                    reason_pauses=reason_pauses,
                    use_upsampler=False,
                    add_pause=False,
                )
            finally:
                if temporary_upscaled_path:
                    delete_silently(temporary_upscaled_path)
            if isinstance(result, str): # error
                ConcatUtil.close_ffmpeg_stream(process) # TODO clean up more and message user
                return result

            curr_sound = result

            # Short sentence-boundary silences are extended first, then over-long
            # gaps are cut (the extension target is capped at the max above, so
            # the two don't undo each other). Sidecar timings are against the
            # saved audio, hence the map below accounts for both steps.
            curr_sound, insertions = ConcatUtil.extend_sentence_gaps_for_segment(
                curr_sound, path, min_sentence_silence
            )

            # Over-long internal silence gaps are cut here rather than at
            # generation time, so re-concatenating saved audio with a different
            # threshold re-cuts it without regenerating. Word timings were
            # recorded against the un-cut audio, hence the map.
            curr_sound, gap_trims = SoundPipeline.limit_silence_gaps_with_trims(
                curr_sound,
                enabled=limit_silence_gaps,
                max_gap_duration=limit_silence_gaps_duration,
            )
            if gap_trims or insertions:
                gap_trim_maps[i] = GapTrimMap(result.duration, gap_trims, insertions)

            # Loudness leveling: after all per-segment shaping (high shelf
            # included, which affects K-weighted loudness), before the pause
            # and panning. Pseudo-silence detection below is relative to the
            # segment's peak, so it's unaffected by the gain. The voice's
            # volume lowers the leveled segment; the narration feeds its room
            # at that level, so the voice's dry/wet balance is unchanged.
            if loudness is not None:
                leveled = LoudnessLeveling.level(
                    curr_sound, loudness, pan=voice_pans[i], channels=channels, volume_db=voice_volume_dbs[i],
                )
                leveling_stats.add(leveled)
                curr_sound = leveled.sound

            if Interrupts().did_interrupt:
                Interrupts().clear()
                ConcatUtil.close_ffmpeg_stream(process)
                delete_silently(dest_path) # TODO delete parent dir silently if empty
                return "Interrupted by user"

            # Flush the previously-held segment using the current segment as the
            # "next" segment for pseudo-silence compensation.
            if pending is not None:
                flush_pending(curr_sound)

            pending = (curr_sound, phrase, i, path)

        # Flush the final held segment. No following segment exists, so no
        # pseudo-silence compensation is applied (hardcoded pause is used).
        # Break effects are suppressed because they should only separate audio
        # segments, never trail the output file.
        if pending is not None:
            flush_pending(None)

        if print_progress:
            printt()
        printt()

        if print_progress:
            ConcatUtil.print_render_metrics(gap_trim_maps, leveling_stats if loudness is not None else None, loudness)

        Interrupts().clear()
        ConcatUtil.close_ffmpeg_stream(process)
        return ConcatRenderResult(durations=durations, gap_trim_maps=gap_trim_maps)

    @staticmethod
    def print_render_metrics(
        gap_trim_maps: list[GapTrimMap | None],
        leveling_stats: LevelingStats | None = None,
        loudness: NormalizationSpecs | None = None,
    ) -> None:
        """
        Prints post-render metrics as a single contiguous block, followed by
        one blank line (nothing at all if there's nothing to report).
        """
        lines: list[str] = []
        gap_maps = [gap_map for gap_map in gap_trim_maps if gap_map]

        if leveling_stats is not None and loudness is not None and leveling_stats.count:
            lines.append(leveling_stats.summary(loudness))

        insertion_count = sum(len(gap_map.insertions) for gap_map in gap_maps)
        if insertion_count:
            added = sum(gap_map.inserted_duration for gap_map in gap_maps)
            lines.append(f"Extended inner sentence gaps: {insertion_count} gaps, {added:.2f}s added")

        gap_trim_count = sum(len(gap_map.trims) for gap_map in gap_maps)
        if gap_trim_count:
            removed = sum(
                trim.removed_duration for gap_map in gap_maps for trim in gap_map.trims
            )
            lines.append(f"Limited silence gaps: {gap_trim_count} gaps, {removed:.2f}s removed")

        if not lines:
            return
        for line in lines:
            printt(line)
        printt()

    @staticmethod
    def extend_sentence_gaps_for_segment(
            sound: Sound,
            segment_path: str,
            min_sentence_silence: float,
    ) -> tuple[Sound, list[SentenceGapInsertion]]:
        """
        Loads the segment's sidecar and extends its short sentence-boundary
        silences (with digital silence). Returns the sound unchanged (and no
        insertions) if disabled or if the sidecar is missing/unreadable/has
        unusable timings.
        """
        if min_sentence_silence <= 0:
            return sound, []

        sidecar_path = get_segment_stt_info_path(segment_path)
        if not sidecar_path.exists():
            return sound, []
        try:
            # from_dict can raise on malformed legacy metadata even when the
            # JSON itself loaded successfully. Keep this guard around data only,
            # not the audio pipeline, so unrelated processing errors still surface.
            info = SegmentTranscriptUtil.load(sidecar_path)
            if isinstance(info, str):
                raise ValueError(info)
            _validate_sentence_gap_timings(info.timed_phrases, info.transcript_words)
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            L.w(f"Couldn't use sidecar for sentence gap extension: {sidecar_path}: {error}")
            return sound, []
        if not info.transcript_words or len(info.timed_phrases) < 2:
            return sound, []

        return SoundPipeline.extend_sentence_gaps_with_insertions(
            sound, info.timed_phrases, info.transcript_words, min_sentence_silence
        )

    @staticmethod
    def init_ffmpeg_stream(
            dest_path: str,
            is_aac_not_flac: bool,
            aac_bitrate: str=AAC_BITRATE_DEFAULT,
            channels: int = 1,
    ) -> subprocess.Popen:
        """
        Initializes and returns an ffmpeg process for streaming FLAC encoding.
        Stereo input is interleaved (n, 2) data, see add_audio_to_ffmpeg_stream.
        """
        args = [
            "ffmpeg",
            "-y"  # Overwrite output file if it exists
        ]
        # Input stream related
        args.extend([
            "-f", "s16le",
            "-ar", f"{APP_SAMPLE_RATE}",
            "-ac", str(channels),
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
        Converts float audio data to 16-bit PCM format. Stereo is (n, 2), which
        `tobytes` writes interleaved, as s16le expects.
        """
        # Float audio in [-1.0, 1.0] to 16-bit PCM. Out-of-range floats don't
        # saturate when cast to int16 (they wrap around, ie loud crackles), so
        # clip first as a safety net.
        pcm_data = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16) # type: ignore

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
                ask.ask_error("No chapter files have generated audio")
                return

            ConcatUtil.make_files(
                state=state,
                file_cut_indices=output_indices,
                bookmark_indices=[]
            )
            return

        if state.project.can_use_bookmark_section_markers():
            bookmark_indices = sorted(state.project.markers)
        else:
            bookmark_indices = []

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

    extant_file_paths: list[str] = []
    num_missing = 0

    for group_index in range(len(project.phrase_groups)):
        out_of_range = (group_index < index_start or group_index > index_end)
        if out_of_range:
            continue
        stem = project.sound_segments.get_best_file_for(group_index)
        if stem:
            file_path = os.path.join(project.sound_segments_path, stem)
            extant_file_paths.append(file_path)
        else:
            num_missing += 1

    # Make Filename
    extant_file_names = extant_file_paths
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
    # [6] voice tag (most common voice; "+n" = count of other distinct voices in the set)
    voice_tag, voice_extra = SoundSegmentUtil.get_voice_tag_summary(extant_file_names)
    if voice_tag:
        extra = f" +{voice_extra}" if voice_extra > 0 else ""
        stem += f"[{voice_tag}{extra}]" + " "

    stem = stem.strip()
    return stem

def make_subdivided_timed_phrases(
        timed_phrases: list[TimedPhrase],
        phrase_groups: list[PhraseGroup],
        sound_paths: list[str],
        sound_durations: list[float],
        bookmark_indices: list[int],
        gap_trim_maps: list[GapTrimMap | None] | None = None,
    ) -> tuple[list[AppMetadataTextSegment], list[int], list[int]]:
    """
    Break timed PhraseGroups into their constituent phrases. Valid forced-alignment
    sidecars are preserved as nested lists in ABR metadata. Groups without audio
    are likewise emitted as nested lists of zero-timed constituent phrases so the
    browser player can render them as no-audio segment groups.

    The first four arguments are parallel lists. gap_trim_maps, if given, is
    parallel to them too and carries the gap-cut map of each segment, recorded
    while concatenating: sidecar word timings are measured against the un-cut
    audio, so they are mapped through it before being used as offsets.

    Returns updated timed_phrases, flattened-leaf bookmark indices, and the
    flattened-leaf starting index corresponding to each original PhraseGroup.
    """

    if not (len(timed_phrases) == len(phrase_groups) == len(sound_paths) == len(sound_durations)):
        raise ValueError("lists must have same lengths")

    if gap_trim_maps is not None and len(gap_trim_maps) != len(timed_phrases):
        raise ValueError("gap_trim_maps must have the same length as the other parallel lists")

    new_timed_phrases: list[AppMetadataTextSegment] = []
    new_bookmark_indices: list[int] = []
    phrase_to_text_segment_start_indices: list[int] = []
    leaf_count = 0

    for i in range(0, len(timed_phrases)):
        phrase_to_text_segment_start_indices.append(leaf_count)

        def add_to_new_bookmark_indices(debug_reason: str, debug_text: str) -> None:
            if i in bookmark_indices:
                new_bookmark_indices.append(leaf_count)
                if False:
                    print("added", leaf_count, "new len", len(new_bookmark_indices), debug_reason, debug_text)

        original_timed_phrase = timed_phrases[i]
        sound_path = sound_paths[i]

        if not sound_path:
            phrase_group = phrase_groups[i]
            if not phrase_group.phrases:
                add_to_new_bookmark_indices("no-time-segment", original_timed_phrase.presentable_text)
                new_timed_phrases.append(original_timed_phrase)
                leaf_count += 1
                continue

            missing_phrases: list[TimedPhrase] = []
            for phrase_index, phrase in enumerate(phrase_group.phrases):
                if phrase_index == 0:
                    add_to_new_bookmark_indices("first-missing-audio-phrase", phrase.presentable_text)
                missing_phrases.append(TimedPhrase.make_using(phrase, 0.0, 0.0))
            new_timed_phrases.append(missing_phrases)
            leaf_count += len(missing_phrases)
            continue

        subdivided_items_json_path = get_segment_stt_info_path(sound_path)
        if not subdivided_items_json_path.exists():
            L.w(f"Missing segment timing/STT sidecar JSON: {subdivided_items_json_path}")
            add_to_new_bookmark_indices("no-subdivided-json", original_timed_phrase.presentable_text)
            new_timed_phrases.append(original_timed_phrase)
            leaf_count += 1
            continue

        parse_result = SegmentTranscriptUtil.load_timed_phrases(subdivided_items_json_path)
        if isinstance(parse_result, str) or not parse_result:
            # File/parse error or empty data; use original item.
            add_to_new_bookmark_indices("parse-error", original_timed_phrase.presentable_text)
            new_timed_phrases.append(original_timed_phrase)
            leaf_count += 1
            continue
        subdivided_timed_phrases = parse_result

        # Finally, do subdivision action.
        gap_trim_map = gap_trim_maps[i] if gap_trim_maps is not None else None

        def map_segment_time(time: float) -> float:
            return time if gap_trim_map is None else gap_trim_map.map_time(time)

        updated_subdivisions: list[TimedPhrase] = []
        offset = original_timed_phrase.time_start
        previous_mapped_time_end = 0.0
        for subdivided_index, item in enumerate(subdivided_timed_phrases):
            if subdivided_index == 0:
                add_to_new_bookmark_indices("first-subdivision", item.presentable_text)
                time_start = offset
            else:
                time_start = offset + previous_mapped_time_end
            time_end = offset + map_segment_time(item.time_end)
            # map_time() is monotonic, but a word inside a removed gap clamps
            # onto the previous leaf, so keep the leaves non-decreasing anyway.
            time_end = max(time_end, time_start)
            previous_mapped_time_end = time_end - offset
            updated_subdivisions.append(TimedPhrase(item.text, time_start, time_end))

        # Set the last item's time_end using the duration of the source audio clip
        # rather than the transcription end word timestamp. This prevents
        # discontinuities in segment selectedness across boundaries.
        updated_subdivisions[-1].time_end = sound_durations[i] + offset
        new_timed_phrases.append(updated_subdivisions)
        leaf_count += len(updated_subdivisions)

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
