from __future__ import annotations

from dataclasses import dataclass
import os
import time

import numpy as np

from tts_audiobook_tool import app_support, ask, text_util
from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant, VoiceSelectMode
from tts_audiobook_tool.app_support import app_hint_util, app_memory
from tts_audiobook_tool.concat_util import ConcatUtil
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool.generation_events import (
    GenerationEvents,
    GenerationPhase,
    GenerationProgress,
    GenerationStarted,
    GenerationStats,
    ModelUnhealthy,
)
from tts_audiobook_tool.gen_timeout_util import GenTimeoutTracker
from tts_audiobook_tool.project_support.project_util import ProjectUtil
from tts_audiobook_tool.project_support.project_voice_util import ProjectVoiceUtil
from tts_audiobook_tool.project_support.segment_transcript_util import SegmentTranscriptUtil
from tts_audiobook_tool.app_types.phrase import PhraseGroup
from tts_audiobook_tool import readiness
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.sound.sound_extra_util import SoundExtraUtil
from tts_audiobook_tool.app_types.segment_transcript_data import SegmentTranscriptData
from tts_audiobook_tool.sound.silence_util import SilenceUtil
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil, get_segment_stt_info_path
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.text_ops.range_string_util import RangeStringUtil
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.app_types.validation_result import MusicFailResult, SkippedResult, ExcessiveDurationResult, TranscriptResult, TrimmedResult, ValidationResult, WordErrorResult
from tts_audiobook_tool.sound.silence_util import SilenceGapTrim
from tts_audiobook_tool.transcriber import Transcriber


@dataclass(frozen=True)
class TtsModelError:
    message: str

@dataclass(frozen=True)
class SubBatch:
    """
    One model-inference call's worth of items, all of which use a single
    voice sample.

    `voice_selection_index` is None when the existing voice logic inside
    `GenerateUtil.generate()` applies (auto-advance rotation, disabled mode,
    or a single-item user-defined selection).
    """
    voice_selection_index: int | None
    items: tuple[tuple[int, int], ...]

# Number of batch iterations applied per group in the windowed word-count
# ordering used to preserve a semblance of index monotonicity.
BATCH_ITERATIONS_PER_GROUP = 5

# Number of per-batch durations kept in the rolling history used to
# extrapolate the estimated time remaining.
BATCH_DURATION_HISTORY_SIZE = 50

# The ETA is first shown once this many durations have been recorded;
# further samples only refine the rolling average.
ETA_MIN_BATCH_DURATIONS = 10


def make_consecutive_model_errors_reset_message(max_consecutive_model_errors: int) -> str:
    """Single source of the reason text for a consecutive-model-errors abort."""
    return (
        f"Too many consecutive TTS model errors ({max_consecutive_model_errors}); "
        "generation loop aborted; model-worker hard reset required"
    )


def make_oom_reset_message() -> str:
    """Single source of the reason text for an OOM abort."""
    return (
        "TTS model ran out of memory; generation loop aborted; "
        "model-worker hard reset required"
    )


class GenerateUtil:

    @staticmethod
    def generate_files(
            state: State,
            indices_set: set[int],
            batch_size: int,
            is_regen: bool
    ) -> bool:
        """
        Subroutine for doing a series of audio generations to files
        Prints feedback at end of each item, and summary at end of loop.

        :param batch_size:
            When set to 1, batch mode is effectively disabled.

        Returns:
            True if ended because interrupted (user pressed control-c)
            or aborted (OOM detected or too many consecutive model errors;
            an abort also emits a ModelUnhealthy event so the main process
            hard-resets the model worker)
        """

        project = state.project
        max_retries = project.max_retries
        stt_variant = state.prefs.stt_variant
        stt_config = state.prefs.stt_config
        showed_vram_warning = False

        GenerationEvents.emit(GenerationPhase("Preparing models"))
        warm_up_result = ModelManager.warm_up_models(state)
        if warm_up_result.should_stop:
            app_support.print_warm_up_result_stop(warm_up_result)
            if warm_up_result.error:
                app_memory.gc_ram_vram()
            return True

        # Do model readiness check now that model instance exists
        err = readiness.get_generate_blocker_text(state, verbose=True)
        if err:
            ask.ask_error(err)
            return True

        # Print warnings if any
        warnings = Tts.get_instance().get_warning_issues(state.project)
        if warnings:
            message = "\n".join(warnings)
            print_feedback(Ansi.ITALICS + message, no_preformat=True)

        showed_vram_warning = app_memory.show_vram_memory_warning_if_necessary()

        # Make 'items' (tuple = phase group index, retry_count)
        sorted_indices = sorted(list(indices_set))
        GenerationEvents.emit(GenerationStarted(total=len(sorted_indices)))
        items: list[tuple[int, int]] = []
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]
            items.append((index, 0))

        # Build a queue of rounds. A "round" is a group of sub-batches (a
        # "batch of batches"); each sub-batch is sent to the model as a single
        # call that can use only one voice sample.
        #
        # For user-defined voice selections, lines are grouped into per-voice
        # sub-batches (see make_multi_voice_rounds). Otherwise the queue
        # degenerates to the single-voice shape: a flat, optionally
        # word-count-bucketed list of items chunked into sub-batches of
        # batch_size, and the voice is resolved by the existing logic inside
        # generate() (auto-advance rotation or disabled mode).
        is_cjk = ProjectVoiceUtil.is_language_cjk(state.project)
        voice_of_index: dict[int, int] | None = None
        if batch_size > 1 and project.voice_select_mode == VoiceSelectMode.USER_DEFINED:
            voice_values = ProjectVoiceUtil.get_voice_values(project, Tts.get_type())
            if voice_values:
                voice_of_index = effective_voice_indices(
                    project.phrase_groups, sorted_indices, len(voice_values)
                )
        if voice_of_index is not None:
            queue = make_multi_voice_rounds(
                items, project.phrase_groups, voice_of_index, batch_size,
                sort_by_words=not is_cjk,
            )
        else:
            if batch_size > 1 and not is_cjk:
                items = bucket_items(items, project.phrase_groups, batch_size)
            queue = make_single_voice_rounds(items, batch_size)

        # Metrics-related variables
        num_errored = 0
        num_failed = 0
        num_failed_music = 0
        num_improved = 0
        num_passed = 0
        num_retries = 0
        saved_indices: set[int] = set()
        word_counts: dict[int, int] = {}
        preexisting_word_error_counts = project.sound_segments.get_word_error_counts_in_generate_range()
        best_word_error_counts = dict(preexisting_word_error_counts)
        gen_val_sum_time = 0
        start_time = time.time()
        consecutive_model_errors = 0
        max_consecutive_model_errors = 5
        # Rolling per-batch duration history for the ETA extrapolation. The
        # run's first batch is never recorded (warm-up dominates it).
        batch_durations: list[float] = []
        did_skip_warmup_batch = False
        # Recalculated once per completed batch; never time-interpolated.
        eta_seconds: float | None = None

        Interrupts().set("generating")
        did_interrupt = False
        did_abort_model_errors = False

        # Audiobook generation is a top-level generation run. Start from a
        # fresh rolling-continuation context, independent of any prior run.
        Tts.clear_continuation()
        Tts.reset_voice_selection_index()

        # Loop through the queue of rounds. Retries (re-added items) are
        # collected and processed at the head of the queue, as their own
        # round(s), before any further items from the main queue.
        pending_retries: list[tuple[int, int]] = []
        current_round: list[SubBatch] = []
        last_voice: int | None = None
        is_first_batch = True

        # The first gen may be dominated by model warm-up or download time,
        # so it alone is exempt from the GEN_TIMEOUT watchdog.
        # Note, we are keeping this even for sgl-omni mode
        gen_timeout_tracker = GenTimeoutTracker()

        while True:

            if Interrupts().did_interrupt:
                did_interrupt = True
                break

            # Start a new round if needed (retries take priority over the queue)
            if not current_round:
                if pending_retries:
                    current_round = make_retry_round(pending_retries, voice_of_index, batch_size)
                    pending_retries = []
                elif queue:
                    current_round = queue.pop(0)
                else:
                    break

            sub = current_round.pop(0)

            # Make parallel arrays from 'sub'
            indices = [item[0] for item in sub.items]
            retry_counts = [item[1] for item in sub.items]
            num_retries += sum(retry_count > 0 for retry_count in retry_counts)

            # A rolling-continuation context is per voice sample, so do not
            # carry it across a voice sample change mid-run.
            if (
                sub.voice_selection_index is not None
                and last_voice is not None
                and sub.voice_selection_index != last_voice
            ):
                Tts.clear_continuation()
            last_voice = sub.voice_selection_index

            if not showed_vram_warning:
                b = app_memory.show_vram_memory_warning_if_necessary()
                if b:
                    print("\a", end="")
                    showed_vram_warning = True

            # Print item info and publish a structured progress snapshot.
            # The snapshot carries the previous batch's ETA so the header
            # does not blank the estimate during this batch's inference.
            num_processed = num_passed + num_failed + num_errored
            GenerationEvents.emit(GenerationPhase("Generating audio"))
            GenerationEvents.emit(
                GenerationProgress(
                    processed=num_processed,
                    remaining=len(sorted_indices) - num_processed,
                    total=len(sorted_indices),
                    current_indices=tuple(indices),
                    passed=num_passed,
                    failed=num_failed,
                    errored=num_errored,
                    retries=num_retries,
                    eta_seconds=eta_seconds,
                )
            )
            GenerateUtil.print_batch_heading(
                indices=indices,
                voice_index=sub.voice_selection_index,
                show_divider=not is_first_batch,
            )
            is_first_batch = False

            # Generate and validate
            gen_start_time = time.time()
            with gen_timeout_tracker.scope() as gen_timeout_guard:
                results = GenerateUtil.generate_and_validate_batch(
                    state=state,
                    indices=indices,
                    phrase_groups=project.phrase_groups,
                    stt_variant=stt_variant, stt_config=stt_config,
                    force_random_seed=is_regen or any(count > 0 for count in retry_counts),
                    is_realtime=False,
                    voice_selection_index=sub.voice_selection_index,
                )
            if gen_timeout_guard.did_time_out:
                # The watchdog already reported the timeout and requested the
                # worker reset; abort the run regardless of any pending cancel.
                printt(f"{COL_ERROR}Aborting generation run")
                Tts.clear_continuation()
                did_interrupt = True
                break
            batch_gen_val_time = time.time() - gen_start_time
            gen_val_sum_time += batch_gen_val_time

            # Check for OOM in results and break early if detected
            if any(GenerateUtil.is_error_result_oom(r) for r in results):
                printt()
                first_oom = next(GenerateUtil.get_error_result_message(r) for r in results if GenerateUtil.is_error_result_oom(r))
                print_gen_oom_message(first_oom)
                GenerationEvents.emit(ModelUnhealthy(reason=make_oom_reset_message()))
                Tts.clear_continuation()
                did_interrupt = True
                break

            # Update the rolling duration history for the ETA extrapolation.
            # The first batch is skipped (model warm-up dominates it), and a
            # batch containing a model-derived error is not representative.
            if not did_skip_warmup_batch:
                did_skip_warmup_batch = True
            elif not any(isinstance(result, TtsModelError) for result in results):
                batch_durations.append(batch_gen_val_time)
                if len(batch_durations) > BATCH_DURATION_HISTORY_SIZE:
                    batch_durations.pop(0)

            # Process and print results # TODO: separate biz n print logic
            re_adds: list[tuple[int, int]] = []

            for i, result in enumerate(results):

                is_model_error = isinstance(result, TtsModelError)
                if isinstance(result, TtsModelError):
                    error_string = result.message
                    validation_result = None
                elif isinstance(result, str):
                    error_string = result
                    validation_result = None
                else:
                    error_string = ""
                    validation_result = result

                index = indices[i]
                phrase_group = project.phrase_groups[index]
                stt_info: SegmentTranscriptData | None = None
                should_show_viz = False
                message_lines = []

                retry_string = f" (retry #{retry_counts[i]})" if retry_counts[i] else ""
                item_line = f"{COL_ACCENT}Line {index + 1}{retry_string}:" # {COL_DEFAULT}{text_string}"
                message_lines.append(item_line)

                new_retry_count = retry_counts[i] + 1


                if error_string:
                    # Model error
                    if is_model_error:
                        consecutive_model_errors += 1
                    else:
                        consecutive_model_errors = 0
                    did_abort_model_errors = consecutive_model_errors >= max_consecutive_model_errors
                    val_line = f"{COL_ERROR}Error: {error_string}"
                    if did_abort_model_errors:
                        val_line += f"; {COL_ERROR}too many consecutive TTS model errors"
                    elif new_retry_count > max_retries:
                        num_errored += 1
                        val_line += f"; {COL_ERROR}max retries reached"
                    else:
                        re_adds.append((index, new_retry_count))
                        val_line += f"; {COL_DEFAULT}will retry"
                    message_lines.append(val_line)

                elif validation_result:
                    consecutive_model_errors = 0

                    val_line = f"{validation_result.get_ui_message_with_extras()}"
                    validation_word_error_count: int | None = None
                    if isinstance(validation_result, TranscriptResult):
                        validation_word_error_count = SegmentTranscriptUtil.make_generation_word_error_count(validation_result)
                    previous_word_error_count = best_word_error_counts.get(index, None)
                    did_improve = (
                        (is_regen or retry_counts[i] > 0)
                        and validation_word_error_count is not None
                        and previous_word_error_count is not None
                        and validation_word_error_count < previous_word_error_count
                    )
                    if did_improve:
                        num_improved += 1
                    if validation_word_error_count is not None and (
                        previous_word_error_count is None or validation_word_error_count < previous_word_error_count
                    ):
                        best_word_error_counts[index] = validation_word_error_count

                    # Modify validation ui message if needed
                    if not validation_result.is_fail:
                        if is_regen:
                            val_line = val_line.replace("Passed", f"{COL_OK}Passed")
                        else:
                            if new_retry_count > 1:
                                val_line = val_line.replace("Passed", f"{COL_OK}Passed on retry")
                    else:
                        if is_regen:
                            if isinstance(validation_result, WordErrorResult):
                                preexisting_count = preexisting_word_error_counts.get(index, None)
                                new_count = validation_result.num_errors
                                if preexisting_count is not None and validation_result.num_errors < preexisting_count:
                                    val_line += f" (improved from {preexisting_count} to {new_count} word errors)"
                                    preexisting_word_error_counts[index] = new_count
                        elif did_improve and isinstance(validation_result, WordErrorResult):
                            val_line += f" (improved from {previous_word_error_count} to {validation_result.num_errors} word errors)"

                    # Failed or not
                    if validation_result.is_fail:
                        Tts.clear_continuation()
                        if new_retry_count > max_retries:
                            num_failed += 1
                            val_line += f"; {COL_ERROR}max retries reached, tagging as failed"
                        else:
                            re_adds.append((index, new_retry_count))
                            val_line += f"; {COL_DEFAULT}will retry"
                        if isinstance(validation_result, MusicFailResult):
                            num_failed_music += 1
                    else:
                        num_passed += 1

                    # Print STT info
                    if isinstance(validation_result, TranscriptResult):
                        stt_info = SegmentTranscriptUtil.from_validation_result(
                            project=project,
                            phrase_group=phrase_group,
                            index=index,
                            validation_result=validation_result
                        )
                        should_show_viz = (
                            isinstance(validation_result, (MusicFailResult, ExcessiveDurationResult))
                            or (isinstance(validation_result, WordErrorResult) and validation_result.num_errors > 0)
                        )

                    val_line += Ansi.RESET
                    message_lines.append(val_line)

                    # Update word count metrics
                    word_counts[index] = phrase_group.num_words

                    # Save
                    err, saved_path = GenerateUtil.save_sound_and_timing_json(
                        state,
                        phrase_group,
                        index,
                        validation_result,
                        is_real_time=False,
                        voice_tag=getattr(validation_result, "voice_tag", ""),
                        stt_info=stt_info,
                    )
                    if err:
                        save_line = f"{COL_ERROR}Couldn't save file: {err} {saved_path}"
                    else:
                        saved_indices.add(index)
                        project.sound_segments.delete_redundants_for(index)

                        url = saved_path
                        text = Path(saved_path).name
                        link = text_util.make_terminal_hyperlink(url=url, text=text, is_file=True)
                        save_line = f"Saved: {COL_DIM}{link}{Ansi.RESET}"

                    message_lines.append(save_line)

                printt("\n".join(message_lines))

                if validation_result:
                    printt()
                    if stt_info is not None:
                        printt(
                            text_util.combine_ansi_lines(
                                SegmentTranscriptUtil.make_stt_details_lines(
                                    stt_info,
                                    should_show_diff=should_show_viz,
                                )
                            )
                        )
                    else:
                        # Just print source text, in a similar style
                        printt(f"{COL_DEFAULT}Source text: {COL_DIM_ITALICS}{phrase_group.presentable_text.strip()}")

                if i < len(results) - 1:
                    printt()

                if did_abort_model_errors:
                    break

            num_processed = num_passed + num_failed + num_errored

            if re_adds:
                # Pending retries are processed at the head of the queue,
                # as the next round(s) before any further items from the
                # main queue.
                pending_retries.extend(re_adds)

            # Extrapolate the remaining time from the rolling history, once
            # per completed batch (the consumer never interpolates it).
            # Remaining work is counted in sub-batches (one model call
            # each): the current round's remainder, the queued rounds, and
            # a ceil-estimate for the pending retries, whose items have
            # already left the main queue.
            eta_seconds = make_batch_eta_seconds(
                batch_durations,
                remaining_batches=(
                    len(current_round)
                    + sum(len(round) for round in queue)
                    + (len(pending_retries) + batch_size - 1) // batch_size
                ),
            )
            GenerationEvents.emit(
                GenerationProgress(
                    processed=num_processed,
                    remaining=len(sorted_indices) - num_processed,
                    total=len(sorted_indices),
                    passed=num_passed,
                    failed=num_failed,
                    errored=num_errored,
                    retries=num_retries,
                    eta_seconds=eta_seconds,
                )
            )

            if did_abort_model_errors:
                printt()
                GenerateUtil.print_consecutive_model_errors_message(max_consecutive_model_errors)
                GenerationEvents.emit(
                    ModelUnhealthy(
                        reason=make_consecutive_model_errors_reset_message(max_consecutive_model_errors)
                    )
                )
                Tts.clear_continuation()
                did_interrupt = True
                break

            # Memory usage is displayed by the generation header, not the
            # console; keep only the blank-line separator between batches.
            printt()

        # Update the persisted queue once per generation run, based only on files
        # that were actually saved successfully.
        if saved_indices:
            selected_indices = ProjectUtil.get_indices_to_generate(project)
            remaining_indices = selected_indices - saved_indices
            updated_range_string = RangeStringUtil.make_ranges_string(
                remaining_indices,
                len(project.phrase_groups),
            )
            if updated_range_string != project.generate_range_string:
                project.generate_range_string = updated_range_string
                save_error = project.save()
                if save_error:
                    ask.ask_error(
                        f"Couldn't save updated generation range: {save_error}",
                    )

        # Build the summary even when it will not be shown so this metrics
        # formatting remains one straightforward path. Clean quick generations
        # return directly to the editor; stopped or problematic quick runs keep
        # the summary visible for review.
        message = ""
        if did_interrupt:
            Tts.clear_continuation()
        message += f"Elapsed: {duration_string(time.time() - start_time)}\n"
        ok = str(num_passed)
        if num_passed == len(sorted_indices):
            ok += " (all)"
        message += f"Lines saved: {COL_OK}{ok}{COL_DEFAULT}\n"
        if Stt.has_instance() and ModelManager.has_yamnet_detector():
            message += f"Num retries triggered due to detected music: {num_failed_music}\n"
        message += f"Num retries: {num_retries}\n"
        if num_improved:
            message += f"Lines improved on retry: {COL_OK}{num_improved}{COL_DEFAULT}\n"
        col = COL_ACCENT if num_failed else ""
        if num_failed:
            message += f"Lines saved, but with excess word errors: {col}{num_failed}{COL_DEFAULT}\n"
        if num_errored:
            message += f"Lines failed to generate: {COL_ERROR}{num_errored}{COL_DEFAULT}\n"
        if DEV:
            message += f"Num words: {sum(word_counts.values())}\n"
            message += f"Gen/val elapsed: {duration_string(gen_val_sum_time)}\n"
        quick_generation_completed_cleanly = (
            is_regen
            and not did_interrupt
            and num_failed == 0
            and num_errored == 0
        )
        if not quick_generation_completed_cleanly:
            printt(message)

        # Prevent rolling-continuation state from leaking into a later run when
        # this run ends successfully without a paragraph/section break.
        Tts.clear_continuation()

        Interrupts().clear()
        return did_interrupt

    @staticmethod
    def generate_and_validate_batch(
        state: State,
        indices: list[int],
        phrase_groups: list[PhraseGroup],
        stt_variant: SttVariant,
        stt_config: SttConfig,
        force_random_seed: bool,
        is_realtime: bool,
        is_skip_reason_buffer: bool=False,
        voice_selection_index: int | None=None
    ) -> list[ValidationResult | str | TtsModelError]:
        """
        Generates and validates a batch of prompts from the Project text.
        Prints updates.
        Returns a list of results (ValidationResult or error string).

        :param indices:
            When length is 1, batch mode is disabled
        :param voice_selection_index:
            When set, all items in the batch are generated with this voice
            sample (a batch must be voice-homogeneous). When None, the voice
            is resolved by the existing logic inside `generate()`.
        """

        project = state.project
        save_debug_files = state.prefs.save_debug_files

        # Set print color to dim during any model inference printouts
        print(f"{COL_DIM}", end="")

        # Generate:
        gen_start_time = time.time()
        gen_results = GenerateUtil.generate(
            project=project,
            indices=indices,
            phrase_groups=phrase_groups,
            force_random_seed=force_random_seed,
            is_realtime=is_realtime,
            save_debug_files=save_debug_files,
            print_generation_request=True,
            voice_selection_index=voice_selection_index
        )
        printt() # Restore print color, print blank line

        # Print speed info
        print_speed_info(time.time() - gen_start_time, gen_results)

        # Should skip or not
        skip_reason = Stt.should_skip(state, is_skip_reason_buffer)
        if not skip_reason:
            GenerationEvents.emit(GenerationPhase("Transcribing and validating"))
            printt(f"{COL_DEFAULT}Transcribing audio...", end="") # gets overwritten
        else:
            GenerationEvents.emit(GenerationPhase("Processing generated audio"))

        val_start_time = time.time()
        results: list[ValidationResult | str | TtsModelError] = []

        for i, gen_result in enumerate(gen_results):

            index = indices[i]

            if isinstance(gen_result, TtsModelError):
                err = gen_result.message
                printt(f"Text segment {index+1} - error: {err}")
                results.append(gen_result)
                continue

            if isinstance(gen_result, str):
                err = gen_result
                printt(f"Text segment {index+1} - error: {err}")
                results.append(err)
                continue

            sound, gap_trims, start_trim_time, end_trim_time, original_duration, token_noise_trim_time, voice_tag = gen_result

            if skip_reason:
                validation_result = SkippedResult(sound=sound, message=skip_reason)
                validation_result.intra_sample_silence_trims = gap_trims
                validation_result.generated_start_trim_time = start_trim_time
                validation_result.generated_end_trim_time = end_trim_time
                validation_result.generated_trim_original_duration = original_duration
                validation_result.trailing_token_noise_trim_time = token_noise_trim_time
                validation_result.voice_tag = voice_tag
                results.append((validation_result))
                continue

            # Transcribe
            gen_result = Transcriber.transcribe_to_words(
                sound, project.language_code, stt_variant, stt_config
            )
            if isinstance(gen_result, str):
                err = gen_result
                results.append(err)
                continue
            transcribed_words = gen_result

            # Validate
            text = phrase_groups[ indices[i] ].as_flattened_phrase().text
            validation_result = Validator.validate(
                sound, text, transcribed_words, project.language_code, strictness=project.strictness
            )
            validation_result.intra_sample_silence_trims = gap_trims
            validation_result.generated_start_trim_time = start_trim_time
            validation_result.generated_end_trim_time = end_trim_time
            validation_result.generated_trim_original_duration = original_duration
            validation_result.trailing_token_noise_trim_time = token_noise_trim_time
            validation_result.voice_tag = voice_tag
            results.append(validation_result)

            if save_debug_files and isinstance(validation_result, TrimmedResult):
                GenerateUtil.save_debug_sound(
                    project=project,
                    index=index,
                    label="trim",
                    sound=validation_result.sound,
                    is_realtime=is_realtime
                )

        if not skip_reason:
            message = f"{COL_DEFAULT}Transcribed audio in {(time.time() - val_start_time):.1f}s"
            printt(f"{Ansi.LINE_HOME}{message}")

        printt()

        return results

    @staticmethod
    def generate(
            project: Project,
            indices: list[int],
            phrase_groups: list[PhraseGroup],
            force_random_seed: bool,
            is_realtime: bool,
            save_debug_files: bool,
            print_generation_request: bool = False,
            voice_selection_index: int | None = None
        ) -> list[tuple[Sound, list[SilenceGapTrim], float | None, float | None, float, float | None, str] | str | TtsModelError]:
        """
        Core audio generation function.

        Returns a list of "results" (same length as that of prompt_text),
        where each result is either a generated Sound or an error string.
        The Sound is trimmed for silence and peak-normalized.

        param indices:
            Use a one-element list to not generate in batch mode
        param voice_selection_index:
            When set, all items are generated with this voice sample (the
            batch must be voice-homogeneous). When None, the voice is
            resolved from the project's voice selection mode.
        """

        if len(indices) == 0:
            raise ValueError("Indices cannot be empty")
        if len(indices) > 1 and not Tts.get_type().value.can_batch:
            raise ValueError("Logic error - Model does not support batching")

        # Make prompts (parallel list to indices)
        prompts = []
        for index in indices:
            prompt = GenerateUtil.phrase_group_to_prompt(phrase_groups[index], project)
            prompts.append(prompt)

        # Generate
        voice_tag = ""
        if Tts.get_type() == TtsModelType.NONE:
            result = "No active TTS model"
        else:
            voice_values = ProjectVoiceUtil.get_voice_values(project, Tts.get_type())
            has_voice_values = bool(voice_values)
            is_user_defined_voice_mode = project.voice_select_mode == VoiceSelectMode.USER_DEFINED
            use_custom_voice_index = (
                len(indices) == 1
                and is_user_defined_voice_mode
                and has_voice_values
            )
            if voice_selection_index is None:
                if len(indices) > 1 and is_user_defined_voice_mode and has_voice_values:
                    raise ValueError(
                        "Logic error - A user-defined multi-item batch must be "
                        "split into per-voice sub-batches with explicit "
                        "voice_selection_index values (see make_multi_voice_rounds)"
                    )
                if project.voice_select_mode == VoiceSelectMode.DISABLED:
                    voice_selection_index = 0
                elif use_custom_voice_index:
                    requested_voice_index = phrase_groups[indices[0]].voice_index
                    # Silently clamp invalid (presumably stale) index to the nearest valid voice index
                    voice_selection_index = min(
                        max(requested_voice_index, 0),
                        len(voice_values) - 1,
                    )
                else:
                    voice_selection_index = Tts.get_next_voice_selection_index()
            voice_tag = Tts.get_voice_tag_for_selection_index(project, voice_selection_index)
            result = Tts.generate_using_project(
                project,
                prompts,
                force_random_seed,
                voice_selection_index=voice_selection_index,
                print_generation_request=print_generation_request
            )

        # `result` is either n generated Sounds or a single error string
        if isinstance(result, str):
            Tts.clear_continuation()
            return [TtsModelError(result) for _ in range(len(prompts))] # return n model errors

        sounds = [result] if isinstance(result, Sound) else result

        results: list[tuple[Sound, list[SilenceGapTrim], float | None, float | None, float, float | None, str] | str | TtsModelError] = [] # TODO: Revisit

        for i, sound in enumerate(sounds):

            if isinstance(sound, str):
                result = TtsModelError(sound)
                results.append(result)
                Tts.clear_continuation()
                Tts.clear_continuation_if_reason(phrase_groups[indices[i]].last_reason)
                continue

            if sound.data.size == 0:
                result = "Model output is empty, discarding"
                Tts.clear_continuation()
            elif np.sum(np.isnan(sound.data)) > 0:
                result = "Model outputted NaN, discarding"
                Tts.clear_continuation()
            else:
                # Save "raw" output
                if save_debug_files:
                    GenerateUtil.save_debug_sound(project, indices[i], "raw", sound, is_realtime=is_realtime)

                # Trim silence ends and peak-normalize
                sound, start_trim_time, end_trim_time, original_duration = SoundPipeline.apply_generate_post_processing_with_info(sound)
                token_noise_trim_time = None

                # Trim model-specific short token-like trailing artifacts.
                # Done before STT transcription so timestamps match the final audio.
                if sound.data.size > 0 and Tts.get_class().should_trim_trailing_token_noise(project, Tts.get_instance_if_exists()):
                    pre_token_noise_trim_duration = sound.duration
                    if save_debug_files:
                        GenerateUtil.save_debug_sound(project, indices[i], "pre_token_noise_trim", sound, is_realtime=is_realtime)
                    trimmed_sound = SoundExtraUtil.trim_trailing_token_noise(sound)
                    if len(trimmed_sound.data) != len(sound.data):
                        sound = trimmed_sound
                        token_noise_trim_time = pre_token_noise_trim_duration - sound.duration
                        if save_debug_files:
                            GenerateUtil.save_debug_sound(project, indices[i], "post_token_noise_trim", sound, is_realtime=is_realtime)

                # Limit internal silence gaps (done before STT transcription so timestamps match)
                if sound.data.size > 0 and project.limit_silence_gaps:
                    # Save debug sound before gap limiting
                    if save_debug_files:
                        GenerateUtil.save_debug_sound(project, indices[i], "pre_gap_limit", sound, is_realtime=is_realtime)
                    sound, gap_trims = SilenceUtil.limit_silence_gaps(sound, project.limit_silence_gaps_duration)
                    # Save debug sound after gap limiting
                    if save_debug_files:
                        GenerateUtil.save_debug_sound(project, indices[i], "post_gap_limit", sound, is_realtime=is_realtime)
                else:
                    gap_trims = []

                if sound.data.size == 0:
                    result = "Model output is silence, discarding"
                    Tts.clear_continuation()
                else:
                    result = (sound, gap_trims, start_trim_time, end_trim_time, original_duration, token_noise_trim_time, voice_tag)

            results.append(result)
            Tts.clear_continuation_if_reason(phrase_groups[indices[i]].last_reason)

        return results

    @staticmethod
    def get_error_result_message(result: ValidationResult | str | TtsModelError) -> str:
        if isinstance(result, TtsModelError):
            return result.message
        if isinstance(result, str):
            return result
        return ""

    @staticmethod
    def is_error_result_oom(result: ValidationResult | str | TtsModelError) -> bool:
        return is_oom_error_message(GenerateUtil.get_error_result_message(result))

    @staticmethod
    def print_consecutive_model_errors_message(max_consecutive_model_errors: int) -> None:
        printt(
            f"{COL_ERROR}Stopping generation: "
            f"Too many consecutive TTS model errors ({max_consecutive_model_errors})."
        )
        printt()

    @staticmethod
    def phrase_group_to_prompt(phrase_group: PhraseGroup, project: Project) -> str:

        prompt = phrase_group.as_flattened_phrase().text
        return prompt

    @staticmethod
    def save_sound_and_timing_json(
        state: State,
        phrase_group: PhraseGroup,
        index: int,
        validation_result: ValidationResult,
        is_real_time: bool,
        voice_tag: str = "",
        stt_info: SegmentTranscriptData | None = None
    ) -> tuple[str, str]:
        """
        Saves sound segment and timing info json
        Returns error string (if any), saved file path
        """

        project = state.project

        # Save sound
        dir_path = project.realtime_path if is_real_time else project.sound_segments_path
        os.makedirs(dir_path, exist_ok=True)

        file_name = SoundSegmentUtil.make_file_name(
            index=index,
            phrase_group=phrase_group,
            project=project,
            validation_result=validation_result,
            tts_model_type=Tts.get_type().value,
            is_real_time=is_real_time,
            voice_tag=voice_tag,
        )
        sound_path = os.path.join(dir_path, file_name)

        err = SoundFileUtil.save_flac(validation_result.sound, sound_path)
        if err:
            return err, sound_path

        if not is_real_time and isinstance(validation_result, TranscriptResult):
            # Make STT/timing sidecar and save to 'parallel' json file
            info = stt_info or SegmentTranscriptUtil.from_validation_result(
                project=project,
                phrase_group=phrase_group,
                index=index,
                validation_result=validation_result
            )
            json_path = get_segment_stt_info_path(sound_path)
            err = SegmentTranscriptUtil.save(json_path, info)
            if err:
                printt(COL_ERROR + str(json_path))
                printt(COL_ERROR + err)

        return "", sound_path

    @staticmethod
    def print_batch_heading(
        indices: list[int],
        voice_index: int | None = None,
        show_divider: bool = True,
    ) -> None:

        line_noun = make_noun("line", "lines", len(indices))
        if len(indices) <= 4:
            indices_string = ", ".join(str(item + 1) for item in indices)
        else:
            index_strings = [str(item + 1) for item in indices[:3]]
            num_more = len(indices) - 3
            indices_string = f"{', '.join(index_strings)}, + {num_more} more"
        processing_string = f"{COL_ACCENT}Processing {line_noun} {indices_string}"
        if voice_index is not None:
            processing_string += f" {COL_DIM}[voice {voice_index + 1}]{COL_DEFAULT}"

        if show_divider:
            printt(f"{COL_ACCENT}{'-' * (len(text_util.strip_ansi_codes(processing_string)))}")
        printt(f"{processing_string}")
        printt()

    @staticmethod
    def save_debug_sound(project: Project, index: int, label: str, sound: Sound, is_realtime: bool):

        dir_path = project.realtime_path if is_realtime else project.sound_segments_path
        os.makedirs(dir_path, exist_ok=True)
        index_string = str(index + 1).zfill(5)
        timestamp = SoundSegmentUtil.make_timestamp_string()

        # Use same start of filename as sound segment files
        if is_realtime:
            file_name = f"[{timestamp}] [{index_string}] [{label}].flac"
        else:
            file_name = f"[{index_string}] [{timestamp}] [{label}].flac"
        path = os.path.join(dir_path, file_name)
        _ = SoundFileUtil.save_flac(sound, path)

    @staticmethod
    def do_quick_generate(state: State, phrase_index: int) -> None:
        """Regenerate exactly one project item."""
        from tts_audiobook_tool.textual.generation_app import run_generation_app
        run_generation_app(
            state=state,
            indices={phrase_index},
            batch_size=ProjectVoiceUtil.get_batch_size(state.project),
            is_regen=True,
        )

    @staticmethod
    def do_generate_using_project_and_state(state: State) -> None:
        """ Generate queued project items and run configured completion actions. """

        indices = ProjectUtil.get_selected_indices_not_generated(state.project)
        if not indices:
            return

        # Show pre-inference hint/warning if necessary
        should_continue = app_hint_util.show_pre_inference_hints(state.prefs, state.project)
        if not should_continue:
            return

        from tts_audiobook_tool.textual.generation_app import run_generation_app
        generation_result = run_generation_app(
            state=state,
            indices=indices,
            batch_size=ProjectVoiceUtil.get_batch_size(state.project),
            is_regen=False,
        )
        if not generation_result.completed:
            return

        app_support.play_done_sound()

        if state.project.gen_auto_concat:
            printt()
            ConcatUtil.auto_concat_after_generation(state)

# ---

def print_speed_info(gen_elapsed: float, gen_results: list) -> None:

    # Elapsed
    message = f"Generated audio in {gen_elapsed:.1f}s"

    # Cumulative duration, speed
    num_sounds = 0
    cum_duration = 0.0
    for item in gen_results:
        if isinstance(item, tuple) and isinstance(item[0], Sound):
            sound = item[0]
            num_sounds += 1
            cum_duration += sound.duration
        elif isinstance(item, Sound):
            num_sounds += 1
            cum_duration += item.duration
    if num_sounds == len(gen_results):
        message += f"; sound duration: {cum_duration:.1f}s; "
        gen_elapsed = max(gen_elapsed, 0.1)
        raw_speed = cum_duration / gen_elapsed
        speed = min(raw_speed, 99.9)
        speed_str = f"{speed:.1f}"
        if speed == 99.9:
            speed_str += "+"
        speed_str += "x"
        message += f"speed: {speed_str}"
        # Publish structured stats only when every result produced audio,
        # mirroring the console message; a partial batch has no meaningful
        # realtime factor (failed results contribute no audio duration).
        realtime_factor = gen_elapsed / cum_duration if cum_duration > 0 else 0.0
        GenerationEvents.emit(
            GenerationStats(
                generation_seconds=gen_elapsed,
                audio_seconds=cum_duration,
                realtime_factor=realtime_factor,
                speed_factor=raw_speed,
            )
        )

    print(message)

def make_batch_eta_seconds(
        batch_durations: list[float],
        remaining_batches: int) -> float | None:
    """
    Extrapolates the estimated remaining seconds of a generation run from
    the rolling per-batch duration history. Returns None until at least
    ETA_MIN_BATCH_DURATIONS samples exist; the history itself holds at
    most BATCH_DURATION_HISTORY_SIZE entries.
    """
    if len(batch_durations) < ETA_MIN_BATCH_DURATIONS:
        return None
    average = sum(batch_durations) / len(batch_durations)
    return average * remaining_batches

def print_eta(saved_elapsed: list[float], num_left) -> None:

    MIN_SAMPLES = 5
    if len(saved_elapsed) < MIN_SAMPLES or num_left <= 0:
        return

    MAX_SAMPLES = 50
    i_start = len(saved_elapsed) - MAX_SAMPLES
    i_start = max(i_start, 0)
    i_end = len(saved_elapsed)
    sum = 0
    for i in range(i_start, i_end):
        elapsed = saved_elapsed[i]
        sum += elapsed
    num_samples = i_end - i_start
    avg = sum / num_samples
    eta = num_left * avg
    printt(f"Est. time remaining: {duration_string(eta)}")

def bucket_items(
        items: list[tuple[int, int]],
        phrase_groups: list[PhraseGroup],
        batch_size: int) -> list[tuple[int, int]]:
    """
    Returns a copy of `items`, sorted by phrase group word count in an alternating
    descending/ascending pattern. Reason for sorting in groups is to preserve the
    ordering of items to some extent. Does not "bucket" per-se.

    Params:
        items: list of tuples (phase group index, retry_count),
               which should be sorted by index
    """

    # Make copy of items, but with added element for word count
    triplets: list[tuple[int, int, int]] = []

    for index, retry_count in items:
        phrase_group = phrase_groups[index]
        num_words = phrase_group.num_words
        triplet = (index, retry_count, num_words)
        triplets.append(triplet)

    group_size = batch_size * BATCH_ITERATIONS_PER_GROUP
    is_descending = True

    # Create a copy of triplets which is sorted in groups of size group_size,
    # alternating between descending and ascending.
    grouped_triplets = []
    for i in range(0, len(triplets), group_size):
        group = triplets[i:i + group_size]
        group.sort(key=lambda x: x[2], reverse=is_descending)
        grouped_triplets.extend(group)
        is_descending = not is_descending

    # "Extract" the index and retry count elements
    sorted_items = []
    for index, retry_count, _ in grouped_triplets:
        sorted_items.append( (index, retry_count) )
    return sorted_items

def effective_voice_indices(
        phrase_groups: list[PhraseGroup],
        indices: list[int],
        num_voice_values: int,
) -> dict[int, int]:
    """
    Maps each item index to the voice sample index that `GenerateUtil.generate`
    will actually use for that line, given a user-defined voice selection mode.

    A per-line selection of -1 (no explicit assignment) or any stale
    out-of-range value is clamped to the nearest valid voice sample index.
    """
    return {
        index: min(max(phrase_groups[index].voice_index, 0), num_voice_values - 1)
        for index in indices
    }

def make_single_voice_rounds(
        items: list[tuple[int, int]],
        batch_size: int,
) -> list[list[SubBatch]]:
    """
    Wraps a flat (possibly word-count-bucketed) queue into a list of rounds,
    each holding a single voice-less sub-batch of up to `batch_size` items.
    The voice is resolved by the existing logic inside
    `GenerateUtil.generate()` (auto-advance rotation or disabled mode).
    """
    return [
        [SubBatch(voice_selection_index=None, items=tuple(items[i:i + batch_size]))]
        for i in range(0, len(items), batch_size)
    ]

def make_multi_voice_rounds(
        items: list[tuple[int, int]],
        phrase_groups: list[PhraseGroup],
        voice_of_index: dict[int, int],
        batch_size: int,
        sort_by_words: bool,
) -> list[list[SubBatch]]:
    """
    Builds a queue of rounds for a project with user-defined voice selections.
    Each round is a group of per-voice sub-batches ("batches of batches");
    each sub-batch holds at most `batch_size` items that all share one voice
    sample, so each can be sent to the model as a single call.

    The single-speaker windowing principle is applied within practical limits:
    items are walked in index-sorted windows of
    `batch_size * BATCH_ITERATIONS_PER_GROUP * num_voices` items, so each
    voice keeps a comparable per-window share as the single-speaker path.
    Within a window, each voice's items are sorted by word count (alternating
    descending/ascending per window) to pack similar-duration sub-batches
    together, keeping the overall order roughly in book order.

    Params:
        items: (phase group index, retry_count) tuples, sorted by index
        voice_of_index: map from index to voice sample index (see
            effective_voice_indices)
        sort_by_words: False for CJK projects (no word-count sorting)
    """
    if not items:
        return []

    distinct_voices = {voice_of_index[index] for index, _ in items}
    window_size = batch_size * BATCH_ITERATIONS_PER_GROUP * len(distinct_voices)

    rounds: list[list[SubBatch]] = []
    is_descending = True

    for i in range(0, len(items), window_size):
        window = items[i:i + window_size]

        # Group the window's items by voice, in order of first appearance so
        # that a round's sub-batches stay roughly in book order.
        by_voice: dict[int, list[tuple[int, int]]] = {}
        voice_order: list[int] = []
        for item in window:
            voice = voice_of_index[item[0]]
            if voice not in by_voice:
                by_voice[voice] = []
                voice_order.append(voice)
            by_voice[voice].append(item)

        sub_batches: list[SubBatch] = []
        for voice in voice_order:
            group = by_voice[voice]
            if sort_by_words:
                group.sort(key=lambda item: phrase_groups[item[0]].num_words, reverse=is_descending)
            for j in range(0, len(group), batch_size):
                sub_batches.append(SubBatch(
                    voice_selection_index=voice,
                    items=tuple(group[j:j + batch_size]),
                ))
        rounds.append(sub_batches)
        is_descending = not is_descending

    return rounds

def make_retry_round(
        items: list[tuple[int, int]],
        voice_of_index: dict[int, int] | None,
        batch_size: int,
) -> list[SubBatch]:
    """
    Builds a single round from pending retries (re-added items). Items keep
    the order they were produced in and are grouped by voice sample (when the
    project uses user-defined voice selections); no word-count re-bucketing is
    done, matching the single-speaker behavior of processing re-adds promptly.
    """
    if voice_of_index is None:
        groups: list[tuple[int | None, list[tuple[int, int]]]] = [(None, list(items))]
    else:
        by_voice: dict[int, list[tuple[int, int]]] = {}
        for item in items:
            by_voice.setdefault(voice_of_index[item[0]], []).append(item)
        groups = [(voice, by_voice[voice]) for voice in sorted(by_voice)]

    sub_batches: list[SubBatch] = []
    for voice, group in groups:
        for i in range(0, len(group), batch_size):
            sub_batches.append(SubBatch(
                voice_selection_index=voice,
                items=tuple(group[i:i + batch_size]),
            ))
    return sub_batches
