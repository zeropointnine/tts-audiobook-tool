from __future__ import annotations
from dataclasses import dataclass
import os
import time

import numpy as np

from tts_audiobook_tool import text_util
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.model_manager import ModelManager
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
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.validator import Validator
from tts_audiobook_tool.app_types.validation_result import MusicFailResult, SkippedResult, TranscriptResult, TrimmedResult, ValidationResult, WordErrorResult
from tts_audiobook_tool.sound.silence_util import SilenceGapTrim
from tts_audiobook_tool.transcriber import Transcriber


@dataclass(frozen=True)
class TtsModelError:
    message: str

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
            or aborted (OOM detected or too many consecutive model errors)
        """

        project = state.project
        max_retries = project.max_retries
        stt_variant = state.prefs.stt_variant
        stt_config = state.prefs.stt_config
        showed_vram_warning = False

        warm_up_result = ModelManager.warm_up_models(state)
        if warm_up_result.should_stop:
            app_support.print_warm_up_result_stop(warm_up_result)
            if warm_up_result.error:
                app_memory.gc_ram_vram()
            return True

        # Do model readiness check now that model instance exists
        err = readiness.get_generate_blocker_text(state, verbose=True)
        if err:
            print_feedback(err, is_error=True)
            return True

        # Print warnings if any
        warnings = Tts.get_instance().get_warning_issues(state.project)
        if warnings:
            warnings_string = "\n".join(warnings)
            print_feedback(Ansi.ITALICS + warnings_string, no_preformat=True)

        showed_vram_warning = app_memory.show_vram_memory_warning_if_necessary()

        # Make 'items' (tuple = phase group index, retry_count)
        sorted_indices = sorted(list(indices_set))
        items: list[tuple[int, int]] = []
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]
            items.append((index, 0))

        should_bucket = (batch_size > 1 and not ProjectVoiceUtil.is_language_cjk(state.project))
        if should_bucket:
            items = bucket_items(items, state.project.phrase_groups, batch_size)

        # Metrics-related variables
        num_errored = 0
        num_failed = 0
        num_failed_music = 0
        num_improved = 0
        num_passed = 0
        word_counts: dict[int, int] = {}
        preexisting_word_error_counts = project.sound_segments.get_word_error_counts_in_generate_range()
        best_word_error_counts = dict(preexisting_word_error_counts)
        gen_val_sum_time = 0
        start_time = time.time()
        consecutive_model_errors = 0
        max_consecutive_model_errors = 5

        Interrupts().set("generating")
        did_interrupt = False
        did_abort_model_errors = False

        # Audiobook generation is a top-level generation run. Start from a
        # fresh rolling-continuation context, independent of any prior run.
        Tts.clear_continuation()

        # Loop through items in the batch
        while items:

            if Interrupts().did_interrupt:
                did_interrupt = True
                break

            if not items:
                break

            if not showed_vram_warning:
                b = app_memory.show_vram_memory_warning_if_necessary()
                if b:
                    print("\a", end="")
                    showed_vram_warning = True

            # Take batch from items
            batch = items[:batch_size]
            items = items[batch_size:]

            # Make parallel arrays from 'batch'
            indices = [item[0] for item in batch]
            retry_counts = [item[1] for item in batch]

            # Print item info
            GenerateUtil.print_batch_heading(
                indices=indices,
                num_complete=num_passed + num_failed + num_errored,
                num_remaining=len(items) + len(batch),
                num_total=len(sorted_indices),
                start_time=start_time
            )

            # Generate and validate
            gen_start_time = time.time()
            results = GenerateUtil.generate_and_validate_batch(
                state=state, 
                indices=indices, 
                phrase_groups=project.phrase_groups,
                stt_variant=stt_variant, stt_config=stt_config,
                force_random_seed=is_regen or any(count > 0 for count in retry_counts),
                is_realtime=False
            )
            gen_val_sum_time += (time.time() - gen_start_time)

            # Check for OOM in results and break early if detected
            if any(GenerateUtil.is_error_result_oom(r) for r in results):
                printt()
                first_oom = next(GenerateUtil.get_error_result_message(r) for r in results if GenerateUtil.is_error_result_oom(r))
                print_gen_oom_message(first_oom)
                Tts.clear_continuation()
                did_interrupt = True
                break

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
                        val_line += f"; {COL_ERROR}too many TTS model errors in a row"
                    elif new_retry_count > max_retries:
                        num_errored += 1
                        val_line += f"; {COL_ERROR}max retries reached"
                    else:
                        re_adds.append((index, new_retry_count))
                        val_line += f"; {COL_DEFAULT}will retry"
                    message_lines.append(val_line)
                 
                elif validation_result:
                    consecutive_model_errors = 0

                    val_line = f"{validation_result.get_ui_message_with_post_processing()}"
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
                            isinstance(validation_result, MusicFailResult)
                            or (isinstance(validation_result, WordErrorResult) and validation_result.num_errors > 0)
                        )

                    val_line += Ansi.RESET
                    message_lines.append(val_line)

                    # Update word count metrics
                    word_counts[index] = phrase_group.num_words

                    # Save
                    err, saved_path = GenerateUtil.save_sound_and_timing_json(
                        state, phrase_group, index, validation_result, is_real_time=False, stt_info=stt_info
                    )
                    if err:
                        save_line = f"{COL_ERROR}Couldn't save file: {err} {saved_path}"
                    else:
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
                        SegmentTranscriptUtil.print_stt_details(
                            stt_info,
                            should_show_diff=should_show_viz
                        )
                    else:
                        # Just print source text, in a similar style
                        printt(f"{COL_DEFAULT}Source text: {COL_DIM_ITALICS}{phrase_group.presentable_text.strip()}")
                
                if i < len(results) - 1:
                    printt()

                if did_abort_model_errors:
                    break

            if did_abort_model_errors:
                printt()
                GenerateUtil.print_consecutive_model_errors_message(max_consecutive_model_errors)
                Tts.clear_continuation()
                did_interrupt = True
                break

            # Print current memory usage
            printt()
            s = f"Memory: {COL_DIM}{text_util.strip_ansi_codes(app_support.make_memory_string())}"
            printt(s)
            printt()

            if re_adds:
                # Insert "re_adds" at the head of the list (not bothering with deque fyi)
                items[:0] = re_adds

        # Print summary, metrics
        warnings_string = ""
        if did_interrupt:
            Tts.clear_continuation()
            warnings_string += "Interrupted. "
        warnings_string += f"Elapsed: {duration_string(time.time() - start_time)}\n"
        ok = str(num_passed)
        if num_passed == len(sorted_indices):
            ok += " (all)"
        warnings_string += f"Lines saved: {COL_OK}{ok}{COL_DEFAULT}\n"
        if num_improved:
            warnings_string += f"Lines improved on retry: {COL_OK}{num_improved}{COL_DEFAULT}\n"
        col = COL_ACCENT if num_failed else ""
        if num_failed:
            warnings_string += f"Lines saved, but with excess word errors: {col}{num_failed}{COL_DEFAULT}\n"
        if num_errored:
            warnings_string += f"Lines failed to generate: {COL_ERROR}{num_errored}{COL_DEFAULT}\n"
        if DEV:
            warnings_string += f"Num words: {sum(word_counts.values())}\n"
            if Stt.has_instance():
                if ModelManager.has_yamnet_detector():
                    warnings_string += f"Lines with music fails: {num_failed_music}\n"
            warnings_string += f"Gen/val elapsed: {duration_string(gen_val_sum_time)}\n"
        printt(warnings_string)

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
        is_skip_reason_buffer: bool=False
    ) -> list[ValidationResult | str | TtsModelError]:
        """
        Generates and validates a batch of prompts from the Project text.
        Prints updates.
        Returns a list of results (ValidationResult or error string).

        :param indices:
            When length is 1, batch mode is disabled
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
            print_generation_request=True
        )        
        printt() # Restore print color, print blank line
        
        # Print speed info
        print_speed_info(time.time() - gen_start_time, gen_results)

        # Should skip or not
        skip_reason = Stt.should_skip(state, is_skip_reason_buffer)
        if not skip_reason:
            printt(f"{COL_DEFAULT}Transcribing audio...", end="") # gets overwritten

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

            sound, gap_trims, start_trim_time, end_trim_time, original_duration, token_noise_trim_time = gen_result

            if skip_reason:
                validation_result = SkippedResult(sound=sound, message=skip_reason)
                validation_result.intra_sample_silence_trims = gap_trims
                validation_result.generated_start_trim_time = start_trim_time
                validation_result.generated_end_trim_time = end_trim_time
                validation_result.generated_trim_original_duration = original_duration
                validation_result.trailing_token_noise_trim_time = token_noise_trim_time
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
            print_generation_request: bool = False
        ) -> list[tuple[Sound, list[SilenceGapTrim], float | None, float | None, float, float | None] | str | TtsModelError]:
        """
        Core audio generation function.
        
        Returns a list of "results" (same length as that of prompt_text), 
        where each result is either a generated Sound or an error string.
        The Sound is trimmed for silence and peak-normalized.

        param indices:
            Use a one-element list to not generate in batch mode
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
        if Tts.get_type() == TtsModelType.NONE:
            result = "No active TTS model"
        else:
            result = Tts.generate_using_project(
                project,
                prompts,
                force_random_seed,
                print_generation_request=print_generation_request
            )

        # `result` is either n generated Sounds or a single error string
        if isinstance(result, str): 
            Tts.clear_continuation()
            return [TtsModelError(result) for _ in range(len(prompts))] # return n model errors

        sounds = [result] if isinstance(result, Sound) else result
        
        results: list[tuple[Sound, list[SilenceGapTrim], float | None, float | None, float, float | None] | str | TtsModelError] = []

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
                    result = (sound, gap_trims, start_trim_time, end_trim_time, original_duration, token_noise_trim_time)
             
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
            f"{max_consecutive_model_errors} TTS model errors occurred in a row."
        )
        printt(f"{COL_ERROR}Check the TTS model/server before retrying.")
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
            is_real_time=is_real_time
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
        indices: list[int], num_complete: int, num_remaining: int, num_total: int, start_time: float
    ) -> None:
        
        line_noun = make_noun("line", "lines", len(indices))
        if len(indices) <= 4:
            indices_string = ", ".join(str(item + 1) for item in indices)
        else:
            index_strings = [str(item + 1) for item in indices[:3]]
            num_more = len(indices) - 3
            indices_string = f"{', '.join(index_strings)}, + {num_more} more"
        processing_string = f"{COL_ACCENT}Processing {line_noun} {indices_string}"

        elapsed = duration_string(time.time() - start_time)
        counts = f"{COL_DIM}(lines processed: {COL_DEFAULT}{num_complete}{COL_DIM}; remaining: {COL_DEFAULT}{num_remaining}{COL_DIM}; elapsed: {COL_DEFAULT}{elapsed}{COL_DIM})"

        message = f"{processing_string} {counts}"
        
        printt(f"{COL_ACCENT}{'-' * (len(text_util.strip_ansi_codes(message)))}")
        printt(f"{message}")
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
        speed = cum_duration / gen_elapsed
        speed = min(speed, 99.9)
        speed_str = f"{speed:.1f}"
        if speed == 99.9:
            speed_str += "+"
        speed_str += "x"
        message += f"speed: {speed_str}"
    
    print(message)
    
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

    BATCH_ITERATIONS_PER_GROUP = 5
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
