from __future__ import annotations
import os
import sys
import random
import time

import numpy as np

from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.force_align_util import ForceAlignUtil
from tts_audiobook_tool.memory_util import MemoryUtil
from tts_audiobook_tool.models_util import ModelsUtil
from tts_audiobook_tool.music_detector import MusicDetector
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.prompt_normalizer import PromptNormalizer
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import ChatterboxType, HiggsModelProtocol, MiraProtocol, VibeVoiceProtocol
from tts_audiobook_tool.tts_model import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.validation_result import MusicFailResult, SkippedResult, TranscriptResult, TrimmedResult, ValidationResult, WordErrorResult
from tts_audiobook_tool.whisper_util import WhisperUtil

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
            When set to 1, batch mode is either explicitly or effectively disabled

        Returns:
            True if ended because interrupted.
        """

        project = state.project
        max_retries = project.max_retries
        stt_variant = state.prefs.stt_variant
        stt_config = state.prefs.stt_config
        showed_vram_warning = False

        did_cancel = ModelsUtil.warm_up_models(state)
        
        # Post-init checks
        if did_cancel:
            print_feedback("\nCancelled")
            return True
                
        if Tts.get_type() == TtsModelInfos.QWEN3TTS:

            err = Tts.get_qwen3().get_post_init_error(project)
            if err:
                print_feedback(err, is_error=True)
                return True
            
            warning = Tts.get_qwen3().get_post_init_warning(project)
            if warning:
                printt(warning)
                printt()

        showed_vram_warning = MemoryUtil.show_vram_memory_warning_if_necessary()

        # Make 'items' (tuple = phase group index, retry_count)
        sorted_indices = sorted(list(indices_set))
        items: list[tuple[int, int]] = []
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]
            items.append((index, 0))

        should_bucket = (batch_size > 1 and not state.project.is_language_cjk)
        if should_bucket:
            items = bucket_items(items, state.project.phrase_groups, batch_size)

        # Metrics-related variables
        num_saved = 0
        num_failed = 0
        num_failed_music = 0
        num_errored = 0
        word_error_counts: dict[int, int] = {} 
        word_counts: dict[int, int] = {}
        start_time = time.time()
        saved_elapsed: list[float] = [] # history of end-to-end times taken to create sound files
        saved_start_time = time.time()
        gen_val_sum_time = 0

        SigIntHandler().set("generating")
        did_interrupt = False

        # Loop through items in the batch
        while items:

            if SigIntHandler().did_interrupt:
                did_interrupt = True
                break

            if not items:
                break

            if not showed_vram_warning:
                b = MemoryUtil.show_vram_memory_warning_if_necessary()
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
                num_complete=num_saved + num_failed + num_errored,
                num_remaining=len(items) + len(batch),
                num_total=len(sorted_indices)
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

            # Process and print results # TODO: separate biz n print logic
            re_adds: list[tuple[int, int]] = []

            for i, result in enumerate(results):

                index = indices[i]
                message_lines = []

                retry_string = f" (retry #{retry_counts[i]})" if retry_counts[i] else ""
                text_string = f"{COL_DIM}{Ansi.ITALICS}{project.phrase_groups[index].presentable_text}{Ansi.RESET}"
                item_line = f"{COL_DEFAULT}Line {index + 1}{retry_string}: {text_string}"
                message_lines.append(item_line)
                
                new_retry_count = retry_counts[i] + 1

                if isinstance(result, str):
                    # Error
                    val_line = f"{COL_ERROR}Error: {result}"
                    if new_retry_count > max_retries:
                        num_errored += 1
                        val_line += f"; {COL_ERROR}max retries reached"
                    else:
                        re_adds.append((index, new_retry_count))
                        val_line += f"; {COL_DEFAULT}will retry"
                    message_lines.append(val_line)
                else:
                    validation_result = result

                    val_line = f"{validation_result.get_ui_message()}"
                    if new_retry_count > 1 and not validation_result.is_fail:
                        val_line = val_line.replace("Passed", f"{COL_OK}Passed on retry") 

                    # Failed or not
                    if validation_result.is_fail:
                        if new_retry_count > max_retries:
                            num_failed += 1
                            val_line += f"; {COL_ERROR}max retries reached, tagging as failed"
                        else:
                            re_adds.append((index, new_retry_count))
                            val_line += f"; {COL_DEFAULT}will retry"
                        if isinstance(validation_result, MusicFailResult):
                            num_failed_music += 1
                    else:
                        num_saved += 1
                    
                    val_line += Ansi.RESET
                    message_lines.append(val_line)

                    # Update word count metrics
                    word_counts[index] = project.phrase_groups[index].num_words

                    # Update error count metrics, word count metrics
                    if isinstance(validation_result, WordErrorResult):
                        fails = validation_result.num_errors
                        if index in word_error_counts:
                            fails = min(fails, word_error_counts[index])
                        word_error_counts[index] = fails

                    # Save
                    phrase_group = project.phrase_groups[index]
                    err, saved_path = GenerateUtil.save_sound_and_timing_json(
                        state, phrase_group, index, validation_result, is_real_time=False
                    )
                    if err:
                        save_line = f"{COL_ERROR}Couldn't save file: {err} {saved_path}"
                    else:
                        project.sound_segments.delete_redundants_for(index)
                        save_line = f"Saved: {Ansi.ITALICS}{COL_DIM}{Path(saved_path).name}{Ansi.RESET}"
                    message_lines.append(save_line)

                printt("\n".join(message_lines))
                
                if i < len(results) - 1:
                    printt()

            # Print memory usage
            if len(results) > 1:
                printt()
            s = f"Memory: {COL_DIM}{strip_ansi_codes(AppUtil.make_memory_string())}"
            printt(s)
            
            printt()

            if re_adds:
                # Insert "re_adds" at the head of the list (not bothering with deque fyi)
                items[:0] = re_adds

        # Print summary, metrics
        s = ""
        if did_interrupt:
            s += "Interrupted. "
        s += f"Elapsed: {duration_string(time.time() - start_time)}\n"
        ok = str(num_saved)
        if num_saved == len(sorted_indices):
            ok += " (all)"
        s += f"Num lines saved: {COL_OK}{ok}{COL_DEFAULT}\n"
        col = COL_ACCENT if num_failed else ""
        if num_failed:
            s += f"Num lines saved, but tagged as failed: {col}{num_failed}{COL_DEFAULT}\n"
        if num_errored:
            s += f"Num lines failed to generate: {COL_ERROR}{num_errored}{COL_DEFAULT}\n"
        if DEV:
            s += f"Num words: {sum(word_counts.values())}\n"
            if Stt.has_instance():
                s += f"Num word fails: {sum(word_error_counts.values())}\n"
                if MusicDetector.has_instance():
                    s += f"Num music fails: {num_failed_music}\n"
            s += f"Gen/val elapsed: {duration_string(gen_val_sum_time)}\n"
        printt(s)

        SigIntHandler().clear()
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
    ) -> list[ ValidationResult | str ]:
        """
        Generates and validates a batch of prompts
        Prints updates
        Returns list of sounds etc

        :param indices:
            When length is 1, batch mode is disabled
        """
        
        project = state.project

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
            save_debug_files=state.prefs.save_debug_files
        )        
        printt() # Restore print color, print blank line
        
        # Print speed info
        print_speed_info(time.time() - gen_start_time, gen_results)

        # Should skip or not
        skip_reason = Stt.should_skip(state, is_skip_reason_buffer)
        if not skip_reason:
            printt(f"{COL_DEFAULT}Transcribing audio...", end="") # gets overwritten

        val_start_time = time.time()
        results: list[ ValidationResult | str ] = []

        for i, gen_result in enumerate(gen_results):

            index = indices[i]

            if isinstance(gen_result, str):
                err = gen_result
                printt(f"Text segment {index+1} - error: {err}")
                results.append(err)
                continue

            sound = gen_result

            if skip_reason:
                validation_result = SkippedResult(sound=sound, message=skip_reason)
                results.append((validation_result))
                continue

            # Transcribe
            gen_result = WhisperUtil.transcribe_to_words(
                sound, project.language_code, stt_variant, stt_config
            )
            if isinstance(gen_result, str):
                err = gen_result
                results.append(err)
                continue
            transcribed_words = gen_result

            # Validate
            text = phrase_groups[ indices[i] ].as_flattened_phrase().text
            validation_result = ValidateUtil.validate(
                sound, text, transcribed_words, project.language_code, strictness=project.strictness
            )
            results.append(validation_result)

            if isinstance(validation_result, TrimmedResult):
                GenerateUtil.save_debug_sound(
                    project=project,
                    index=index,
                    label="trim",
                    sound=validation_result.sound,
                    is_realtime=is_realtime
                )

        if not skip_reason:
            message = f"{COL_DEFAULT}Transcribed audio: {(time.time() - val_start_time):.1f}s"
            printt(f"{Ansi.LINE_HOME}{message}")
        if len(indices) > 1:
            printt()

        return results

    @staticmethod
    def generate(
            project: Project,
            indices: list[int], 
            phrase_groups: list[PhraseGroup],
            force_random_seed: bool, 
            is_realtime: bool,
            save_debug_files: bool
        ) -> list[Sound | str]:
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
        if Tts.get_type() == TtsModelInfos.NONE:
            result = "No active TTS model"
        else:
            result = Tts.get_instance().generate_using_project(project, prompts, force_random_seed)

        # `result` is either n generated Sounds or a single error string
        if isinstance(result, str): 
            return [result for _ in range(len(prompts))] # return n error strings

        sounds = [result] if isinstance(result, Sound) else result
        
        results: list[Sound | str] = []

        for i, sound in enumerate(sounds):

            if sound.data.size == 0:
                result = "Model output is empty, discarding"
            elif np.sum(np.isnan(sound.data)) > 0:
                    result = "Model outputted NaN, discarding"
            else:
                # Save "raw" output
                if save_debug_files:
                    GenerateUtil.save_debug_sound(project, indices[i], "raw", sound, is_realtime=is_realtime)

                # Trim silence from ends of audio clip
                sound = SilenceUtil.trim_silence(sound)[0]

                # Re-check size
                if sound.data.size == 0:
                    result = "Model output is silence, discarding"
                else:
                    # Do peak normalization
                    normalized_data = SoundUtil.normalize(sound.data, headroom_db=3.0)
                    result = Sound(normalized_data, sound.sr)
            
            results.append(result)

        return results

    @staticmethod
    def phrase_group_to_prompt(phrase_group: PhraseGroup, project: Project) -> str:
        
        prompt = phrase_group.as_flattened_phrase().text
        prompt = PromptNormalizer.apply_prompt_word_substitutions(
            prompt, project.word_substitutions, project.language_code
        )
        prompt = PromptNormalizer.normalize_prompt(
            text=prompt, 
            language_code=project.language_code,
            un_all_caps=Tts.get_type().value.un_all_caps
        )
        prompt = Tts.get_instance().massage_for_inference(prompt)
        return prompt

    @staticmethod
    def save_sound_and_timing_json(
        state: State,
        phrase_group: PhraseGroup,
        index: int,
        validation_result: ValidationResult,
        is_real_time: bool
    ) -> tuple[str, str]:
        """
        Saves sound segment and timing info json
        Returns error string (if any), saved file path
        """

        project = state.project
        
        # Save sound
        file_name = SoundSegmentUtil.make_file_name(
            index=index,
            phrase_group=phrase_group,
            project=project,
            validation_result=validation_result,
            tts_model_info=Tts.get_type().value,
            is_real_time=is_real_time
        )
        
        dir_path = project.realtime_path if is_real_time else project.sound_segments_path 
        os.makedirs(dir_path, exist_ok=True)

        sound_path = os.path.join(dir_path, file_name)
        err = SoundFileUtil.save_flac(validation_result.sound, sound_path)
        if err:
            return err, sound_path
        
        if not is_real_time and isinstance(validation_result, TranscriptResult): 
            # Make timing metadata and save to 'parallel' json file        
            phrase_group = project.phrase_groups[index]
            timed_phrases = ForceAlignUtil.make_timed_phrases(
                phrases=phrase_group.phrases,
                words=validation_result.transcript_words,
                sound_duration=validation_result.sound.duration
            )
            if isinstance(validation_result, TrimmedResult):
                # TODO verify 
                timed_phrases = adjust_timed_phrases_trimmed(timed_phrases, validation_result)
            json_path = Path(sound_path).with_suffix(".json")
            dicts = TimedPhrase.timed_phrases_to_dicts(timed_phrases)
            json_string = json.dumps(dicts)
            try:
                with open(json_path, 'w', encoding='utf-8') as file:
                    file.write(json_string)
            except Exception as e:
                ... # eat silently

        if state.prefs.save_debug_files and not is_real_time:
            save_debug_json(project, phrase_group, index, validation_result, dir_path)

        return "", sound_path

    @staticmethod
    def print_batch_heading(
        indices: list[int], num_complete: int, num_remaining: int, num_total: int
    ) -> None:
        
        line_noun = make_noun("line", "lines", len(indices))
        index_strings = [str(item + 1) for item in indices[:3]]
        if len(indices) > 3:
            index_strings.append("...")
        indices_string = ", ".join(index_strings)
        processing_string = f"{COL_ACCENT}Processing {line_noun} {indices_string}"

        counts = f"{COL_DIM}(lines processed: {COL_DEFAULT}{num_complete}{COL_DIM}; remaining: {COL_DEFAULT}{num_remaining}{COL_DIM})"

        message = f"{processing_string} {counts}"
        
        printt(f"{COL_ACCENT}{'-' * (len(strip_ansi_codes(message)))}")
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
    message = f"Generated audio: {gen_elapsed:.1f}s"

    # Cumulative duration, speed
    num_sounds = 0
    cum_duration = 0.0
    for item in gen_results:
        if isinstance(item, Sound):
            num_sounds += 1
            cum_duration += item.duration
    if num_sounds == len(gen_results):
        message += f"; sound duration: {cum_duration:.1f}; "
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

def adjust_timed_phrases_trimmed(timed_phrases: list[TimedPhrase], trimmed_result: TrimmedResult) -> list[TimedPhrase]:
    
    start_offset = trimmed_result.start_time or 0
    end_time = trimmed_result.end_time or trimmed_result.sound.duration
    full_duration = end_time - start_offset
    
    results = []
    
    for item in timed_phrases:
        time_start = max(item.time_start - start_offset, 0)
        time_end = min(item.time_end, full_duration)
        new_item = TimedPhrase(text=item.text, time_start=time_start, time_end=time_end)
        results.append(new_item)

    return results

def save_debug_json(
        project: Project,
        phrase_group: PhraseGroup,
        index: int,
        validation_result: ValidationResult,
        dir_path: str
) -> None:

    if not isinstance(validation_result, TranscriptResult): 
        return

    d = {}
    prompt = GenerateUtil.phrase_group_to_prompt(phrase_group, project)
    source = phrase_group.as_flattened_phrase().text
    transcript = WhisperUtil.get_flat_text_from_words(validation_result.transcript_words)
    normalized_source, normalized_transcript = \
        TextNormalizer.normalize_source_and_transcript(source, transcript, project.language_code)
    d["language_code"] = project.language_code
    d["index_1b"] = index + 1
    d["prompt"] = prompt
    d["source"] = source
    d["transc"] = transcript
    d["normalized_source"] = normalized_source
    d["normalized_transc"] = normalized_transcript
    d["result_class"] = type(validation_result).__name__
    d["result_desc"] = strip_ansi_codes( validation_result.get_ui_message() )
    if isinstance(validation_result, WordErrorResult):
        d["result_num_errors"] = validation_result.num_errors
        d["result_threshold"] = validation_result.threshold
    d["transcript_words"] = WhisperUtil.words_to_json(validation_result.transcript_words)
    file_name = SoundSegmentUtil.make_file_name(
        index=index,
        phrase_group=phrase_group,
        project=project,
        validation_result=validation_result,
        tts_model_info=Tts.get_type().value,
        is_real_time=False,
        is_debug_json=True
    )
    json_path = os.path.join(dir_path, file_name)
    json_string = json.dumps(d, indent=4)
    try:
        with open(json_path, 'w', encoding='utf-8') as file:
            file.write(json_string)
    except Exception as e:
        ... # eat silently

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
