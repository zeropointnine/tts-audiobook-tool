from __future__ import annotations
from enum import Enum, auto
import os
import sys
import random
import time

import numpy as np

from tts_audiobook_tool.app_types import SkippedResult, Sound, SttConfig, SttVariant, TranscriptResult
from tts_audiobook_tool.app_types import FailResult, TrimmableResult, PassResult, ValidationResult
from tts_audiobook_tool.app_util import AppUtil
from tts_audiobook_tool.force_align_util import ForceAlignUtil
from tts_audiobook_tool.gen_info import GenInfo
from tts_audiobook_tool.phrase import PhraseGroup
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sig_int_handler import SigIntHandler
from tts_audiobook_tool.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.text_normalizer import TextNormalizer
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import HiggsModelProtocol, MiraProtocol, VibeVoiceProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.whisper_util import WhisperUtil

class GenerateUtil:

    @staticmethod
    def generate_files(
            project: Project, 
            indices_set: set[int],
            stt_variant: SttVariant,
            stt_config: SttConfig,
            max_retries: int,
            batch_size: int,
            is_regen: bool
    ) -> bool:
        """
        Subroutine for doing a series of audio generations to files
        Prints feedback at end of each item, and summary at end of loop.
        Returns True if ended because interrupted.

        :param batch_size:
            When set to 1, batch mode is disabled
        """

        force_no_stt = ValidateUtil.is_unsupported_language_code(project.language_code)
        did_cancel = Tts.warm_up_models(force_no_stt)
        if did_cancel:
            print_feedback("\nCancelled")
            return True

        # Make 'items' (tuple = phase group index, retry_count)
        sorted_indices = sorted(list(indices_set))
        items: list[tuple[int, int]] = []
        for i in range(len(sorted_indices)):
            index = sorted_indices[i]
            items.append((index, 0))

        num_saved = 0
        num_failed = 0
        num_errored = 0

        start_time = time.time()
        saved_elapsed: list[float] = [] # history of end-to-end times taken to create sound files
        saved_start_time = time.time()

        SigIntHandler().set("generating")
        did_interrupt = False

        # Loop
        while items:

            if SigIntHandler().did_interrupt:
                did_interrupt = True
                break

            if not items:
                break

            # Take batch from items
            batch = items[:batch_size]
            items = items[batch_size:]

            # Make parallel arrays from 'batch'
            indices = [item[0] for item in batch]
            retry_counts = [item[1] for item in batch]

            # Print item info
            GenerateUtil.print_item_count_heading(
                num_complete=num_saved + num_failed + num_errored,
                num_remaining=len(items) + len(batch),
                num_total=len(sorted_indices)
            )
            for index, retry_count in batch:
                s = f"Item {index+1}" + (f" (retry #{retry_count}): " if retry_count else ": ")
                s += f"{COL_DIM}{Ansi.ITALICS}{project.phrase_groups[index].presentable_text}"
                printt(s)
            printt()

            # Generate and validate
            results = GenerateUtil.generate_and_validate(
                project=project, 
                indices=indices, 
                phrase_groups=project.phrase_groups,
                stt_variant=stt_variant, stt_config=stt_config,
                force_random_seed=is_regen or any(count > 0 for count in retry_counts),
                is_realtime=False
            )

            # Process and print results
            re_adds: list[tuple[int, int]] = []

            for i, result in enumerate(results):

                index = indices[i]
                new_retry_count = retry_counts[i] + 1
                message = f"{COL_DEFAULT}Item {index + 1}: "

                if isinstance(result, str):
                    message += f"{COL_ERROR}Error: {result}"
                    if new_retry_count > max_retries:
                        num_errored += 1
                        message += f"; {COL_ERROR}max retries reached"
                    else:
                        re_adds.append((index, new_retry_count))
                        message += f"; {COL_DEFAULT}will retry"
                else:
                    sound, validation_result = result

                    message += f"{validation_result.get_ui_message()}" # ui_message includes color codes

                    if isinstance(validation_result, FailResult):
                        if new_retry_count > max_retries:
                            num_failed += 1
                            message += f"; {COL_ERROR}max retries reached, tagging as failed"
                        else:
                            re_adds.append((index, new_retry_count))
                            message += f"; {COL_DEFAULT}will retry"
                    else:
                        num_saved += 1

                    # Save
                    phrase_group = project.phrase_groups[index]
                    err, saved_path = GenerateUtil.save_gen(
                        project, phrase_group, index, sound, validation_result, is_real_time=False
                    )
                    if err:
                        message += f"\n{COL_ERROR}Couldn't save file: {err} {saved_path}"
                    else:
                        project.sound_segments.delete_redundants_for(index)
                        message += f"\n{Ansi.ITALICS}{COL_DIM}{Path(saved_path).name}"

                printt(message)
            printt()

            if re_adds:
                # Insert "re_adds" at the head of the list (not bothering with deque fyi)
                items[:0] = re_adds

        # Print summary
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
            s += f"Num lines failed to generate: {COL_ERROR}{num_errored}{COL_DEFAULT}"
        printt(s)

        SigIntHandler().clear()
        return did_interrupt

    @staticmethod
    def generate_and_validate(
        project: Project, 
        indices: list[int],
        phrase_groups: list[PhraseGroup],
        stt_variant: SttVariant,
        stt_config: SttConfig,
        force_random_seed: bool,
        is_realtime: bool,
        is_skip_reason_buffer: bool=False
    ) -> list[ tuple[Sound, ValidationResult] | str ]:
        """
        Generates and validates a batch of prompts
        Prints updates
        Returns list of sounds etc

        :param indices:
            When length is 1, batch mode is disabled
        """
        
        # Dim print color during any model inference printouts
        print(f"{COL_DEFAULT}Generating audio...{Ansi.LINE_HOME}{COL_DIM}", end="", flush=True)

        # Generate:
        gen_start_time = time.time()
        gen_results = GenerateUtil.generate(
            project=project, 
            indices=indices,
            phrase_groups=phrase_groups,
            force_random_seed=force_random_seed,
            is_realtime=is_realtime,
            gen_info_path=project.sound_segments_path
        )        
        printt() # Restore print color, print blank line
        
        # Print speed info
        print_speed_info(time.time() - gen_start_time, gen_results)

        # Should skip or not
        skip_reason = ""
        if stt_variant == SttVariant.DISABLED:
            skip_reason = "Whisper disabled"
        elif is_skip_reason_buffer:
            skip_reason = "Buffer duration too short"
        elif ValidateUtil.is_unsupported_language_code(project.language_code):
            skip_reason = "Unsupported language"

        if not skip_reason:
            printt(f"{COL_DEFAULT}Transcribing audio...", end="") # gets printed over

        val_start_time = time.time()
        results: list[ tuple[Sound, ValidationResult] | str ] = []

        for i, gen_result in enumerate(gen_results):

            index = indices[i]

            if isinstance(gen_result, str):
                err = gen_result
                printt(f"Text segment {index+1} - error: {err}")
                results.append(err)
                continue

            sound = gen_result

            if skip_reason:
                validation_result = SkippedResult(skip_reason)
                results.append((sound, validation_result))
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
            validation_result = ValidateUtil.validate_item(
                sound, text, transcribed_words, project.language_code, strictness=project.strictness
            )

            if isinstance(validation_result, TrimmableResult):
                # TODO: adjust transcribed words based on trim 
                gen_start_time = validation_result.start_time or 0
                end_time = validation_result.end_time or sound.duration
                trimmed_sound = SoundUtil.trim(sound, gen_start_time, end_time)
                results.append((trimmed_sound, validation_result))
            else:
                results.append((sound, validation_result))

        if not skip_reason:
            message = f"{COL_DEFAULT}Transcribing audio complete: {(time.time() - val_start_time):.1f}s"
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
            gen_info_path: str
        ) -> list[Sound | str]:
        """
        Core audio generation function.
        Returns a list of "results" (same length as that of prompt_text), 
        where each result is either a generated Sound or an error string.

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
        match Tts.get_type():

            case TtsModelInfos.OUTE:

                result = Tts.get_oute().generate(
                    prompts[0],
                    project.oute_voice_json,
                    project.oute_temperature)

            case TtsModelInfos.CHATTERBOX:

                if project.chatterbox_voice_file_name:
                    voice_path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
                else:
                    voice_path = ""
                
                result = Tts.get_chatterbox().generate(
                    text=prompts[0],
                    voice_path=voice_path,
                    exaggeration=project.chatterbox_exaggeration,
                    cfg=project.chatterbox_cfg,
                    temperature=project.chatterbox_temperature,
                    seed=-1 if force_random_seed else project.chatterbox_seed,
                    language_id=project.language_code
                )

            case TtsModelInfos.FISH:

                if project.fish_voice_file_name:
                    source_path = os.path.join(project.dir_path, project.fish_voice_file_name)
                    Tts.get_fish().set_voice_clone_using(
                        source_path=source_path,
                        transcribed_text=project.fish_voice_transcript
                    )
                else:
                    Tts.get_fish().clear_voice_clone()

                result = Tts.get_fish().generate(
                    prompts[0], 
                    project.fish_temperature,
                    seed=-1 if force_random_seed else project.fish_seed,
                )

            case TtsModelInfos.HIGGS:

                if project.higgs_voice_file_name:
                    voice_path = os.path.join(project.dir_path, project.higgs_voice_file_name)
                    voice_transcript = project.higgs_voice_transcript
                else:
                    voice_path = ""
                    voice_transcript = ""
                if project.higgs_temperature == -1:
                    temperature = HiggsModelProtocol.DEFAULT_TEMPERATURE
                else:
                    temperature = project.higgs_temperature
                result = Tts.get_higgs().generate(
                    p_voice_path=voice_path, # TODO is this loading every gen?
                    p_voice_transcript=voice_transcript,
                    text=prompts[0],
                    seed=random.randint(1, sys.maxsize),
                    temperature=temperature
                )

            case TtsModelInfos.VIBEVOICE:

                voice_path = os.path.join(project.dir_path, project.vibevoice_voice_file_name)
                cfg_scale = VibeVoiceProtocol.DEFAULT_CFG if project.vibevoice_cfg == -1 else project.vibevoice_cfg
                num_steps = VibeVoiceProtocol.DEFAULT_NUM_STEPS if project.vibevoice_steps == -1 else project.vibevoice_steps

                result = Tts.get_vibevoice().generate(
                    text=prompts[0],
                    voice_path=voice_path,
                    cfg_scale=cfg_scale,
                    num_steps=num_steps
                )

            case TtsModelInfos.INDEXTTS2:

                voice_path = os.path.join(project.dir_path, project.indextts2_voice_file_name)

                if project.indextts2_emo_voice_file_name:
                    emo_voice_path = os.path.join(project.dir_path, project.indextts2_emo_voice_file_name)
                else:
                    emo_voice_path = ""

                result = Tts.get_indextts2().generate(
                    text=prompts[0],
                    voice_path=voice_path,
                    temperature=project.indextts2_temperature,
                    emo_alpha=project.indextts2_emo_alpha,
                    emo_voice_path=emo_voice_path,
                    emo_vector=project.indextts2_emo_vector
                )

            case TtsModelInfos.GLM:

                voice_path = os.path.join(project.dir_path, project.glm_voice_file_name)
                voice_transcript = project.glm_voice_transcript
                
                result = Tts.get_glm().generate(
                    prompt_text=voice_transcript,
                    prompt_speech=voice_path,
                    syn_text=prompts[0],
                    seed=-1 if force_random_seed else project.glm_seed
                )

            case TtsModelInfos.MIRA:

                voice_path = os.path.join(project.dir_path, project.mira_voice_file_name)
                Tts.get_mira().set_voice_clone(voice_path)

                if project.mira_temperature == -1:
                    temperature = MiraProtocol.TEMPERATURE_DEFAULT
                elif project.mira_temperature < MiraProtocol.TEMPERATURE_MIN or project.mira_temperature > MiraProtocol.TEMPERATURE_MAX:
                    temperature = MiraProtocol.TEMPERATURE_DEFAULT
                else:
                    temperature = project.mira_temperature
                Tts.get_mira().set_temperature(temperature)

                if len(prompts) == 1:
                    result = Tts.get_mira().generate(prompts[0])
                else:
                    result = Tts.get_mira().generate_batch(prompts)

            case TtsModelInfos.NONE:
                result = "No active TTS model"

        # Result is either an error string or n generated Sounds
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
                GenInfo.save_sound(gen_info_path, indices[i], "raw", sound, is_realtime=is_realtime)

                # Trim "silence" from ends of audio clip
                sound = SilenceUtil.trim_silence(sound)

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
        prompt = TextNormalizer.apply_prompt_word_substitutions(prompt, project.word_substitutions, project.language_code)
        prompt = TextNormalizer.normalize_prompt_common(prompt, project.language_code)
        prompt = Tts.get_instance().massage_for_inference(prompt)
        return prompt

    @staticmethod
    def save_gen(
        project: Project,
        phrase_group: PhraseGroup,
        index: int,
        sound: Sound,
        validation_result: ValidationResult,
        is_real_time: bool
    ) -> tuple[str, str]:
        """
        Saves sound segment and timing info json
        Returns error string (if any), saved file path
        """
        
        # Save sound
        num_word_fails = validation_result.num_word_fails if isinstance(validation_result, (PassResult, FailResult)) else -1
        file_name = SoundSegmentUtil.make_file_name(
            index=index,
            phrase_group=phrase_group,
            project=project,
            tts_model_info=Tts.get_type().value,
            num_word_fails=num_word_fails,
            is_real_time=is_real_time
        )
        if isinstance(validation_result, FailResult):
            file_name = AppUtil.insert_bracket_tag_file_path(file_name, "fail")
        
        dir_path = project.realtime_path if is_real_time else project.sound_segments_path 
        os.makedirs(dir_path, exist_ok=True)

        sound_path = os.path.join(dir_path, file_name)
        err = SoundFileUtil.save_flac(sound, sound_path)
        if err:
            return err, sound_path
        
        if not is_real_time and isinstance(validation_result, TranscriptResult): 
            # Make timing metadata and save to 'parallel' json file        
            phrase_group = project.phrase_groups[index]
            timed_phrases = ForceAlignUtil.make_timed_phrases(
                phrases=phrase_group.phrases,
                words=validation_result.transcript_words,
                sound_duration=sound.duration
            )
            if isinstance(validation_result, TrimmableResult):
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

        if DEV_SAVE_INTERMEDIATE_FILES and not is_real_time:
            # Make and save debug json
            if isinstance(validation_result, TranscriptResult): 
                d = {}
                prompt = GenerateUtil.phrase_group_to_prompt(phrase_group, project)
                source = phrase_group.as_flattened_phrase().text
                transcript = WhisperUtil.get_flat_text_from_words(validation_result.transcript_words)
                normalized_source, normalized_transcript = \
                    TextNormalizer.normalize_source_and_transcript(source, transcript, project.language_code)
                d["language_code"] = project.language_code
                d["prompt"] = repr(prompt)
                d["source"] = repr(source)
                d["transc"] = repr(transcript)
                d["normalized_source"] = repr(normalized_source)
                d["normalized_transc"] = repr(normalized_transcript)
                d["result_class"] = type(validation_result).__name__
                d["result_desc"] = strip_ansi( validation_result.get_ui_message() )
                file_name = SoundSegmentUtil.make_file_name(
                    index=index,
                    phrase_group=phrase_group,
                    project=project,
                    tts_model_info=Tts.get_type().value,
                    num_word_fails=num_word_fails,
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

        return "", sound_path

    @staticmethod
    def print_item_count_heading(num_complete: int, num_remaining: int, num_total: int) -> None:
        message = f"{COL_ACCENT}Items complete: {COL_DEFAULT}{num_complete} "
        message += f"{COL_ACCENT}Remaining: {COL_DEFAULT}{num_remaining} "
        message += f"{COL_ACCENT}Total: {COL_DEFAULT}{num_total}"
        printt(f"{COL_ACCENT}{'-' * 60}")
        printt(f"{message}")
        printt()

# ---

def print_speed_info(gen_elapsed: float, gen_results: list) -> None:
    
    # Elapsed
    message = f"Generating audio complete: {gen_elapsed:.1f}s"

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
        speed = min(speed, 100)
        speed_str = f"{speed:.1f}"
        if speed < 1.0:
            speed_str = "0" + speed_str
        elif speed == 100:
            speed_str += "+"
        speed_str += "x"
        message += f"speed: {speed_str}"
    
    print(message)
    printt()
    
    # Inference time: 10.0s; cumulative sound duration: 20.0s; speed: 2.0x


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

def adjust_timed_phrases_trimmed(timed_phrases: list[TimedPhrase], trimmable_result: TrimmableResult) -> list[TimedPhrase]:
    
    start_offset = trimmable_result.start_time or 0
    end_time = trimmable_result.end_time or trimmable_result.duration
    full_duration = end_time - start_offset
    
    results = []
    
    for item in timed_phrases:
        time_start = max(item.time_start - start_offset, 0)
        time_end = min(item.time_end, full_duration)
        new_item = TimedPhrase(text=item.text, time_start=time_start, time_end=time_end)
        results.append(new_item)

    return results

