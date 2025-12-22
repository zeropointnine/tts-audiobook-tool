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
from tts_audiobook_tool.text_util import TextUtil
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_model import HiggsModelProtocol, VibeVoiceProtocol
from tts_audiobook_tool.tts_model_info import TtsModelInfos
from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.validate_util import ValidateUtil
from tts_audiobook_tool.whisper_util import WhisperUtil

class GenerateUtil:

    @staticmethod
    def generate_to_files(
            project: Project, 
            phrase_groups: list[PhraseGroup],
            indices_set: set[int],
            language_code: str,
            is_regenerate: bool,
            stt_variant: SttVariant,
            stt_config: SttConfig,
    ) -> bool:
        """
        Subroutine for doing a series of audio generations to files

        :param indices_to_generate:

        :param items_to_regenerate:
            is mutually exclusive to indices_to_generate
            "Regenerate" here simply means that the currently existing, failed file
            for the given index will be deleted first.

        Prints feedback at end of each item, and summary at end of loop.

        Returns True if ended because interrupted.
        """

        force_no_stt = ValidateUtil.is_unsupported_language_code(project.language_code)
        did_cancel = Tts.warm_up_models(force_no_stt)
        if did_cancel:
            print_feedback("\nCancelled")
            return True

        num_processed = 0
        num_saved_ok = 0
        num_saved_with_fail = 0
        num_couldnt_save = 0
        did_interrupt = False
        saved_elapsed: list[float] = [] # history of end-to-end time taken to create sound files
        start_time = time.time()
        saved_start_time = time.time()

        indices = sorted(list(indices_set))

        for i in range(len(indices)):
            
            index = indices[i]
            phrase_group = phrase_groups[index]
 
            printt()
            GenerateUtil.print_item_heading(
                is_regenerate, phrase_group.presentable_text, index, num_processed, len(indices)
            )

            result, did_interrupt = GenerateUtil.generate_full_flow(
                project, phrase_group, index, language_code, stt_variant, stt_config
            )
            match result:
                case FullFlowResult.SAVED_OK:
                    num_saved_ok += 1
                    num_processed += 1
                case FullFlowResult.SAVED_WITH_FAIL:
                    num_saved_with_fail += 1
                    num_processed += 1
                case FullFlowResult.COULDNT_SAVE:
                    num_couldnt_save += 1
                    num_processed += 1

            # Update durations and print ETA info
            if result != FullFlowResult.NOTHING:                    
                saved_elapsed.append(time.time() - saved_start_time)
                saved_start_time = time.time()
            if not (i == len(indices) - 1):
                print_eta(saved_elapsed, len(indices) - num_processed)

            if did_interrupt:
                break

        # Print summary
        s = "\n"
        if did_interrupt:
            s += "Interrupted. "
        s += f"Elapsed: {duration_string(time.time() - start_time)}\n"
        s += "\n"
        ok = str(num_saved_ok)
        if num_saved_ok == len(indices):
            ok += " (all)"
        s += f"Num lines saved: {COL_OK}{ok}{COL_DEFAULT}\n"
        col = COL_ACCENT if num_saved_with_fail else ""
        if num_saved_with_fail:
            s += f"Num lines saved, but flagged with potential errors: {col}{num_saved_with_fail}{COL_DEFAULT}\n"
        if num_couldnt_save:
            s += f"Num lines failed to generate: {COL_ERROR}{num_couldnt_save}{COL_DEFAULT}"
        printt(s)

        return did_interrupt

    @staticmethod
    def generate_full_flow(
            project: Project,
            phrase_group: PhraseGroup,
            index: int,
            language_code: str,
            stt_variant: SttVariant,
            stt_config: SttConfig
    ) -> tuple[FullFlowResult, bool]:
        """
        Generates, validates, saves, retries if necessary
        Returns: FullFlowResult and if user interrupted
        """

        SigIntHandler().set("generating")

        result: FullFlowResult = FullFlowResult.NOTHING
        did_interrupt = False

        MAX_ATTEMPTS = 2
        for attempt in range(MAX_ATTEMPTS):

            if SigIntHandler().did_interrupt:
                did_interrupt = True

            if result == FullFlowResult.SAVED_OK or did_interrupt:
                break

            validation_result = GenerateUtil.generate_and_validate(
                project=project,
                phrase_group=phrase_group,
                index=index,
                language_code=language_code,
                stt_variant=stt_variant,
                stt_config=stt_config,
                gen_info_path=project.sound_segments_path,
                is_retry=(attempt > 0),
                is_realtime=False
            )

            if isinstance(validation_result, str):
                if result == FullFlowResult.NOTHING:
                    result = FullFlowResult.COULDNT_SAVE
                err = validation_result
                printt(f"{COL_ERROR}Model fail: {err}")
                printt()
                continue

            sound, validation_result = validation_result

            GenerateUtil.print_validation_result(
                validation_result, is_last_attempt=(attempt == MAX_ATTEMPTS - 1),is_real_time=False
            )

            did_save = GenerateUtil._do_save_gen(project, phrase_group, index, sound, validation_result)
            if not did_save:
                if result == FullFlowResult.NOTHING:
                    result = FullFlowResult.COULDNT_SAVE
            else:
                result = FullFlowResult.SAVED_WITH_FAIL if isinstance(validation_result, FailResult) else FullFlowResult.SAVED_OK

        SigIntHandler().clear()
        return result, did_interrupt

    @staticmethod
    def _do_save_gen(
        project: Project,
        phrase_group: PhraseGroup,
        index: int,
        sound: Sound,
        validation_result: ValidationResult,
    ) -> bool:
        """ 
        Saves, prints save feedback.
        Returns True on success
        """
        err, saved_path = GenerateUtil.save_gen(
            project, phrase_group, index, sound, validation_result, is_real_time=False
        )
        if err:
            printt(f"{COL_ERROR}Couldn't save file: {err} {saved_path}")
            return False
        printt(f"Saved: {saved_path}")
        project.sound_segments.delete_redundants_for(index)
        return True

    @staticmethod
    def generate_and_validate(
        project: Project, 
        phrase_group: PhraseGroup,
        index: int,
        language_code: str,
        stt_variant: SttVariant,
        stt_config: SttConfig,
        is_retry: bool,
        is_realtime: bool,
        gen_info_path: str,
        is_skip_reason_buffer: bool=False
    ) -> tuple[Sound, ValidationResult] | str:
        """
        Returns generated sound and ValidationResult, or error string
        """

        start_time = time.time()
        
        text = phrase_group.as_flattened_phrase().text

        gen_info = GenInfo(gen_info_path, index, phrase_group, language_code=language_code)

        # Generate:
        
        # Dim color during any model inference printouts
        print(COL_DIM, end="", flush=True) 
        
        result = GenerateUtil.generate(
            project=project, 
            prompt_text=text, 
            force_random_seed=is_retry,
            index=index,
            is_realtime=is_realtime,
            gen_info_path=gen_info_path
        )

        # Restore print color, print blank line
        printt() 

        if isinstance(result, str):
            err = result
            if err == NO_VOCALIZABLE_CONTENT:
                # Special case - return short silence # TODO: refactor this out
                sr = Tts.get_type().value.sample_rate
                dtype = Tts.get_type().value.dtype
                silence_sound = SoundUtil.make_silence_sound(0.1, sr, dtype)
                if stt_variant == SttVariant.DISABLED or is_skip_reason_buffer:
                    validation_result = SkippedResult("No vocalizable content")
                else:
                    validation_result = PassResult([], 0, 1)
                gen_info.set_validation_result(validation_result)
                gen_info.save(is_realtime=is_realtime)
                return silence_sound, validation_result
            else:
                gen_info.set_model_error(err)
                gen_info.save(is_realtime=is_realtime)
                return err

        sound = result

        elapsed = time.time() - start_time or 1.0
        print_speed_info(result.duration, elapsed)

        # Check if should skip transcription
        skip_reason = ""
        if stt_variant == SttVariant.DISABLED:
            skip_reason = "Whisper disabled"
        elif is_skip_reason_buffer:
            skip_reason = "Buffer duration too short"
        elif ValidateUtil.is_unsupported_language_code(project.language_code):
            skip_reason = "Unsupported language"
        if skip_reason:
            validation_result = SkippedResult(skip_reason)
            gen_info.set_validation_result(validation_result)
            gen_info.save(is_realtime=is_realtime)
            return sound, validation_result

        # Transcribe
        transcribe_start_time = time.time()
        result = WhisperUtil.transcribe_to_words(
            sound, stt_variant, stt_config, language_code=project.language_code
        )
        if isinstance(result, str):
            err = result                
            gen_info.set_model_error(err)
            gen_info.save(is_realtime=is_realtime)
            return err
        transcribed_words = result
        gen_info.set_transcript(transcribed_words)
        printt(f"Transcribed audio ({COL_ACCENT}{(time.time() - transcribe_start_time):.1f}s{COL_DEFAULT})")

        # Validate
        validation_result = ValidateUtil.validate_item(sound, text, transcribed_words, project.language_code)
        gen_info.set_validation_result(validation_result)
        gen_info.save(is_realtime=is_realtime, is_retry=is_retry)

        if isinstance(validation_result, TrimmableResult):
            # TODO: adjust transcribed words based on trim 
            start_time = validation_result.start_time or 0
            end_time = validation_result.end_time or sound.duration
            trimmed_sound = SoundUtil.trim(sound, start_time, end_time)
            GenInfo.save_sound(gen_info_path, index, "trimmed", trimmed_sound, is_realtime=is_realtime)
            return trimmed_sound, validation_result
        else:
            return sound, validation_result

    @staticmethod
    def generate(
            project: Project, # TODO: make ModelSettings proper
            prompt_text: str, 
            force_random_seed: bool, 
            index: int, 
            is_realtime: bool,
            gen_info_path: str
        ) -> Sound | str:
        """
        Core audio generation function.
        Returns model-generated sound data (in model's native samplerate) or error string
        """

        if DEV:
            printt(f"{COL_DIM}Pre-normalized prompt text before word substitutions:")
            printt(f"{COL_DIM}{prompt_text.strip()}")
        prompt_text = TextNormalizer.apply_prompt_word_substitutions(prompt_text, project.word_substitutions, project.language_code)
        if DEV:
            printt(f"{COL_DIM}Pre-normalized prompt text after word substitutions:")
            printt(f"{COL_DIM}{prompt_text.strip()}")
            printt()
        prompt_text = TextNormalizer.normalize_prompt_common(prompt_text, project.language_code)
        prompt_text = Tts.get_instance().massage_for_inference(prompt_text)

        if TextUtil.is_ws_punc(prompt_text):
            return NO_VOCALIZABLE_CONTENT

        match Tts.get_type():

            case TtsModelInfos.OUTE:

                result = Tts.get_oute().generate(
                    prompt_text,
                    project.oute_voice_json,
                    project.oute_temperature)

            case TtsModelInfos.CHATTERBOX:

                if project.chatterbox_voice_file_name:
                    voice_path = os.path.join(project.dir_path, project.chatterbox_voice_file_name)
                else:
                    voice_path = ""
                result = Tts.get_chatterbox().generate(
                    text=prompt_text,
                    voice_path=voice_path,
                    exaggeration=project.chatterbox_exaggeration,
                    cfg=project.chatterbox_cfg,
                    temperature=project.chatterbox_temperature,
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
                result = Tts.get_fish().generate(prompt_text, project.fish_temperature)

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
                    text=prompt_text,
                    seed=random.randint(1, sys.maxsize),
                    temperature=temperature
                )

            case TtsModelInfos.VIBEVOICE:

                voice_path = os.path.join(project.dir_path, project.vibevoice_voice_file_name)
                cfg_scale = VibeVoiceProtocol.DEFAULT_CFG if project.vibevoice_cfg == -1 else project.vibevoice_cfg
                num_steps = VibeVoiceProtocol.DEFAULT_NUM_STEPS if project.vibevoice_steps == -1 else project.vibevoice_steps

                result = Tts.get_vibevoice().generate(
                    text=prompt_text,
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
                    text=prompt_text,
                    voice_path=voice_path,
                    temperature=project.indextts2_temperature,
                    emo_alpha=project.indextts2_emo_alpha,
                    emo_voice_path=emo_voice_path,
                    emo_vector=project.indextts2_emo_vector
                )

            case TtsModelInfos.GLM:

                voice_path = os.path.join(project.dir_path, project.glm_voice_file_name)
                voice_transcript = project.glm_voice_transcript
                if project.glm_seed == -1 or force_random_seed:
                    seed = random.randint(0,  2**32 - 1)
                else:
                    seed = project.glm_seed
                result = Tts.get_glm().generate(
                    prompt_text=voice_transcript,
                    prompt_speech=voice_path,
                    syn_text=prompt_text,
                    seed=seed
                )

            case TtsModelInfos.NONE:
                return "No active TTS model"

        if isinstance(result, str):
            return result

        sound = result
        if sound.data.size == 0:
            return "Model output is empty, discarding"

        num_nans = np.sum(np.isnan(sound.data))
        if num_nans > 0:
            return "Model outputted NaN, discarding"

        # Save "raw" output
        GenInfo.save_sound(gen_info_path, index, "raw", sound, is_realtime=is_realtime)

        # Trim "silence" from ends of audio clip
        sound = SilenceUtil.trim_silence(sound)

        # Re-check size
        if sound.data.size == 0:
            return "Model output is silence, discarding"

        # Do peak normalization
        normalized_data = SoundUtil.normalize(sound.data, headroom_db=3.0)
        sound = Sound(normalized_data, sound.sr)

        return sound

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
        Saves "sound segment" and timing info json
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

        return "", sound_path

    @staticmethod
    def print_item_heading(is_regenerate: bool, text: str, index: int, count: int, total: int) -> None:
        verb = "Regenerating" if is_regenerate else "Generating"
        s  = f"{COL_ACCENT}[{COL_DEFAULT}{count+1}{COL_ACCENT}/{COL_DEFAULT}{total}{COL_ACCENT}] "
        s += f"{COL_ACCENT}{verb} audio for text segment {COL_DEFAULT}{index+1}{COL_ACCENT}:{COL_DEFAULT}"
        printt(s)
        printt(f"{COL_DIM}{Ansi.ITALICS}{text.strip()}")
        printt()

    @staticmethod
    def print_validation_result(
        result: ValidationResult, 
        is_last_attempt: bool,
        is_real_time: bool,
        custom_skip_text: str = ""
    ) -> None:

        s = "Speech-to-text validation: "
        if isinstance(result, SkippedResult):
            skip_message = custom_skip_text if custom_skip_text else result.get_ui_message()
            s += f"Skipped {COL_DIM}({skip_message}){COL_DEFAULT}"
        elif isinstance(result, PassResult):
            s += f"{COL_OK}Passed {COL_DIM}(word fails={result.num_word_fails}, threshold={result.word_fail_threshold}){COL_DEFAULT}"
        elif isinstance(result, TrimmableResult):
            s += f"{COL_OK}Fixed{COL_DEFAULT} - {result.get_ui_message()}"
        elif isinstance(result, FailResult):
            s += f"{COL_ERROR}Failed {COL_DIM}(word fails={result.num_word_fails}, threshold={result.word_fail_threshold}){COL_DEFAULT}\n"
            if not is_last_attempt:
                s += f"Will retry"
            else:
                s += f"Max attempts reached, "
                s += "playing anyway" if is_real_time else "continuing to next"
        printt(s)

# ---

class FullFlowResult(Enum):
    SAVED_OK = auto(), # a good gen was saved
    SAVED_WITH_FAIL = auto(), # a FailResult gen was saved
    COULDNT_SAVE = auto(), # couldn't save due to model error or save error
    NOTHING = auto() # no gen attempted (user broke out before anything could happen)

def print_speed_info(sound_duration: float, elapsed: float) -> None:
    multi = sound_duration / elapsed
    s = f"Generated audio ({COL_ACCENT}{elapsed:.1f}s{COL_DEFAULT}) duration: {COL_ACCENT}{sound_duration:.1f}s"
    s += f"{COL_DEFAULT} = {COL_ACCENT}{multi:.2f}x"
    printt(s)

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

NO_VOCALIZABLE_CONTENT = "No vocalizable content"
