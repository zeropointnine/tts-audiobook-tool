import time

import numpy as np
from tts_audiobook_tool import app_support, text_util
from tts_audiobook_tool.app_types import Sound, SttVariant
from tts_audiobook_tool import ask
from tts_audiobook_tool.generate_util import GenerateUtil, TtsModelError
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool import readiness
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.menus.menu_util import MenuUtil
from tts_audiobook_tool.util import *
from tts_audiobook_tool.app_types.validation_result import ValidationResult

"""
Coordinates and drives TTS playback in "real time".
Manages output buffer growth, validate-and-retry handling
intra-segment pauses, interruption/shutdown behavior.

Similar to `GenerateUtil.generate_files()` but outputs to sound device instead of to files.

Is blocking.
"""

def start(
        state: State,
        phrase_groups: list[PhraseGroup],
        line_range: tuple[int, int] | None
    ) -> None:
    """
    line_range is one-indexed
    """
    if not line_range or line_range == (0, 0):
        line_range = (1, len(phrase_groups))

    # Warm up models
    warm_up_result = ModelManager.warm_up_models(state)
    if warm_up_result.should_stop:
        app_support.print_warm_up_result_stop(warm_up_result)
        if warm_up_result.error:
            app_memory.gc_ram_vram()
        if state.prefs.menu_clears_screen:
            ask.ask_enter_to_continue()
        return

    # Do model readiness check now that model instance exists
    err = readiness.get_generate_blocker_text(state, verbose=True, is_realtime_playback=True)
    if err:
        print_feedback(err, is_error=True)
        return

    # Print warnings if any
    warnings = Tts.get_instance().get_warning_issues(state.project)
    if warnings:
        warnings_string = "\n".join(warnings)
        print_feedback(Ansi.ITALICS + warnings_string, no_preformat=True)

    showed_vram_warning = app_memory.show_vram_memory_warning_if_necessary()

    s = "Starting real-time playback..."
    if state.prefs.stt_variant == SttVariant.DISABLED:
        s += f" {COL_DIM}(speech-to-text validation disabled){COL_ACCENT}"
    MenuUtil.print_heading(None, s, dont_clear=True, non_menu=True)
    printt(f"{COL_DIM}Press {COL_ACCENT}[control-c]{COL_DIM} to interrupt")
    printt()

    # Realtime playback is a top-level generation run. If the previous run ended
    # normally within one paragraph, there may be no phrase-level break reason to
    # clear rolling continuation state, so force a fresh context at run start.
    Tts.clear_continuation()
    Tts.reset_voice_rotation_index()

    # Outer loop

    Interrupts().set("generating")
    did_interrupt = False
    stream = None
    count = 0
    consecutive_model_errors = 0
    max_consecutive_model_errors = 5

    start_index, end_index = line_range
    start_index -= 1
    end_index -= 1
    start_time = time.time()

    for one_second in range(start_index, end_index + 1):

        if did_interrupt:
            break

        if not showed_vram_warning:
            b = app_memory.show_vram_memory_warning_if_necessary()
            if b:
                print("\a", end="")
                showed_vram_warning = True

        phrase_group = phrase_groups[one_second]
        phrase = phrase_group.as_flattened_phrase()

        printt()
        GenerateUtil.print_batch_heading(
            indices=[one_second],
            num_complete=one_second - start_index,
            num_remaining=end_index + 1 - one_second,
            num_total=end_index + 1 - start_index,
            start_time=start_time
        )
        printt(f"{COL_DIM_ITALICS}{phrase_group.presentable_text}")
        printt()

        # TODO: make dynamic - if "estimated gen time" < buffer duration x ~2 x max_retries...
        has_runway = (stream is not None and stream.buffer_duration >= (REQUIRED_SECONDS_PER_RETRY * state.project.max_retries))

        sound_opt, did_interrupt, consecutive_model_errors = generate_full_flow(
            state,
            phrase_groups,
            one_second,
            has_runway=has_runway,
            consecutive_model_errors=consecutive_model_errors,
            max_consecutive_model_errors=max_consecutive_model_errors,
        )
        if not did_interrupt:
            # generate_full_flow() clears Interrupts at the end, so re-arm
            # Ctrl-C handling for the outer realtime loop and buffer-throttle sleep.
            Interrupts().set("generating")
        if not sound_opt:
            if did_interrupt:
                Tts.clear_continuation()
                break
            printt(f"{COL_ERROR}Coun't generate sound{COL_DIM}, continuing to next segment")
            printt()
            continue
        else:
            sound = sound_opt

        original_duration = sound.duration
        sound = SoundPipeline.prepare_generated_sound_for_playback(
            sound=sound,
            high_shelf=state.project.get_high_shelf(),
            limit_silence_gaps=state.project.limit_silence_gaps,
            limit_silence_gaps_duration=state.project.limit_silence_gaps_duration,
        )

        if sound.data.size > 0 and abs(sound.duration - original_duration) > 0.01:
            trimmed_ms = (original_duration - sound.duration) * 1000
            L.d(f"Trimmed: Duration {original_duration:.3f}s -> {sound.duration:.3f}s (trimmed {trimmed_ms:.0f}ms)")

        # Add appended sound
        if one_second == end_index:
            appended_sound = None
        else:
            is_first_in_section = one_second == 0 or one_second in state.project.markers
            use_sound_effect = SoundPipeline.should_append_break_sound_effect(
                phrase.reason,
                use_break_sound_effect=state.project.use_break_sound_effect,
                is_first_in_section=is_first_in_section,
            )
            if use_sound_effect:
                if phrase.reason == Reason.SECTION_BREAK:
                    path = SECTION_BREAK_SOUND_EFFECT_PATH
                else:
                    path = SPACE_BREAK_SOUND_EFFECT_PATH
                result = SoundFileUtil.load(path, sound.sr)
                if isinstance(result, str):
                    printt(f"{COL_ERROR}Error loading sound effect: {path}")
                    appended_sound = None
                else:
                    appended_sound = result.data
            else:
                silence_duration = phrase.reason.pause_duration
                appended_sound = np.zeros(int(sound.sr * silence_duration), dtype=sound.data.dtype)

        # Start stream lazy
        if not stream:
            stream = SoundDeviceStream()
            stream.start()

        # Add sound to the stream
        stream.add_data(sound.data)
        if appended_sound is not None:
            # Add page-turn sound
            stream.add_data(appended_sound)

        full_duration = sound.duration
        if appended_sound is not None:
            full_duration += len(appended_sound) / sound.sr

        # Print buffer duration
        value = stream.buffer_duration - full_duration
        if value <= 0.0:
            value = +0.0
        s = f"{COL_ERROR}" if value < 0.1 else f"{COL_OK}"
        s += f"{duration_string(value, include_tenth=True)}"
        printt(f"Buffer duration: {s}")

        # Sleep if necessary to prevent growing buffer beyond threshold
        if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS:
            sleep_duration = int(full_duration)
            printt(f"{COL_DIM_ITALICS}Sleeping for {sleep_duration}s ...")
            did_interrupt = sleep_interruptibly(sleep_duration)
            if did_interrupt:
                break

        count += 1

        if did_interrupt:
            Tts.clear_continuation()
            break

    # Finished
    Interrupts().clear()

    should_prompt_before_shutdown = stream and stream.buffer_duration > 0
    if should_prompt_before_shutdown:
        # Prompt allows buffer to play until enter pressed
        printt()
        ask.ask_enter_to_continue()
    if stream:
        stream.shut_down()

    # Do not let rolling continuation state leak into the next realtime run.
    Tts.clear_continuation()

    if not should_prompt_before_shutdown:
        printt()
        ask.ask_enter_to_continue()

    printt()


def sleep_interruptibly(duration_s: float) -> bool:
    """
    Sleeps in short increments so Ctrl-C can stop realtime playback even
    while we're throttling to let the audio buffer drain.

    Returns True if interrupted.
    """
    deadline = time.time() + max(0.0, duration_s)
    while True:
        if Interrupts().did_interrupt:
            return True

        remaining = deadline - time.time()
        if remaining <= 0.0:
            return False

        time.sleep(min(INTERRUPTIBLE_SLEEP_POLL_SECONDS, remaining))


def generate_full_flow(
        state: State,
        phrase_groups: list[PhraseGroup],
        index: int,
        has_runway: bool,
        consecutive_model_errors: int = 0,
        max_consecutive_model_errors: int = 5,
) -> tuple[Sound | None, bool, int]:
    """
    Similar to `GenerateUtil.generate_full_flow()` but simpler control flow.
    Returns tuple: (Sound or None if problem, did_interrupt, consecutive_model_errors)
    """

    Interrupts().set("generating")

    project = state.project
    phrase_group = phrase_groups[index]
    did_interrupt = False

    gen_result: ValidationResult | str = ""
    num_attempts = 1 + project.max_retries if has_runway else 1

    for attempt in range(num_attempts):

        results = GenerateUtil.generate_and_validate_batch(
            state=state,
            indices=[index],
            phrase_groups=phrase_groups,
            stt_variant=state.prefs.stt_variant,
            stt_config=state.prefs.stt_config,
            force_random_seed=(attempt > 0),
            is_realtime=True,
            is_skip_reason_buffer=not has_runway
        )
        result = results[0]
        if isinstance(result, TtsModelError):
            consecutive_model_errors += 1
            gen_result = result.message
        else:
            consecutive_model_errors = 0
            gen_result = result

        if consecutive_model_errors >= max_consecutive_model_errors:
            GenerateUtil.print_consecutive_model_errors_message(max_consecutive_model_errors)
            Tts.clear_continuation()
            did_interrupt = True
            break

        # Check for OOM in results and break early to avoid wasting time
        if isinstance(gen_result, str) and is_oom_error_message(gen_result):
            print_gen_oom_message(gen_result)
            Tts.clear_continuation()
            did_interrupt = True
            break

        # Print result info
        if isinstance(gen_result, str):
            err = gen_result
            printt(f"{COL_ERROR}Model fail: {err}")
            printt()
            Tts.clear_continuation()
        else:
            printt(f"Transcript validation: {gen_result.get_ui_message_with_post_processing()}")

        if Interrupts().did_interrupt:
            did_interrupt = True

        if did_interrupt:
            Tts.clear_continuation()
            break
        is_pass = isinstance(gen_result, ValidationResult) and not gen_result.is_fail
        if isinstance(gen_result, ValidationResult) and gen_result.is_fail:
            Tts.clear_continuation()
        if is_pass:
            break

    Interrupts().clear()

    if isinstance(gen_result, str):
        Tts.clear_continuation()
        return None, did_interrupt, consecutive_model_errors  # is error
    else:
        validation_result = gen_result
        Tts.clear_continuation_if_reason(phrase_group.last_reason)
        if project.realtime_save:
            err, saved_path = GenerateUtil.save_sound_and_timing_json(
                state,
                phrase_group,
                index,
                validation_result,
                is_real_time=True,
                voice_tag=getattr(validation_result, "voice_tag", ""),
            )
            if err:
                printt(f"{COL_ERROR}Couldn't save file: {err} {saved_path}")
            else:
                url = saved_path
                text = Path(saved_path).name
                link = text_util.make_terminal_hyperlink(url=url, text=text, is_file=True)
                printt(f"Saved: {COL_DIM}{link}")
        return validation_result.sound, did_interrupt, consecutive_model_errors

# ---

INTERRUPTIBLE_SLEEP_POLL_SECONDS = 0.1
REQUIRED_SECONDS_PER_RETRY = 60.0
