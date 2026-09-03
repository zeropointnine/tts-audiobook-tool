"""
Coordinates and drives TTS playback in "real time".
Manages output buffer growth, validate-and-retry handling,
intra-segment pauses, interruption/shutdown behavior.

Similar to `GenerateUtil.generate_files()` but outputs to sound device instead of to files.

Is blocking.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np
from tts_audiobook_tool import text_util
from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool import ask
from tts_audiobook_tool.generate_util import (
    GenerateUtil,
    TtsModelError,
    make_consecutive_model_errors_reset_message,
    make_oom_reset_message,
)
from tts_audiobook_tool.gen_timeout_util import GenTimeoutTracker
from tts_audiobook_tool import app_support
from tts_audiobook_tool.app_support import app_memory
from tts_audiobook_tool.app_support.interrupts import Interrupts
from tts_audiobook_tool.model_manager import ModelManager
from tts_audiobook_tool import readiness
from tts_audiobook_tool.project_support.project_book_util import ProjectBookUtil
from tts_audiobook_tool.real_time_playback_events import (
    RealTimePlaybackAwaitingContinue,
    RealTimePlaybackBuffer,
    RealTimePlaybackEvents,
    RealTimePlaybackProgress,
    RealTimePlaybackSegmentText,
    RealTimePlaybackStarted,
)
from tts_audiobook_tool.generation_events import GenerationEvents, GenerationPhase, ModelUnhealthy
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.app_types.phrase import PhraseGroup, Reason
from tts_audiobook_tool.constants_config import *
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.l import L
from tts_audiobook_tool.util import *
from tts_audiobook_tool.app_types.validation_result import ValidationResult


class ContinueEvent(Protocol):
    def wait(self, timeout: float | None = None) -> bool: ...


class RealTimePlaybackRunStatus(str, Enum):
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    FAILED = "failed"


@dataclass(frozen=True)
class RealTimePlaybackRunResult:
    status: RealTimePlaybackRunStatus
    message: str = ""


def start(
    state: State,
    phrase_groups: list[PhraseGroup],
    line_range: tuple[int, int] | None,
    continue_event: ContinueEvent | None = None,
) -> RealTimePlaybackRunResult:
    """Run realtime generation/playback and always release the audio device."""
    stream_holder: list[SoundDeviceStream] = []
    try:
        return _start_impl(
            state,
            phrase_groups,
            line_range,
            continue_event,
            stream_holder,
        )
    finally:
        try:
            if stream_holder:
                stream_holder[0].shut_down()
        finally:
            # Realtime runs share the resident worker/model with later jobs;
            # continuation state must not survive failures or teardown errors.
            Tts.clear_continuation()


def _start_impl(
        state: State,
        phrase_groups: list[PhraseGroup],
        line_range: tuple[int, int] | None,
        continue_event: ContinueEvent | None,
        stream_holder: list[SoundDeviceStream],
    ) -> RealTimePlaybackRunResult:
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
        status = (
            RealTimePlaybackRunStatus.CANCELLED
            if warm_up_result.did_interrupt
            else RealTimePlaybackRunStatus.FAILED
        )
        return RealTimePlaybackRunResult(status, warm_up_result.error or "")

    # Do model readiness check now that model instance exists
    err = readiness.get_generate_blocker_text(state, verbose=True, is_realtime_playback=True)
    if err:
        ask.ask_error(err)
        return RealTimePlaybackRunResult(RealTimePlaybackRunStatus.FAILED, err)

    # Print warnings if any
    warnings = Tts.get_instance().get_warning_issues(state.project)
    if warnings:
        warnings_string = "\n".join(warnings)
        print_feedback(Ansi.ITALICS + warnings_string, no_preformat=True)
    showed_vram_warning = app_memory.show_vram_memory_warning_if_necessary()

    # Realtime playback is a top-level generation run. If the previous run ended
    # normally within one paragraph, there may be no phrase-level break reason to
    # clear rolling continuation state, so force a fresh context at run start.
    Tts.clear_continuation()
    Tts.reset_voice_selection_index()

    # Outer loop

    Interrupts().set("generating")
    did_interrupt = False
    stream = None
    consecutive_model_errors = 0
    max_consecutive_model_errors = 5
    # The first gen of the run may be dominated by model warm-up or download
    # time, so it alone is exempt from the GEN_TIMEOUT watchdog.
    gen_timeout_tracker = GenTimeoutTracker()

    start_index, end_index = line_range
    start_index -= 1
    end_index -= 1
    # line_range is one-indexed; defensively clamp a 0 start so we don't
    # wrap to the last group via negative indexing
    start_index = max(start_index, 0)
    total = max(0, end_index - start_index + 1)
    RealTimePlaybackEvents.emit(
        RealTimePlaybackStarted(total, start_index, end_index)
    )

    section_start_indices = set(ProjectBookUtil.get_section_start_indices(state.project))

    for index in range(start_index, end_index + 1):

        RealTimePlaybackEvents.emit(GenerationPhase("Generating audio"))
        RealTimePlaybackEvents.emit(
            RealTimePlaybackProgress(index - start_index, total, index)
        )

        # Pick up any Ctrl-C pressed during the previous iteration's
        # inter-segment gap (sound prep, buffer print, throttle sleep);
        # generate_full_flow()'s set("generating") would otherwise reset the
        # pending flag and silently swallow the interrupt.
        did_interrupt = did_interrupt or Interrupts().did_interrupt
        if did_interrupt:
            Tts.clear_continuation()
            break

        if not showed_vram_warning:
            b = app_memory.show_vram_memory_warning_if_necessary()
            if b:
                print("\a", end="")
                showed_vram_warning = True

        phrase_group = phrase_groups[index]
        phrase = phrase_group.as_flattened_phrase()

        printt()
        GenerateUtil.print_batch_heading(
            indices=[index],
            show_divider=index != start_index,
        )
        printt(f"{COL_DIM_ITALICS}{phrase_group.presentable_text}")
        printt()

        # TODO: make dynamic - if "estimated gen time" < buffer duration x ~2 x max_retries...
        has_runway = (stream is not None and stream.buffer_duration >= (REQUIRED_SECONDS_PER_RETRY * state.project.max_retries))

        sound_opt, did_interrupt, consecutive_model_errors = generate_full_flow(
            state,
            phrase_groups,
            index,
            has_runway=has_runway,
            consecutive_model_errors=consecutive_model_errors,
            max_consecutive_model_errors=max_consecutive_model_errors,
            gen_timeout_tracker=gen_timeout_tracker,
        )
        if not did_interrupt:
            # generate_full_flow() clears Interrupts at the end, so re-arm
            # Ctrl-C handling for the outer realtime loop and buffer-throttle sleep.
            Interrupts().set("generating")
        if did_interrupt:
            # Interrupt during generation takes priority even if this segment
            # still produced a sound; break before the throttle sleep so the
            # flag cannot be clobbered (generate_full_flow() already cleared
            # the Interrupts state).
            Tts.clear_continuation()
            break
        if not sound_opt:
            printt(f"{COL_ERROR}Couldn't generate sound{COL_DIM}, continuing to next segment")
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
        if index == end_index:
            appended_sound = None
        else:
            is_first_in_section = index in section_start_indices
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
                silence_duration = state.project.reason_pauses.get_pause_for(phrase.reason)
                appended_sound = np.zeros(int(sound.sr * silence_duration), dtype=sound.data.dtype)

        # Start stream lazy
        if not stream:
            stream = SoundDeviceStream()
            stream_holder.append(stream)
            if not stream.start():
                # Abort. Stream is in a not-started state, so no shut_down() is needed;
                # skip the buffer-drain prompt.
                Tts.clear_continuation()
                Interrupts().clear()
                s = "Aborting real-time playback: sound output stream failed to start"
                ask.ask_error(s)
                return RealTimePlaybackRunResult(
                    RealTimePlaybackRunStatus.FAILED,
                    s,
                )

        # Add sound to the stream, capturing the half-open sample range it
        # occupies so the app can show which source text is being played.
        segment_start, segment_end = stream.add_data(sound.data)
        if appended_sound is not None:
            # Add page-turn sound
            _, segment_end = stream.add_data(appended_sound)

        full_duration = sound.duration
        if appended_sound is not None:
            full_duration += len(appended_sound) / sound.sr

        RealTimePlaybackEvents.emit(RealTimePlaybackBuffer(stream.buffer_duration))
        if segment_end > segment_start:
            RealTimePlaybackEvents.emit(
                RealTimePlaybackSegmentText(
                    index=index,
                    text=phrase_group.presentable_text,
                    start_sample=segment_start,
                    end_sample=segment_end,
                    played_samples=stream.played_samples,
                )
            )

        # Pick up an interrupt pressed during sound prep or the buffer print
        # so we stop before throttling playback further.
        did_interrupt = did_interrupt or Interrupts().did_interrupt
        if did_interrupt:
            Tts.clear_continuation()
            break

        # Sleep if necessary to prevent growing buffer beyond threshold
        if stream.buffer_duration > REAL_TIME_BUFFER_MAX_SECONDS and full_duration > 0.0:
            printt(f"{COL_DIM_ITALICS}Sleeping for {full_duration:.1f}s ...")
            did_interrupt = did_interrupt or sleep_interruptibly(full_duration)
            if did_interrupt:
                break

    # Finished. Preserve the legacy behavior: buffered audio continues until
    # the user presses Enter, at which point the stream is closed by start()'s
    # finally block and control returns to the menu.
    was_cancelled = Interrupts().did_interrupt
    Interrupts().clear()
    if not did_interrupt:
        RealTimePlaybackEvents.emit(RealTimePlaybackProgress(total, total))

    buffer_duration = stream.buffer_duration if stream else 0.0
    RealTimePlaybackEvents.emit(
        RealTimePlaybackAwaitingContinue(buffer_duration, did_interrupt)
    )
    printt()
    if continue_event is None:
        ask.ask_enter_to_continue()
    else:
        # The Textual app extrapolates buffer drain and the playhead on its
        # header refresh interval; no worker-side polling is needed here.
        continue_event.wait()

    # Cancellation can arrive while waiting for Enter (for example during
    # application shutdown), so sample the external event again afterwards.
    was_cancelled = was_cancelled or Interrupts().did_interrupt
    printt()

    if was_cancelled:
        status = RealTimePlaybackRunStatus.CANCELLED
    elif did_interrupt:
        status = RealTimePlaybackRunStatus.ABORTED
    else:
        status = RealTimePlaybackRunStatus.COMPLETED
    return RealTimePlaybackRunResult(status)


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
        gen_timeout_tracker: GenTimeoutTracker | None = None,
) -> tuple[Sound | None, bool, int]:
    """
    Similar to `GenerateUtil.generate_full_flow()` but simpler control flow.
    Returns tuple: (Sound or None if problem, did_interrupt, consecutive_model_errors)

    Params:
        gen_timeout_tracker:
            The run-shared GEN_TIMEOUT tracker; the run's very first gen is
            exempt from the watchdog (warm-up/download). When None, a fresh
            tracker is used (this call's first attempt is the exempt one).
    """

    if gen_timeout_tracker is None:
        gen_timeout_tracker = GenTimeoutTracker()

    Interrupts().set("generating")

    project = state.project
    phrase_group = phrase_groups[index]
    did_interrupt = False

    gen_result: ValidationResult | str = ""
    num_attempts = 1 + project.max_retries if has_runway else 1

    for attempt in range(num_attempts):

        with gen_timeout_tracker.scope() as gen_timeout_guard:
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
        if gen_timeout_guard.did_time_out:
            # The watchdog already reported the timeout and requested the
            # worker reset; abort the run regardless of any pending cancel.
            printt(f"{COL_ERROR}Aborting real-time playback")
            Tts.clear_continuation()
            did_interrupt = True
            break
        result = results[0]
        if isinstance(result, TtsModelError):
            consecutive_model_errors += 1
            gen_result = result.message
        else:
            consecutive_model_errors = 0
            gen_result = result

        if consecutive_model_errors >= max_consecutive_model_errors:
            GenerateUtil.print_consecutive_model_errors_message(max_consecutive_model_errors)
            GenerationEvents.emit(
                ModelUnhealthy(
                    reason=make_consecutive_model_errors_reset_message(max_consecutive_model_errors)
                )
            )
            Tts.clear_continuation()
            did_interrupt = True
            break

        # Check for OOM in results and break early to avoid wasting time
        if isinstance(gen_result, str) and is_oom_error_message(gen_result):
            print_gen_oom_message(gen_result)
            GenerationEvents.emit(ModelUnhealthy(reason=make_oom_reset_message()))
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
            printt(f"Transcript validation: {gen_result.get_ui_message_with_extras()}")

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
