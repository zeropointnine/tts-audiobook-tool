from __future__ import annotations

import itertools
import threading
import time
from dataclasses import dataclass
from typing import Callable

from tts_audiobook_tool.app_types import Segment, Sound
from tts_audiobook_tool.constants import APP_SAMPLE_RATE, WHISPER_SAMPLERATE
from tts_audiobook_tool.constants_config import (
    CHAT_SILENCE_THRESHOLD_CHUNKED,
    CHAT_SILENCE_THRESHOLD_IMMEDIATE,
)
from tts_audiobook_tool.conversation.chat_config import (
    get_resolved_system_prompt,
    is_immediate_input_mode,
    is_microphone_input_mode,
    is_text_input_mode,
)
from tts_audiobook_tool.conversation.conversation_types import (
    ChatInputMode,
    ChunkingConfig,
    ResponseResult,
)
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.conversation.prompt_draft import PromptSubmission
from tts_audiobook_tool.conversation.realtime_transcriber import RealtimeTranscriber
from tts_audiobook_tool.conversation.response_session import ResponseSession
from tts_audiobook_tool.conversation.sound_input_device_util import SoundInputDeviceInfo
from tts_audiobook_tool.l import L
from tts_audiobook_tool.model_worker import (
    ConsoleEventHandler,
    ModelWorker,
    discard_console_output,
)
from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.state import State
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@dataclass(frozen=True)
class ConversationInitialization:
    warnings: tuple[str, ...] = ()
    input_device: str = ""


# After an interrupted turn's playback is cleared, frames PortAudio already
# consumed still play out at the DAC. Before reopening the microphone, poll
# is_playback_complete at this interval, give up after this timeout, then let
# the output settle for this margin so the mic cannot pick up the tail of the
# assistant's speech.
OUTPUT_TAIL_POLL_INTERVAL_S = 0.02
OUTPUT_TAIL_WAIT_TIMEOUT_S = 1.0
OUTPUT_TAIL_SETTLE_MARGIN_S = 0.15


class ConversationRuntime:
    """Own long-lived LLM, microphone, and playback resources without a UI."""

    _session_ids = itertools.count(1)

    def __init__(
        self,
        state: State,
        *,
        # None keeps phrase alignment for microphone chat except with GLM,
        # whose tighter CUDA budget cannot afford redundant output STT. Text
        # chat never loads STT implicitly. An explicit bool remains available
        # for diagnostic callers that intentionally want to override this.
        phrase_stt_enabled: bool | None = None,
        on_transcription: Callable[[int, list[Segment], object | None], None]
        | None = None,
        on_transcription_error: Callable[[int, str], None] | None = None,
    ) -> None:
        self.state = state
        self.prefs = state.prefs
        self.project = state.project
        self.input_mode: ChatInputMode = state.prefs.chat_input_mode
        self.echo_override: bool = getattr(state.prefs, "chat_echo_override", False)
        self.phrase_stt_enabled = phrase_stt_enabled
        self.on_transcription = on_transcription
        self.on_transcription_error = on_transcription_error
        self.session_id = next(self._session_ids)
        self._lock = threading.RLock()
        self.initialized = False
        self.closing = False
        self.llm: LlmSession | None = None
        self.sound_stream: SoundDeviceStream | None = None
        self.transcriber: RealtimeTranscriber | None = None
        self.active_response: ResponseSession | None = None
        self.inspection = None
        self._worker_touched = False

    @property
    def is_text_input(self) -> bool:
        return is_text_input_mode(self.input_mode)

    @property
    def is_microphone_input(self) -> bool:
        return is_microphone_input_mode(self.input_mode)

    @property
    def is_immediate_input(self) -> bool:
        return is_immediate_input_mode(self.input_mode)

    @property
    def should_align_output_phrases(self) -> bool:
        """Whether generated assistant audio should be re-transcribed by STT."""
        if self.phrase_stt_enabled is not None:
            return self.phrase_stt_enabled
        return self.is_microphone_input and Tts.get_type() is not TtsModelType.GLM

    @property
    def input_level_db(self) -> float | None:
        """Return the latest microphone RMS level without blocking capture."""
        with self._lock:
            transcriber = self.transcriber
        return (
            transcriber.latest_input_level_db if transcriber is not None else None
        )

    def initialize(
        self, console_handler: ConsoleEventHandler | None = None
    ) -> ConversationInitialization:
        with self._lock:
            if self.initialized:
                warnings = tuple(getattr(self.inspection, "warnings", ()))
                device = (
                    SoundInputDeviceInfo.get_input_device_description()
                    if self.is_microphone_input
                    else ""
                )
                return ConversationInitialization(warnings, device)
            if self.closing:
                raise RuntimeError("Conversation is closing")

        self._worker_touched = True
        L.i(f"[chat] initialize: mode={self.input_mode.id} inspecting TTS")
        t0 = time.monotonic()
        # Warm up through the shared generation business logic so chat emits
        # the same worker init output ("Warming up models...", the STT model's
        # own init line) that the generation session shows.
        inspection, error = ModelWorker.inspect_tts_blocking(
            self.state,
            console_handler=console_handler,
            warm_models=True,
            warm_stt=self.is_microphone_input or self.should_align_output_phrases,
        )
        if error or inspection is None:
            raise RuntimeError(error or "Couldn't initialize TTS model")
        if inspection.blocking_issues:
            raise RuntimeError("\n".join(inspection.blocking_issues))

        llm = None
        if not self.echo_override:
            llm = LlmSession(
                api_endpoint_url=self.prefs.llm_url,
                token=self.prefs.llm_api_key,
                model=self.prefs.llm_model,
                system_prompt=get_resolved_system_prompt(self.state),
                extra_params=self.prefs.llm_extra_params,
                verbose=False,
            )
        reset_error = ModelWorker.reset_chat_session_blocking(
            console_handler=console_handler
        )
        if reset_error:
            raise RuntimeError(reset_error)
        L.i(f"[chat] initialize: worker ready in {time.monotonic() - t0:.1f}s")

        use_streaming = Tts.get_info().can_stream and self.project.streaming_chat
        sample_rate = (
            Tts.get_class().get_output_sample_rate(self.project)
            if use_streaming
            else APP_SAMPLE_RATE
        )
        sound_stream = SoundDeviceStream(sample_rate)
        if not sound_stream.start():
            sound_stream.shut_down()
            raise RuntimeError("Could not start the sound output device")

        transcriber = None
        device = ""
        if self.is_microphone_input:
            device = SoundInputDeviceInfo.get_input_device_description()
            silence = (
                CHAT_SILENCE_THRESHOLD_IMMEDIATE
                if self.is_immediate_input
                else CHAT_SILENCE_THRESHOLD_CHUNKED
            )
            session_id = self.session_id

            def transcription_callback(
                segments: list[Segment], audio: object | None = None
            ) -> None:
                with self._lock:
                    valid = not self.closing and self.session_id == session_id
                if valid and self.on_transcription is not None:
                    self.on_transcription(session_id, segments, audio)

            def transcription_error(message: str) -> None:
                with self._lock:
                    valid = not self.closing and self.session_id == session_id
                if valid and self.on_transcription_error is not None:
                    self.on_transcription_error(session_id, message)

            transcriber = RealtimeTranscriber(
                prefs=self.prefs,
                on_transcription=transcription_callback,
                silence_duration_s=silence,
                on_error=transcription_error,
            )
            try:
                transcriber.start()
            except Exception:
                sound_stream.shut_down()
                raise

        with self._lock:
            if self.closing:
                if transcriber is not None:
                    transcriber.stop()
                sound_stream.shut_down()
                if llm is not None:
                    llm.clear()
                raise RuntimeError("Conversation closed during initialization")
            self.inspection = inspection
            self.llm = llm
            self.sound_stream = sound_stream
            self.transcriber = transcriber
            self.initialized = True
        llm_description = (
            "echo override"
            if self.echo_override
            else f"{self.prefs.llm_model} @ {self.prefs.llm_url}"
        )
        L.i(
            f"[chat] initialize: complete in {time.monotonic() - t0:.1f}s"
            f" (llm={llm_description}, mic={'on' if transcriber is not None else 'off'})"
        )
        return ConversationInitialization(tuple(inspection.warnings), device)

    def create_response(
        self, on_error: Callable[[str], None] | None = None
    ) -> ResponseSession:
        with self._lock:
            if (
                not self.initialized
                or self.sound_stream is None
                or (self.llm is None and not self.echo_override)
            ):
                raise RuntimeError("Conversation is not initialized")
            if self.active_response is not None:
                raise RuntimeError("A response is already active")
            response = ResponseSession(
                llm=self.llm,
                state=self.state,
                chunking_config=ChunkingConfig(
                    language_code=self.project.language_code,
                    max_words=self.project.max_words,
                ),
                stt_variant=self.prefs.stt_variant,
                stt_config=self.prefs.stt_config,
                sound_stream=self.sound_stream,
                phrase_stt_enabled=self.should_align_output_phrases,
                echo_override=self.echo_override,
                on_error=on_error,
            )
            self.active_response = response
            return response

    def run_response(
        self, response: ResponseSession, submission: PromptSubmission
    ) -> ResponseResult:
        L.i(f"[chat] response turn starting: chars={len(submission.text)}")
        t0 = time.monotonic()
        transcriber = self.transcriber
        stream = self.sound_stream
        # Assume the turn ends messily until it finishes cleanly; on an
        # exception the finally block still settles playback before resume.
        interrupted = True
        try:
            if transcriber is not None:
                L.i("[chat] pausing transcriber before response")
                # Inside the try: a stuck-transcriber timeout must still
                # clear active_response and resume the mic in the finally.
                transcriber.pause()
            if submission.audio is not None and self.prefs.chat_save_mic:
                response.user_input_sound = Sound(submission.audio, WHISPER_SAMPLERATE)
                response.save_chat_mic_input_if_needed(submission.text)
            result = response.run(
                submission.text,
                user_input_sound=(
                    Sound(submission.audio, WHISPER_SAMPLERATE)
                    if submission.audio is not None
                    else None
                ),
            )
            interrupted = result.interrupted
            L.i(
                f"[chat] response turn finished in {time.monotonic() - t0:.1f}s:"
                f" interrupted={result.interrupted} chars={len(result.text)}"
            )
            if transcriber is not None and not interrupted:
                transcriber.flush()
                latency = stream.output_latency if stream is not None else 0.0
                time.sleep(max(0.25, min(0.6, latency + 0.1)))
                transcriber.flush()
            return result
        finally:
            with self._lock:
                if self.active_response is response:
                    self.active_response = None
                should_resume = not self.closing
            if transcriber is not None and should_resume:
                if interrupted:
                    self._wait_for_output_tail_to_settle(stream)
                    transcriber.flush()
                transcriber.resume()

    @staticmethod
    def _wait_for_output_tail_to_settle(stream: SoundDeviceStream | None) -> None:
        """Wait for cleared-but-still-scheduled playback to finish at the DAC.

        clear_buffer() drops queued audio, but frames PortAudio already
        consumed keep playing for up to the output block size plus device
        latency. Reopening the microphone before that tail ends lets it
        transcribe the last word(s) of the assistant's speech.
        """
        if stream is None:
            return
        deadline = time.monotonic() + OUTPUT_TAIL_WAIT_TIMEOUT_S
        while not stream.is_playback_complete and time.monotonic() < deadline:
            time.sleep(OUTPUT_TAIL_POLL_INTERVAL_S)
        time.sleep(OUTPUT_TAIL_SETTLE_MARGIN_S)

    def request_interrupt(self) -> None:
        with self._lock:
            response = self.active_response
        if response is not None:
            L.i("[chat] interrupt requested for active response")
            response.request_interrupt()

    def _detach_resources_for_close(
        self,
    ) -> tuple[
        bool,
        ResponseSession | None,
        RealtimeTranscriber | None,
        SoundDeviceStream | None,
        LlmSession | None,
    ]:
        """Mark the runtime closing and atomically detach owned resources."""
        with self._lock:
            first_close = not self.closing
            self.closing = True
            response = self.active_response
            transcriber = self.transcriber
            sound_stream = self.sound_stream
            llm = self.llm
            self.active_response = None
            self.transcriber = None
            self.sound_stream = None
            self.llm = None
            self.initialized = False
        return first_close, response, transcriber, sound_stream, llm

    @staticmethod
    def _dispose_resources(
        response: ResponseSession | None,
        transcriber: RealtimeTranscriber | None,
        sound_stream: SoundDeviceStream | None,
        llm: LlmSession | None,
    ) -> None:
        if response is not None:
            response.request_interrupt()
        if transcriber is not None:
            try:
                transcriber.stop()
            except Exception:
                pass
        if sound_stream is not None:
            try:
                sound_stream.shut_down()
            except Exception:
                pass
        if llm is not None:
            llm.clear()

    def cancel_initialization(self) -> str:
        """Stop initialization and hard-reset its model-worker process."""
        resources = self._detach_resources_for_close()
        first_close, response, transcriber, sound_stream, llm = resources
        if not first_close:
            return ""
        L.i("[chat] cancelling initialization and hard-resetting model worker")
        try:
            return ModelWorker.reset()
        finally:
            self._dispose_resources(response, transcriber, sound_stream, llm)

    def close(self) -> None:
        resources = self._detach_resources_for_close()
        first_close, response, transcriber, sound_stream, llm = resources
        if not first_close:
            return
        L.i("[chat] closing conversation runtime")
        self._dispose_resources(response, transcriber, sound_stream, llm)
        if self._worker_touched:
            _ = ModelWorker.reset_chat_session_blocking(
                console_handler=discard_console_output
            )
