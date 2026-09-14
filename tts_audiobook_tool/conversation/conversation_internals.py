from __future__ import annotations

import os
import queue
import re
import threading
import time
from typing import Callable, Protocol

import numpy as np

from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.conversation.conversation_types import ChunkingConfig, ResponseSnapshot
from tts_audiobook_tool.app_types.force_align_util import ForceAlignUtil
from tts_audiobook_tool.conversation.llm_session import LlmSession
from tts_audiobook_tool.l import L
from tts_audiobook_tool.model_worker import ModelWorker, discard_console_output
from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter
from tts_audiobook_tool.text_ops.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.app_types.phrase import Reason
from tts_audiobook_tool.app_support import app_text
from tts_audiobook_tool.sound.sound_pipeline import SoundPipeline
from tts_audiobook_tool.sound.sound_file_util import SoundFileUtil
from tts_audiobook_tool.sound.sound_util import SoundUtil
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.sound.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.app_types.timed_phrase import TimedPhrase
from tts_audiobook_tool.transcriber import Transcriber
from tts_audiobook_tool.util import make_error_string


class ResponseUi(Protocol):
    """Minimal UI-neutral sink used by the response engine."""

    def println(self, text: str = "") -> None: ...


class ConversationStreamingTts:
    @staticmethod
    def generate_to_sound_stream(
        state: State,
        text: str,
        reason: Reason,
        sound_stream: SoundDeviceStream,
        interrupt_requested: threading.Event,
        on_segment_range: Callable[[int, int], None] | None = None,
    ) -> tuple[tuple[int, int] | None, Sound | None, str | None]:
        """
        Stream one TTS chunk directly into the conversation output stream.

        Returns:
            ((start_sample, end_sample), saved_sound, None) on success, where the range covers
            the full streamed audio plus any appended trailing silence.
            (None, None, error_string) on failure.
        """
        project = state.project
        sample_rate = Tts.get_class().get_output_sample_rate(project)
        if sound_stream.sample_rate != sample_rate:
            return (
                None,
                None,
                (
                    f"Streaming sample rate mismatch: output stream={sound_stream.sample_rate}, "
                    f"tts={sample_rate}"
                ),
            )

        stream_start: int | None = None
        stream_end: int | None = None
        speech_end: int | None = None
        saved_chunks: list[np.ndarray] = []

        def append_chunk(data: np.ndarray) -> None:
            nonlocal stream_start, stream_end, speech_end
            if interrupt_requested.is_set():
                return
            saved_chunks.append(np.copy(data))
            start, end = sound_stream.add_data(data)
            if stream_start is None:
                stream_start = start
            stream_end = end
            speech_end = end
            if (
                on_segment_range is not None
                and stream_start is not None
                and speech_end is not None
            ):
                on_segment_range(stream_start, speech_end)

        def on_stream_end() -> None:
            nonlocal stream_start, stream_end
            if interrupt_requested.is_set():
                return

            # Streaming path intentionally appends only silence here. We do not
            # support the normal full-sound post-processing/effect path, and we
            # intentionally do not play section sound effects on this branch.
            pause_duration = project.reason_pauses.get_pause_for(reason)
            if pause_duration <= 0:
                return

            silence_sound = SoundUtil.make_silence_sound(
                seconds=pause_duration,
                sr=sound_stream.sample_rate,
                dtype=np.dtype(np.float32),
            )
            saved_chunks.append(np.copy(silence_sound.data))
            start, end = sound_stream.add_data(silence_sound.data)
            if stream_start is None:
                stream_start = start
            stream_end = end

        _, error = ModelWorker.synthesize_chat_blocking(
            state,
            text,
            reason,
            streaming=True,
            on_chunk=append_chunk,
            interrupt_event=interrupt_requested,
            console_handler=discard_console_output,
        )
        if not error:
            on_stream_end()
        if error:
            return None, None, error
        if stream_start is None or speech_end is None or speech_end <= stream_start:
            return None, None, "No streamed audio output"

        saved_sound = None
        if saved_chunks:
            saved_sound = Sound(np.concatenate(saved_chunks), sound_stream.sample_rate)

        return (stream_start, speech_end), saved_sound, None


class _ResponseEngine:
    """
    Handles one LLM→TTS response turn. Construct fresh per Enter-press.
    Owns all per-turn threading state. The sound_stream is injected by
    the caller (Conversation) and persists across response turns.
    """

    RESPONSE_PLACEHOLDER = "..."
    REASONING_PLACEHOLDER = "(thinking...)"

    def __init__(
        self,
        ui: ResponseUi,
        llm: LlmSession | None,
        state: State,
        chunking_config: ChunkingConfig,
        stt_variant: SttVariant,
        stt_config: SttConfig,
        sound_stream: SoundDeviceStream,
        phrase_stt_enabled: bool = True,
        echo_override: bool = False,
    ) -> None:
        self.ui = ui
        self.llm = llm
        self.state = state
        self.prefs = state.prefs
        self.project = state.project
        self.chunking_config = chunking_config
        self.stt_variant = stt_variant
        self.stt_config = stt_config
        self.phrase_stt_enabled = phrase_stt_enabled
        self.echo_override = echo_override
        self.sound_stream: SoundDeviceStream = sound_stream
        self.user_input_sound: Sound | None = None

        # Per-turn threading state is created at construction, not in run(),
        # so an interrupt requested before run() (or during construction
        # races) persists into the turn instead of being wiped by run().
        self.state_lock = threading.Lock()
        self.tts_q: queue.Queue[tuple[str, Reason] | None] = queue.Queue()
        self.interrupt_requested = threading.Event()
        self.spoken_segments: list[tuple[str, int, int]] = []
        self.saved_turn_sounds: list[Sound] = []
        self.pending_sentences: list[str] = []
        self.tts_buffer = ""
        self.render_buffer = _ResponseEngine.RESPONSE_PLACEHOLDER
        self.playback_done = False
        self.worker: threading.Thread | None = None
        self.llm_content_received = False
        self.first_audio_latency_logged = False
        self.output_turn_tts_started_at: float | None = None
        self.output_turn_tts_mode: str | None = None
        self.output_turn_tts_preview_text: str = ""

    def snapshot(self) -> ResponseSnapshot:
        """Read immutable turn state before, during, or after run().

        The engine owns synchronization of its text state. Playback position
        and lifecycle flags are sampled independently, not atomically with text.
        """
        with self.state_lock:
            segments = tuple(self.spoken_segments)
            pending = tuple(self.pending_sentences)
            render_buffer = self.render_buffer
            received = self.llm_content_received
        return ResponseSnapshot(
            spoken_segments=segments,
            pending_sentences=pending,
            render_buffer=render_buffer,
            play_position_samples=self.sound_stream.play_position_samples,
            playback_done=self.playback_done,
            llm_content_received=received,
            interrupted=self.interrupt_requested.is_set(),
        )

    def request_interrupt(self) -> None:
        """Request cooperative cancellation from any thread, even before run().

        Repeated requests are safe. A pre-run request survives turn startup.
        The engine stops generation now and clears buffered playback only when
        the turn settles; in-flight TTS inference cannot be interrupted.
        """
        self._signal_interrupt()

    def _signal_interrupt(self) -> None:
        """Single interrupt primitive: stop the LLM and the TTS worker.

        Idempotent and safe from any thread at any point of the turn
        lifecycle, including before run(). Covers, in one place, what used
        to be five separate mechanisms: the interrupt event, the tts_q
        poison pill, and the HTTP stream cancellation.

        The playback buffer is deliberately NOT cleared here: a turn with a
        TTS inference already in flight cannot converge until that inference
        finishes (the model call is uninterruptible), so the session would
        sit in its waiting state in silence if the audio stopped now. Queued
        audio keeps playing; the buffer is cleared by _settle_interrupted_turn()
        once the turn has actually converged, which is also when the session
        returns to input (in microphone mode, capture cannot resume over
        assistant audio, so the clear must precede the resume).
        """
        self.interrupt_requested.set()
        # Poison-pill the TTS worker: drain queued sentences so a blocked
        # get() unblocks and a mid-synthesis worker finds an empty queue.
        while True:
            try:
                self.tts_q.get_nowait()
            except queue.Empty:
                break
        self.tts_q.put(None)
        if self.llm is not None:
            self.llm.cancel_active_request()

    def _settle_interrupted_turn(self) -> None:
        """Converge an interrupted/failed turn once generation has returned.

        Re-signs the interrupt and waits for in-flight synthesis to finish
        before clearing playback. Terminal reset must not race an active model
        operation, and no TTS worker may enqueue audio after the clear.
        """
        self._signal_interrupt()
        if self.worker is not None and self.worker.ident is not None:
            self.worker.join()
        self.sound_stream.clear_buffer()

    def _reset_chat_session(self, phase: str) -> None:
        """Reset continuation state without changing the selected voice.

        Start/end resets isolate turns; recovery resets allow subsequent chunks
        to proceed after a TTS failure. Failed resets remain visible in logs.
        """
        error = ModelWorker.reset_chat_session_blocking(
            reset_voice_selection=False, console_handler=discard_console_output
        )
        if error:
            L.w(f"[chat] engine: {phase} reset failed: {error}")

    def run(self, assembled: str, user_input_sound: Sound | None = None) -> None:
        """Own terminal cleanup for every outcome, including startup failures."""
        try:
            self._run_turn(assembled, user_input_sound)
        except BaseException:
            self._settle_interrupted_turn()
            raise
        finally:
            self.playback_done = True
            self._reset_chat_session("turn-end")

    def _run_turn(self, assembled: str, user_input_sound: Sound | None) -> None:
        # Each Enter-press/LLM response turn is its own rolling-continuation
        # context. Continuation may still bridge generated chunks within this
        # turn, but must not leak across independent turns.
        t0 = time.monotonic()
        L.i("[chat] engine: turn start, resetting chat session")
        self._reset_chat_session("turn-start")

        # Reset per-turn content state. The interrupt event, lock, and queue
        # live from construction and are intentionally left untouched so a
        # pre-run interrupt (including its poison pill) survives into the
        # turn.
        self.user_input_sound = user_input_sound
        with self.state_lock:
            self.tts_buffer = ""
            self.render_buffer = _ResponseEngine.RESPONSE_PLACEHOLDER
            self.spoken_segments = []
            self.pending_sentences = []
            self.llm_content_received = False
        self.saved_turn_sounds = []
        self.playback_done = False
        self.first_audio_latency_logged = False
        self.output_turn_tts_started_at = None
        self.output_turn_tts_mode = None
        self.output_turn_tts_preview_text = ""

        self.worker = threading.Thread(target=self.tts_worker, daemon=True)
        self.worker.start()

        llm_failed = False
        try:
            if self.echo_override:
                L.i(f"[chat] engine: echoing input ({len(assembled)} chars)")
                self.on_chunk(assembled)
            else:
                if self.llm is None:
                    raise RuntimeError("LLM session is unavailable")
                L.i(f"[chat] engine: sending LLM request ({len(assembled)} chars)")
                self.llm.send(
                    assembled,
                    on_chunk=self.on_chunk,
                    interrupt_event=self.interrupt_requested,
                    on_reasoning=self.on_reasoning_chunk,
                )
            L.i(
                f"[chat] engine: response text ready in {time.monotonic() - t0:.1f}s,"
                f" flushing {len(self.pending_sentences)} pending chunk(s) to TTS"
            )

            final_text = ""
            with self.state_lock:
                if self.tts_buffer.strip():
                    final_text = self.tts_buffer
                    self.pending_sentences.append(self.tts_buffer)
                self.tts_buffer = ""
                self.render_buffer = ""
            if final_text:
                self.tts_q.put((final_text, Reason.SENTENCE))
            self.tts_q.put(None)
            self.worker.join()
            L.i(f"[chat] engine: TTS worker done in {time.monotonic() - t0:.1f}s, waiting for playback")
            wait_logged = time.monotonic()
            while (
                not self.sound_stream.is_playback_complete
                and not self.interrupt_requested.is_set()
            ):
                time.sleep(0.02)
                now = time.monotonic()
                if now - wait_logged > 10.0:
                    L.w(f"[chat] engine: playback still incomplete after {now - t0:.1f}s")
                    wait_logged = now
        except Exception as e:
            llm_failed = True
            L.e(f"[chat] engine: LLM turn failed: {make_error_string(e)}")
            with self.state_lock:
                self.pending_sentences.clear()
                self.tts_buffer = ""
                self.render_buffer = ""
            self.ui.println(f"[LLM error: {make_error_string(e)}]")

        was_interrupted = self.interrupt_requested.is_set()
        if llm_failed or was_interrupted:
            L.i(f"[chat] engine: aborting response (failed={llm_failed}, interrupted={was_interrupted})")
            self._settle_interrupted_turn()

        self.playback_done = True
        if not llm_failed and not was_interrupted:
            self.save_chat_output_if_needed()
        L.i(f"[chat] engine: turn complete in {time.monotonic() - t0:.1f}s")

    def tts_worker(self) -> None:
        while True:
            item = self.tts_q.get()
            if item is None:
                break
            text, reason = item
            if self.interrupt_requested.is_set():
                # Cancellation cleanup belongs to run(), after this worker exits.
                continue
            if not text.strip():
                with self.state_lock:
                    if self.pending_sentences and self.pending_sentences[0] == text:
                        self.pending_sentences.pop(0)
                continue
            try:
                if self.output_turn_tts_started_at is None:
                    self.output_turn_tts_started_at = time.monotonic()

                if Tts.get_info().can_stream and self.project.streaming_chat:
                    streamed_segment_idx: int | None = None
                    first_audio_callback_registered = False
                    self.log_tts_inference_start(mode="streaming", text=text)

                    def on_segment_range(start: int, end: int) -> None:
                        nonlocal streamed_segment_idx, first_audio_callback_registered
                        if (
                            not first_audio_callback_registered
                            and not self.first_audio_latency_logged
                        ):
                            first_audio_callback_registered = True
                            self.output_turn_tts_mode = "streaming"
                            if not self.output_turn_tts_preview_text:
                                self.output_turn_tts_preview_text = text
                            self.sound_stream.set_first_audio_output_callback(
                                start,
                                self.log_tts_first_audio_latency,
                            )
                        with self.state_lock:
                            if (
                                self.pending_sentences
                                and self.pending_sentences[0] == text
                            ):
                                self.pending_sentences.pop(0)
                            if streamed_segment_idx is None:
                                self.spoken_segments.append((text, start, end))
                                streamed_segment_idx = len(self.spoken_segments) - 1
                            else:
                                self.spoken_segments[streamed_segment_idx] = (
                                    text,
                                    start,
                                    end,
                                )

                    stream_range, saved_sound, err = (
                        ConversationStreamingTts.generate_to_sound_stream(
                            state=self.state,
                            text=text,
                            reason=reason,
                            sound_stream=self.sound_stream,
                            interrupt_requested=self.interrupt_requested,
                            on_segment_range=on_segment_range,
                        )
                    )

                    if self.interrupt_requested.is_set():
                        # Keep the active sentence in the response snapshot.
                        # It was received from the LLM even though its TTS work
                        # was interrupted before completing.
                        continue

                    if err is not None:
                        self._reset_chat_session("TTS recovery")
                        if not self.interrupt_requested.is_set():
                            self.ui.println(
                                f"[TTS error: {err}]"
                            )
                        with self.state_lock:
                            if (
                                self.pending_sentences
                                and self.pending_sentences[0] == text
                            ):
                                self.pending_sentences.pop(0)
                        continue

                    assert stream_range is not None
                    if saved_sound is not None:
                        self.saved_turn_sounds.append(saved_sound)
                    start, end = stream_range
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                        if streamed_segment_idx is None:
                            self.spoken_segments.append((text, start, end))
                        else:
                            self.spoken_segments[streamed_segment_idx] = (
                                text,
                                start,
                                end,
                            )
                    continue

                self.log_tts_inference_start(mode="non-streaming", text=text)
                result, tts_error = ModelWorker.synthesize_chat_blocking(
                    self.state,
                    text,
                    reason,
                    streaming=False,
                    interrupt_event=self.interrupt_requested,
                    console_handler=discard_console_output,
                )
                if self.interrupt_requested.is_set():
                    # Preserve the accepted LLM text for the final transcript;
                    # only its audio generation was interrupted.
                    continue
                if tts_error or result is None:
                    if not self.interrupt_requested.is_set():
                        self.ui.println(
                            f"[TTS error: {tts_error}]"
                        )
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                    continue
                sound = result[0]
                if self.interrupt_requested.is_set():
                    # Preserve the accepted LLM text for the final transcript;
                    # only its audio generation was interrupted.
                    continue

                if sound.data.size == 0:
                    self._reset_chat_session("TTS recovery")
                    if not self.interrupt_requested.is_set():
                        self.ui.println(
                            "[TTS error: empty/silent output]"
                        )
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                    continue

                sound = SoundPipeline.prepare_generated_sound_for_playback(
                    sound,
                    high_shelf=self.project.get_high_shelf(),
                    limit_silence_gaps=self.project.limit_silence_gaps,
                    limit_silence_gaps_duration=self.project.limit_silence_gaps_duration,
                )
                sound = SoundPipeline.append_pause_or_section_effect(
                    sound,
                    reason=reason,
                    reason_pauses=self.project.reason_pauses,
                    break_effect=None,
                )
                self.saved_turn_sounds.append(sound)

                if not self.interrupt_requested.is_set():
                    start, end = self.sound_stream.add_data(sound.data)
                    if not self.first_audio_latency_logged:
                        self.output_turn_tts_mode = "non-streaming"
                        if not self.output_turn_tts_preview_text:
                            self.output_turn_tts_preview_text = text
                        self.sound_stream.set_first_audio_output_callback(
                            start,
                            self.log_tts_first_audio_latency,
                        )
                    is_first_tts_chunk = not self.spoken_segments
                    spoken_segments = self.make_spoken_segments(
                        text, sound, start, end, is_first_tts_chunk
                    )
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                        self.spoken_segments.extend(spoken_segments)
                # On interruption, preserve accepted text; run() owns cleanup.

            except Exception as e:
                interrupted = self.interrupt_requested.is_set()
                if not interrupted:
                    self._reset_chat_session("TTS recovery")
                    self.ui.println(f"[TTS exception: {e}]")
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)

    def log_tts_first_audio_latency(self) -> None:
        if self.first_audio_latency_logged or self.output_turn_tts_started_at is None:
            return
        self.first_audio_latency_logged = True
        elapsed_ms = (time.monotonic() - self.output_turn_tts_started_at) * 1000.0
        preview = re.sub(r"\s+", " ", self.output_turn_tts_preview_text).strip()
        if len(preview) > 80:
            preview = preview[:80] + "..."
        L.i(
            f"TTS first-audio latency ({self.output_turn_tts_mode or 'unknown'}): {elapsed_ms:.1f} ms | "
            f"chars={len(self.output_turn_tts_preview_text)} | text='{preview}'"
        )

    def log_tts_inference_start(self, mode: str, text: str) -> None:
        preview = re.sub(r"\s+", " ", text).strip()
        if len(preview) > 80:
            preview = preview[:80] + "..."
        L.i(f"TTS inference start ({mode}) | chars={len(text)} | text='{preview}'")

    def on_reasoning_chunk(self, delta: str) -> None:
        """Show a thinking indicator while a reasoning model works before content."""
        del delta
        if self.interrupt_requested.is_set():
            return
        with self.state_lock:
            if self.llm_content_received:
                return
            if self.render_buffer != _ResponseEngine.REASONING_PLACEHOLDER:
                L.i("[chat] engine: reasoning phase started, showing thinking indicator")
            self.render_buffer = _ResponseEngine.REASONING_PLACEHOLDER

    def on_chunk(self, delta: str) -> None:
        if self.interrupt_requested.is_set():
            return

        to_send: list[tuple[str, Reason]] = []
        use_streaming_tts = Tts.get_info().can_stream and self.project.streaming_chat
        with self.state_lock:
            if not self.llm_content_received:
                L.i(f"[chat] engine: first LLM content received ({len(delta)} chars)")
            if (
                not self.llm_content_received
                and self.render_buffer == _ResponseEngine.RESPONSE_PLACEHOLDER
            ):
                self.render_buffer = ""
            self.llm_content_received = True
            complete_chunks, self.tts_buffer, self.render_buffer = (
                _ResponseEngine.consume_tts_delta(
                    tts_buffer=self.tts_buffer,
                    delta=delta,
                    config=self.chunking_config,
                    has_pending_sentences=bool(self.pending_sentences),
                    has_spoken_segments=bool(self.spoken_segments),
                    allow_first_chunk_latency_split=not use_streaming_tts,
                )
            )
            for s, reason in complete_chunks:
                self.pending_sentences.append(s)
                to_send.append((s, reason))
            if to_send:
                self.render_buffer = ""
        for s, reason in to_send:
            self.tts_q.put((s, reason))

    def save_chat_output_if_needed(self) -> None:
        if (
            not self.prefs.chat_save
            or not self.project.dir_path
            or not self.saved_turn_sounds
        ):
            return

        dir_path = os.path.join(self.project.dir_path, PROJECT_CHAT_OUTPUT_SUBDIR)
        os.makedirs(dir_path, exist_ok=True)

        sound = self.saved_turn_sounds[0]
        if len(self.saved_turn_sounds) > 1:
            sound_data = np.concatenate([item.data for item in self.saved_turn_sounds])
            sound = Sound(sound_data, sound.sr)

        file_path = os.path.join(dir_path, self.make_chat_file_name())
        err = SoundFileUtil.save_flac(sound, file_path)
        if err:
            self.ui.println(f"[Save error: {err}]")
            return

    def make_chat_file_name(self) -> str:
        timestamp = SoundSegmentUtil.make_timestamp_string()
        model = Tts.get_info().file_tag
        voice = Tts.get_class().get_voice_tag(self.project)
        text = " " + app_text.sanitize_for_filename(self.render_response_text()[:50])
        return f"[{timestamp}] [chat] [{model}] [{voice}]{text}.flac"

    def save_chat_mic_input_if_needed(self, assembled: str) -> None:
        if (
            not self.prefs.chat_save_mic
            or not self.project.dir_path
            or self.user_input_sound is None
        ):
            return

        dir_path = os.path.join(self.project.dir_path, PROJECT_CHAT_OUTPUT_SUBDIR)
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, self.make_chat_mic_file_name(assembled))
        err = SoundFileUtil.save_flac(self.user_input_sound, file_path)
        if err:
            self.ui.println(f"[Save error: {err}]")
            return

    def make_chat_mic_file_name(self, assembled: str) -> str:
        timestamp = SoundSegmentUtil.make_timestamp_string()
        text = " " + app_text.sanitize_for_filename(assembled[:50])
        return f"[{timestamp}] [chat] [mic]{text}.flac"

    def render_response_text(self) -> str:
        parts: list[str] = []
        with self.state_lock:
            parts.extend(text for text, _, _ in self.spoken_segments)
            parts.extend(self.pending_sentences)
            if self.tts_buffer.strip():
                parts.append(self.tts_buffer)
            elif (
                self.render_buffer.strip()
                and self.render_buffer
                not in (
                    _ResponseEngine.RESPONSE_PLACEHOLDER,
                    _ResponseEngine.REASONING_PLACEHOLDER,
                )
            ):
                parts.append(self.render_buffer)
        return " ".join(part.strip() for part in parts if part.strip())

    def make_spoken_segments(
        self,
        text: str,
        sound: Sound,
        stream_start: int,
        stream_end: int,
        is_first_tts_chunk: bool = False,
    ) -> list[tuple[str, int, int]]:
        if self.phrase_stt_enabled and not is_first_tts_chunk:
            phrase_segments = self.make_phrase_spoken_segments(
                text, sound, stream_start, stream_end
            )
            if phrase_segments:
                return phrase_segments
        return [(text, stream_start, stream_end)]

    def make_phrase_spoken_segments(
        self,
        text: str,
        sound: Sound,
        stream_start: int,
        stream_end: int,
    ) -> list[tuple[str, int, int]]:
        try:
            if not text.strip():
                return []

            phrases = PhraseSegmenter.text_to_phrases(
                text,
                max_words=self.chunking_config.max_words,
                pysbd_lang=self.chunking_config.language_code,
            )
            if len(phrases) <= 1:
                return []

            words = Transcriber.transcribe_to_words(
                sound,
                self.project.language_code,
                self.stt_variant,
                self.stt_config,
                self.state,
                console_handler=discard_console_output,
            )
            if isinstance(words, str) or not words:
                return []

            timed_phrases = ForceAlignUtil.make_timed_phrases(
                phrases, words, sound.duration
            )
            return self.timed_phrases_to_spoken_segments(
                timed_phrases, sound.sr, stream_start, stream_end
            )
        except Exception:
            return []

    def timed_phrases_to_spoken_segments(
        self,
        timed_phrases: list[TimedPhrase],
        sample_rate: int,
        stream_start: int,
        stream_end: int,
    ) -> list[tuple[str, int, int]]:
        if not timed_phrases:
            return []

        segments: list[tuple[str, int, int]] = []
        last_end = stream_start

        for i, timed_phrase in enumerate(timed_phrases):
            text = timed_phrase.text
            if not text.strip():
                continue

            start = stream_start + int(round(timed_phrase.time_start * sample_rate))
            end = stream_start + int(round(timed_phrase.time_end * sample_rate))
            start = max(stream_start, min(start, stream_end))
            end = max(stream_start, min(end, stream_end))
            start = max(start, last_end)

            if i == len(timed_phrases) - 1:
                end = stream_end
            elif end < start:
                end = start

            if end <= start:
                continue

            segments.append((text, start, end))
            last_end = end

        if not segments:
            return []

        final_text, final_start, _ = segments[-1]
        segments[-1] = (final_text, final_start, stream_end)
        return segments

    @staticmethod
    def make_display_text(text: str) -> str:
        # Preserve intended line breaks and normalize CRLF/CR to LF for consistent UI line counting.
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def segment_text_to_chunks(
        text: str, config: ChunkingConfig
    ) -> list[tuple[str, Reason]]:
        groups = PhraseGrouper.text_to_groups(
            text=text,
            max_words=config.max_words,
            strategy=config.strategy,
            pysbd_lang=config.language_code,
        )
        return [
            (group.text, group.last_reason) for group in groups if group.text.strip()
        ]

    @staticmethod
    def extract_complete_chunks(
        text: str, config: ChunkingConfig
    ) -> tuple[list[tuple[str, Reason]], str]:
        sentences = PhraseSegmenter.string_to_sentence_strings(
            text, config.language_code
        )
        if len(sentences) < 2:
            return [], text
        complete_text = "".join(sentences[:-1])
        remainder = sentences[-1]
        return _ResponseEngine.segment_text_to_chunks(complete_text, config), remainder

    @staticmethod
    def make_stable_chunk_preview(text: str, config: ChunkingConfig) -> str:
        chunks = _ResponseEngine.segment_text_to_chunks(text, config)
        if len(chunks) >= 2:
            return "".join(t for t, _ in chunks[:-1])
        return ""

    @staticmethod
    def consume_tts_delta(
        tts_buffer: str,
        delta: str,
        config: ChunkingConfig,
        has_pending_sentences: bool,
        has_spoken_segments: bool,
        allow_first_chunk_latency_split: bool = True,
    ) -> tuple[list[tuple[str, Reason]], str, str]:
        """
        Fold a newly streamed LLM delta into the TTS buffer.

        Returns:
        - chunks ready to send to TTS (each paired with its Reason for pause/silence)
        - updated tts_buffer remainder
        - render_buffer preview text

        Important invariant: do not eagerly emit the very first chunk when the
        current buffer is still a single chunk. Otherwise a stream pause after a
        word fragment like "to" can cause the next delta starting with " ask"
        to be spoken/displayed as "toask".
        """
        next_buffer = tts_buffer + delta
        full_buffer = next_buffer
        to_send, next_buffer = _ResponseEngine.extract_complete_chunks(
            next_buffer, config
        )

        is_first_chunk = not has_pending_sentences and not has_spoken_segments

        if to_send:
            if is_first_chunk and sum(len(t.split()) for t, _ in to_send) < 5:
                # First chunk must be >= 5 words. Hold the boundary, restore full buffer.
                to_send = []
                next_buffer = full_buffer
            else:
                return to_send, next_buffer, ""

        if is_first_chunk and allow_first_chunk_latency_split:
            # Use phrase-level boundaries (commas, semicolons, etc.) rather than
            # PhraseGrouper, which merges short sentences and would hide the
            # boundary inside a single group. Cut at the smallest phrase prefix
            # with >= 5 words, leaving a non-empty remainder.
            phrases = PhraseSegmenter.text_to_phrases(
                next_buffer,
                max_words=config.max_words,
                pysbd_lang=config.language_code,
            )
            if len(phrases) >= 2:
                cumulative_text = ""
                cumulative_words = 0
                for i in range(len(phrases) - 1):
                    cumulative_text += phrases[i].text
                    cumulative_words += phrases[i].num_words
                    if cumulative_words >= 5:
                        remainder = "".join(p.text for p in phrases[i + 1 :])
                        if remainder.strip():
                            return [(cumulative_text, phrases[i].reason)], remainder, ""
                        break

        render_buffer = _ResponseEngine.make_stable_chunk_preview(next_buffer, config)
        return [], next_buffer, render_buffer
