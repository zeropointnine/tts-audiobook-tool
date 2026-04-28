from __future__ import annotations

import logging
import os
import queue
import re
import sys
import threading
import time
import unicodedata

from faster_whisper.transcribe import Segment

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.app_types import Sound, SttConfig, SttVariant
from tts_audiobook_tool.constants import COL_ACCENT, COL_DIM, COL_MEDIUM, COL_OK
from tts_audiobook_tool.conversation.console_session import (
    ConsoleSession,
    KEY_CTRL_C,
    KEY_DEL,
    KEY_DEL2,
    KEY_ENTER,
    KEY_FWDDEL,
    KEY_LEFT,
    KEY_RIGHT,
)
from tts_audiobook_tool.conversation.conversation_types import ChunkingConfig, QueuedStream, UiOp
from tts_audiobook_tool.force_align_util import ForceAlignUtil
from tts_audiobook_tool.llm_util import LlmUtil
from tts_audiobook_tool.phrase_segmenter import PhraseSegmenter
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.timed_phrase import TimedPhrase
from tts_audiobook_tool.whisper_util import WhisperUtil
from tts_audiobook_tool.util import make_error_string


class Ui:
    """
    Single-thread terminal UI dispatcher.

    Serializes all stdout writes through one worker thread so cursor-relative
    renders never race with line-buffered prints from other threads. Public
    methods enqueue ops; the worker drains the queue and coalesces consecutive
    renders into the latest one before painting.
    """

    ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

    def __init__(self, real_stdout: object) -> None:
        self.real_stdout = real_stdout
        self.queue: queue.Queue[UiOp] = queue.Queue()
        self.thread: threading.Thread | None = None
        # Producer-side dedupe: drop render calls whose display text is
        # identical to the last enqueued one within a short window. Worker-side
        # coalescing handles in-flight bursts; this stops pointless enqueues
        # during the 50ms ticker + dense on_chunk calls.
        self.dedupe_text: str | None = None
        self.dedupe_time: float = 0.0
        self.dedupe_lock = threading.Lock()

    def start(self) -> None:
        if self.thread is not None:
            return
        self.thread = threading.Thread(target=self.worker, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        if self.thread is None:
            return
        self.queue.put(UiOp(kind="stop"))
        self.queue.join()
        self.thread.join(timeout=1.0)
        self.thread = None

    def render(self, display: str) -> None:
        now = time.monotonic()
        with self.dedupe_lock:
            if display == self.dedupe_text and now - self.dedupe_time < 0.016:
                return
            self.dedupe_text = display
            self.dedupe_time = now
        self.queue.put(UiOp(kind="render", text=display))

    def println(self, text: str = "") -> None:
        self.queue.put(UiOp(kind="println", text=text))

    def clear(self) -> None:
        with self.dedupe_lock:
            self.dedupe_text = None
            self.dedupe_time = 0.0
        self.queue.put(UiOp(kind="clear"))

    def commit_render(self, extra_blank_lines: int = 0) -> None:
        with self.dedupe_lock:
            self.dedupe_text = None
            self.dedupe_time = 0.0
        self.queue.put(UiOp(kind="commit_render", count=extra_blank_lines))

    def wait_idle(self) -> None:
        self.queue.join()

    def worker(self) -> None:
        current_render_lines = 0
        pending_op: UiOp | None = None
        while True:
            if pending_op is None:
                op = self.queue.get()
                task_done_count = 1
            else:
                op = pending_op
                pending_op = None
                task_done_count = 1
            try:
                if op.kind == "stop":
                    return
                if op.kind == "clear":
                    clear_seq = Ui.make_clear_seq(current_render_lines)
                    if clear_seq:
                        self.real_stdout.write(clear_seq) # type: ignore
                        self.real_stdout.flush() # type: ignore
                    current_render_lines = 0
                elif op.kind == "render":
                    latest_render = op
                    while True:
                        try:
                            queued_op = self.queue.get_nowait()
                        except queue.Empty:
                            break
                        if queued_op.kind == "render":
                            latest_render = queued_op
                            task_done_count += 1
                            continue
                        pending_op = queued_op
                        break

                    display = latest_render.text
                    if not display:
                        clear_seq = Ui.make_clear_seq(current_render_lines)
                        if clear_seq:
                            self.real_stdout.write(clear_seq) # type: ignore
                            self.real_stdout.flush() # type: ignore
                        current_render_lines = 0
                        continue
                    new_lines = Ui.count_display_lines(display)
                    clear_seq = Ui.make_clear_seq(current_render_lines)
                    # End with \r so the cursor parks at column 1 of the last
                    # rendered row. make_clear_seq matches this invariant.
                    self.real_stdout.write(clear_seq + display + "\r") # type: ignore
                    self.real_stdout.flush() # type: ignore
                    current_render_lines = new_lines
                elif op.kind == "println":
                    clear_seq = Ui.make_clear_seq(current_render_lines)
                    self.real_stdout.write(clear_seq + op.text + "\n") # type: ignore
                    self.real_stdout.flush() # type: ignore
                    current_render_lines = 0
                elif op.kind == "commit_render":
                    # Cursor is at col 1 of the last rendered row; advance
                    # past it so subsequent output doesn't overwrite it.
                    if current_render_lines > 0:
                        self.real_stdout.write("\n") # type: ignore
                    if op.count > 0:
                        self.real_stdout.write("\n" * op.count) # type: ignore
                    self.real_stdout.flush() # type: ignore
                    current_render_lines = 0
            finally:
                for _ in range(task_done_count):
                    self.queue.task_done()

    @staticmethod
    def make_clear_seq(num_lines: int) -> str:
        # Renders leave the cursor at column 1 of the *last* rendered row (we
        # end each render with \r, not \n). To clear a region of N rows we
        # therefore only need to move up N-1 rows and clear downward. Ending
        # on the last row rather than the line below avoids a phantom-row
        # off-by-one when the final line happens to fill the terminal width
        # exactly.
        if num_lines <= 0:
            return ""
        if num_lines == 1:
            return "\r\033[J"
        return f"\033[{num_lines - 1}A\r\033[J"

    @staticmethod
    def term_cols() -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    @staticmethod
    def display_width(s: str) -> int:
        # Approximate display columns. Treats East Asian Wide / Fullwidth as 2,
        # combining marks and most control chars as 0, everything else as 1.
        # Doesn't fully model regional-indicator / skin-tone emoji sequences,
        # but those are rare in conversational LLM output.
        width = 0
        for ch in s:
            if unicodedata.combining(ch):
                continue
            cat = unicodedata.category(ch)
            if cat.startswith("C"):
                continue
            if unicodedata.east_asian_width(ch) in ("W", "F"):
                width += 2
            else:
                width += 1
        return width

    @staticmethod
    def count_display_lines(text: str) -> int:
        plain = Ui.ANSI_RE.sub("", text)
        cols = Ui.term_cols()
        total = 0
        for line in plain.split("\n"):
            w = Ui.display_width(line)
            total += max(1, (w + cols - 1) // cols)
        return max(1, total)


class PromptBuilder:
    """
    Owns one user-prompt input concern: receive transcriptions, render the
    editable prompt line, dispatch keys, return the assembled prompt on Enter.
    Long-lived for the whole conversation; build() drives one input turn and
    is called in a loop by Conversation.
    """

    def __init__(
        self,
        ui: Ui,
        console: ConsoleSession,
        ctrl_c_requested: threading.Event,
    ) -> None:
        self.ui = ui
        self.console = console
        self.ctrl_c_requested = ctrl_c_requested
        self.chunk_queue: queue.Queue[str] = queue.Queue()
        self.prompt_chunks: list[str] = []
        self.selected_idx: int | None = None
        self.mic_paused = False

    def on_transcription(self, segments: list[Segment]) -> None:
        if self.mic_paused:
            return
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            self.chunk_queue.put(text)

    def resume(self) -> None:
        self.mic_paused = False

    def build(self) -> str:
        """
        Run one input turn. Returns the assembled prompt on Enter, with the
        finalized prompt line already committed to the UI and internal state
        reset for the response turn (mic_paused=True, chunks cleared, queue
        drained). Raises KeyboardInterrupt if Ctrl-C was requested.
        """
        self.render()
        while True:
            if self.ctrl_c_requested.is_set():
                raise KeyboardInterrupt

            updated = False
            while not self.chunk_queue.empty():
                chunk = self.chunk_queue.get_nowait()
                self.prompt_chunks.append(chunk)
                self.selected_idx = len(self.prompt_chunks) - 1
                updated = True
            if updated:
                self.render()

            key = self.console.read_key()
            if key is None:
                time.sleep(0.02)
                continue
            if key == KEY_CTRL_C:
                self.ctrl_c_requested.set()
                raise KeyboardInterrupt

            if key == KEY_LEFT:
                if self.prompt_chunks:
                    self.selected_idx = (
                        len(self.prompt_chunks) - 1
                        if self.selected_idx is None
                        else max(0, self.selected_idx - 1)
                    )
            elif key == KEY_RIGHT:
                if self.selected_idx is not None:
                    self.selected_idx = min(self.selected_idx + 1, len(self.prompt_chunks) - 1)
            elif key in (KEY_DEL, KEY_DEL2, KEY_FWDDEL):
                if self.selected_idx is not None and self.prompt_chunks:
                    self.prompt_chunks.pop(self.selected_idx)
                    if not self.prompt_chunks:
                        self.selected_idx = None
                    else:
                        self.selected_idx = min(self.selected_idx, len(self.prompt_chunks) - 1)
                elif self.prompt_chunks:
                    self.prompt_chunks.pop()
            elif key == KEY_ENTER and self.prompt_chunks:
                assembled = " ".join(self.prompt_chunks)
                self.commit_finalized_prompt(assembled)
                return assembled

            self.render()

    def commit_finalized_prompt(self, assembled: str) -> None:
        self.ui.println(f"> {COL_DIM}{assembled}{Ansi.RESET}")
        self.ui.println()
        self.prompt_chunks.clear()
        self.selected_idx = None
        self.mic_paused = True
        while not self.chunk_queue.empty():
            self.chunk_queue.get_nowait()

    def render(self) -> None:
        parts: list[str] = []
        for i, chunk in enumerate(self.prompt_chunks):
            if i == self.selected_idx:
                parts.append(f"{COL_MEDIUM}{Ansi.ITALICS}{chunk}{Ansi.RESET}")
            else:
                parts.append(f"{COL_DIM}{chunk}{Ansi.RESET}")
        content = " ".join(parts) if parts else f"{COL_OK}{Ansi.ITALICS}Speak into the mic...{Ansi.RESET}"
        self.ui.render(f"> {content}")


class MuteCurrentThreadOutput:
    """
    Context manager that suppresses output from the current thread for the
    duration of the `with` block. Mutes any QueuedStream installed on
    sys.stdout/sys.stderr, attaches a logging filter that drops records
    emitted on this thread, and redirects fd 2 to /dev/null to silence native
    code that writes directly to the stderr file descriptor.
    """

    def __init__(self, real_stderr: object, fd2_redirect_lock: threading.Lock) -> None:
        self.real_stderr = real_stderr
        self.fd2_redirect_lock = fd2_redirect_lock
        self.muted: list[QueuedStream] = []
        self.log_filter: logging.Filter | None = None
        self.filtered_handlers: list[logging.Handler] = []
        self.saved_stderr_fd: int | None = None
        self.devnull_fd: int | None = None

    def __enter__(self) -> None:
        for name in ("stdout", "stderr"):
            stream = getattr(sys, name)
            if isinstance(stream, QueuedStream):
                stream.mute()
                self.muted.append(stream)

        muted_thread_id = threading.get_ident()

        class CurrentThreadLogFilter(logging.Filter):
            def filter(self, record: logging.LogRecord) -> bool:
                return record.thread != muted_thread_id

        self.log_filter = CurrentThreadLogFilter()
        seen: set[int] = set()
        for logger_obj in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
            if not isinstance(logger_obj, logging.Logger):
                continue
            for handler in logger_obj.handlers:
                if id(handler) in seen:
                    continue
                seen.add(id(handler))
                handler.addFilter(self.log_filter)
                self.filtered_handlers.append(handler)

        # Some TTS backends emit directly to the process stderr file
        # descriptor from native code, bypassing Python's sys.stderr and
        # logging entirely. Temporarily redirect fd 2 to /dev/null around
        # inference to suppress that.
        self.fd2_redirect_lock.acquire()
        try:
            self.real_stderr.flush() # type: ignore
        except Exception:
            pass
        try:
            self.saved_stderr_fd = os.dup(2)
            self.devnull_fd = os.open(os.devnull, os.O_WRONLY)
            os.dup2(self.devnull_fd, 2)
        except Exception:
            if self.devnull_fd is not None:
                os.close(self.devnull_fd)
                self.devnull_fd = None
            if self.saved_stderr_fd is not None:
                os.close(self.saved_stderr_fd)
                self.saved_stderr_fd = None
            self.fd2_redirect_lock.release()
            raise

    def __exit__(self, *_: object) -> None:
        try:
            if self.saved_stderr_fd is not None:
                try:
                    self.real_stderr.flush() # type: ignore
                except Exception:
                    pass
                os.dup2(self.saved_stderr_fd, 2)
        finally:
            if self.devnull_fd is not None:
                os.close(self.devnull_fd)
                self.devnull_fd = None
            if self.saved_stderr_fd is not None:
                os.close(self.saved_stderr_fd)
                self.saved_stderr_fd = None
            if self.fd2_redirect_lock.locked():
                self.fd2_redirect_lock.release()

        if self.log_filter is not None:
            for handler in self.filtered_handlers:
                handler.removeFilter(self.log_filter)
            self.filtered_handlers.clear()
            self.log_filter = None
        for stream in self.muted:
            stream.unmute()
        self.muted.clear()


class ResponseSession:
    """
    Handles one LLM→TTS response turn. Construct fresh per Enter-press.
    Owns all per-turn threading state. Exposes sound_stream so the outer
    finally can shut it down if the session is interrupted mid-turn.
    """

    RESPONSE_PLACEHOLDER = "..."

    def __init__(
        self,
        ui: Ui,
        llm: LlmUtil,
        project: Project,
        chunking_config: ChunkingConfig,
        stt_variant: SttVariant,
        stt_config: SttConfig,
        real_stderr: object,
        fd2_redirect_lock: threading.Lock,
        ctrl_c_requested: threading.Event,
        phrase_stt_enabled: bool = True,
    ) -> None:
        self.ui = ui
        self.llm = llm
        self.project = project
        self.chunking_config = chunking_config
        self.stt_variant = stt_variant
        self.stt_config = stt_config
        self.real_stderr = real_stderr
        self.fd2_redirect_lock = fd2_redirect_lock
        self.ctrl_c_requested = ctrl_c_requested
        self.phrase_stt_enabled = phrase_stt_enabled
        self.sound_stream: SoundDeviceStream | None = None

    def run(self, assembled: str) -> None:
        self.tts_q: queue.Queue[str | None] = queue.Queue()
        self.tts_buffer = ""
        self.render_buffer = ResponseSession.RESPONSE_PLACEHOLDER
        self.spoken_segments: list[tuple[str, int, int]] = []
        self.pending_sentences: list[str] = []
        self.state_lock = threading.Lock()
        self.render_lock = threading.Lock()
        self.last_render_key: tuple | None = None
        self.playback_done = False
        self.worker: threading.Thread | None = None
        self.render_stop = threading.Event()
        self.interrupt_requested = threading.Event()
        self.response_aborted = threading.Event()
        self.llm_content_received = False

        self.worker = threading.Thread(target=self.tts_worker, daemon=True)
        self.worker.start()
        ticker = threading.Thread(target=self.render_ticker, daemon=True)
        ticker.start()
        self.render_response()

        llm_failed = False
        response_cancelled = False
        llm_completed = False
        try:
            self.llm.send(assembled, on_chunk=self.on_chunk, interrupt_event=self.interrupt_requested)
            llm_completed = True

            final_text = ""
            with self.state_lock:
                if self.tts_buffer.strip():
                    final_text = self.tts_buffer
                    self.pending_sentences.append(self.tts_buffer)
                self.tts_buffer = ""
                self.render_buffer = ""
            if final_text:
                self.tts_q.put(final_text)
            self.tts_q.put(None)
            self.worker.join()
            if self.sound_stream is not None:
                while not self.sound_stream.is_playback_complete and not self.interrupt_requested.is_set():
                    time.sleep(0.02)
        except KeyboardInterrupt:
            self.ctrl_c_requested.clear()
            response_cancelled = True
            self.interrupt_requested.set()
            if not llm_completed:
                self.rollback_interrupted_llm_user_turn(assembled)
        except Exception as e:
            llm_failed = True
            with self.state_lock:
                self.pending_sentences.clear()
                self.tts_buffer = ""
                self.render_buffer = ""
            self.ui.clear()
            self.ui.println(f"[LLM error: {make_error_string(e)}]")

        was_interrupted = self.interrupt_requested.is_set() or response_cancelled

        if llm_failed or was_interrupted:
            self.abort_response(ticker)
        else:
            self.render_stop.set()
            ticker.join(timeout=1.0)

        self.playback_done = True
        self.last_render_key = None
        if not self.response_aborted.is_set():
            self.render_response()
        self.ui.commit_render(1)
        self.ui.wait_idle()

    def tts_worker(self) -> None:
        while True:
            text = self.tts_q.get()
            if text is None:
                break
            if self.interrupt_requested.is_set() or self.response_aborted.is_set():
                continue
            if not text.strip():
                with self.state_lock:
                    if self.pending_sentences and self.pending_sentences[0] == text:
                        self.pending_sentences.pop(0)
                self.render_response()
                continue
            try:
                tts_instance = Tts.get_instance()
                massaged = tts_instance.massage_for_inference(text)
                with MuteCurrentThreadOutput(self.real_stderr, self.fd2_redirect_lock):
                    result = tts_instance.generate_using_project(self.project, [massaged])
                if self.interrupt_requested.is_set() or self.response_aborted.is_set():
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                    continue
                if isinstance(result, str):
                    if not self.response_aborted.is_set():
                        self.ui.println(f"[TTS error: {result}]")
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                    self.render_response()
                    continue
                sound = result[0]
                if self.interrupt_requested.is_set() or self.response_aborted.is_set():
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                    continue
                if self.sound_stream is None:
                    self.sound_stream = SoundDeviceStream(sound.sr)
                    self.sound_stream.start()
                if not self.interrupt_requested.is_set() and not self.response_aborted.is_set():
                    start, end = self.sound_stream.add_data(sound.data)
                    is_first_tts_chunk = not self.spoken_segments
                    spoken_segments = self.make_spoken_segments(text, sound, start, end, is_first_tts_chunk)
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                        self.spoken_segments.extend(spoken_segments)
                    self.render_response()
                else:
                    with self.state_lock:
                        if self.pending_sentences and self.pending_sentences[0] == text:
                            self.pending_sentences.pop(0)
                    self.render_response()
            except Exception as e:
                if not self.response_aborted.is_set():
                    self.ui.println(f"[TTS exception: {e}]")
                with self.state_lock:
                    if self.pending_sentences and self.pending_sentences[0] == text:
                        self.pending_sentences.pop(0)
                self.render_response()

    def render_response(self) -> None:
        if self.response_aborted.is_set():
            return
        with self.render_lock:
            with self.state_lock:
                segs = list(self.spoken_segments)
                pending = list(self.pending_sentences)
                buf = self.render_buffer
            pos = self.sound_stream.play_position_samples if self.sound_stream else 0
            active_idx = None if self.playback_done else next(
                (i for i, (_, start, end) in enumerate(segs) if start <= pos < end),
                None,
            )
            render_key = (
                active_idx,
                tuple(text for text, _, _ in segs),
                tuple(pending),
                buf,
            )
            if render_key == self.last_render_key:
                return
            self.last_render_key = render_key
            parts = []
            for i, (text, _, end) in enumerate(segs):
                display_text = ResponseSession.make_display_text(text)
                if end <= pos:
                    parts.append(f"{COL_DIM}{display_text}{Ansi.RESET}")
                elif i == active_idx:
                    parts.append(f"{COL_ACCENT}{display_text}{Ansi.RESET}")
                else:
                    parts.append(f"{COL_DIM}{display_text}{Ansi.RESET}")
            for i, text in enumerate(pending):
                display_text = ResponseSession.make_display_text(text)
                parts.append(f"{COL_MEDIUM}{Ansi.ITALICS}{display_text}{Ansi.RESET}" if i == 0 else f"{COL_DIM}{display_text}{Ansi.RESET}")
            if buf:
                parts.append(f"{COL_DIM}{ResponseSession.make_display_text(buf)}{Ansi.RESET}")
            content = "".join(parts)
            display = f"{COL_ACCENT}>{Ansi.RESET} {content}" if content else ""
            if display:
                self.ui.render(display)
            else:
                self.ui.clear()

    def on_chunk(self, delta: str) -> None:
        if self.interrupt_requested.is_set() or self.response_aborted.is_set():
            return

        to_send: list[str] = []
        with self.state_lock:
            if not self.llm_content_received and self.render_buffer == ResponseSession.RESPONSE_PLACEHOLDER:
                self.render_buffer = ""
            self.llm_content_received = True
            complete_chunks, self.tts_buffer, self.render_buffer = ResponseSession.consume_tts_delta(
                tts_buffer=self.tts_buffer,
                delta=delta,
                config=self.chunking_config,
                has_pending_sentences=bool(self.pending_sentences),
                has_spoken_segments=bool(self.spoken_segments),
            )
            for s in complete_chunks:
                self.pending_sentences.append(s)
                to_send.append(s)
            if to_send:
                self.render_buffer = ""
        self.render_response()
        for s in to_send:
            self.tts_q.put(s)

    def render_ticker(self) -> None:
        while not self.render_stop.wait(0.05):
            self.render_response()

    def abort_response(self, ticker: threading.Thread) -> None:
        self.interrupt_requested.set()
        self.response_aborted.set()
        while not self.tts_q.empty():
            try:
                self.tts_q.get_nowait()
            except queue.Empty:
                break
        self.tts_q.put(None)
        if self.sound_stream is not None:
            self.sound_stream.shut_down()
            self.sound_stream = None
        self.render_stop.set()
        ticker.join(timeout=1.0)
        if self.worker is not None:
            self.worker.join(timeout=0.1)

    def rollback_interrupted_llm_user_turn(self, message: str) -> None:
        with self.llm.history_lock:
            if self.llm.history and self.llm.history[-1] == {"role": "user", "content": message}:
                self.llm.history.pop()

    def make_spoken_segments(
        self,
        text: str,
        sound: Sound,
        stream_start: int,
        stream_end: int,
        is_first_tts_chunk: bool = False,
    ) -> list[tuple[str, int, int]]:
        if self.phrase_stt_enabled and not is_first_tts_chunk:
            phrase_segments = self.make_phrase_spoken_segments(text, sound, stream_start, stream_end)
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

            words = WhisperUtil.transcribe_to_words(
                sound,
                self.project.language_code,
                self.stt_variant,
                self.stt_config,
            )
            if isinstance(words, str) or not words:
                return []

            timed_phrases = ForceAlignUtil.make_timed_phrases(phrases, words, sound.duration)
            return self.timed_phrases_to_spoken_segments(timed_phrases, sound.sr, stream_start, stream_end)
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
    def segment_text_to_chunks(text: str, config: ChunkingConfig) -> list[str]:
        groups = PhraseGrouper.text_to_groups(
            text=text,
            max_words=config.max_words,
            strategy=config.strategy,
            pysbd_lang=config.language_code,
        )
        return [group.text for group in groups if group.text.strip()]

    @staticmethod
    def extract_complete_chunks(text: str, config: ChunkingConfig) -> tuple[list[str], str]:
        sentences = PhraseSegmenter.string_to_sentence_strings(text, config.language_code)
        if len(sentences) < 2:
            return [], text
        complete_text = "".join(sentences[:-1])
        remainder = sentences[-1]
        return ResponseSession.segment_text_to_chunks(complete_text, config), remainder

    @staticmethod
    def make_stable_chunk_preview(text: str, config: ChunkingConfig) -> str:
        chunks = ResponseSession.segment_text_to_chunks(text, config)
        if len(chunks) >= 2:
            return "".join(chunks[:-1])
        return ""

    @staticmethod
    def consume_tts_delta(
        tts_buffer: str,
        delta: str,
        config: ChunkingConfig,
        has_pending_sentences: bool,
        has_spoken_segments: bool,
    ) -> tuple[list[str], str, str]:
        """
        Fold a newly streamed LLM delta into the TTS buffer.

        Returns:
        - chunks ready to send to TTS
        - updated tts_buffer remainder
        - render_buffer preview text

        Important invariant: do not eagerly emit the very first chunk when the
        current buffer is still a single chunk. Otherwise a stream pause after a
        word fragment like "to" can cause the next delta starting with " ask"
        to be spoken/displayed as "toask".
        """
        next_buffer = tts_buffer + delta
        full_buffer = next_buffer
        to_send, next_buffer = ResponseSession.extract_complete_chunks(next_buffer, config)

        is_first_chunk = not has_pending_sentences and not has_spoken_segments

        if to_send:
            if is_first_chunk and sum(len(c.split()) for c in to_send) < 5:
                # First chunk must be >= 5 words. Hold the boundary, restore full buffer.
                to_send = []
                next_buffer = full_buffer
            else:
                return to_send, next_buffer, ""

        if is_first_chunk:
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
                        remainder = "".join(p.text for p in phrases[i + 1:])
                        if remainder.strip():
                            return [cumulative_text], remainder, ""
                        break

        render_buffer = ResponseSession.make_stable_chunk_preview(next_buffer, config)
        return [], next_buffer, render_buffer
