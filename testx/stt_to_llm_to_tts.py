#!/usr/bin/env python3
"""
STT to LLM to TTS prototype

Linux
"""
import os
import queue
import re
import select
import signal
import sys
import termios
import threading
import time
import tty
import logging
from pathlib import Path



sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper.transcribe import Segment

from tts_audiobook_tool.util import *
from tts_audiobook_tool.constants import *

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.llm_util import LlmUtil
from tts_audiobook_tool.phrase_segmenter import PhraseSegmenter
from tts_audiobook_tool.models_util import ModelsUtil
from tts_audiobook_tool.state import State
from tts_audiobook_tool.sound_device_stream import SoundDeviceStream
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.whisper_realtime_util import WhisperRealTimeUtil

API_ENDPOINT_URL = "https://api.deepseek.com/v1/chat/completions"
TOKEN  = ""
MODEL  = "deepseek-v4-flash"
PARAMS = {"thinking": {"type": "disabled"}}
SYSTEM_PROMPT = "You are have a voice conversation with a user. The user is speaking using voice, transcribed by STT. Your responses are vocalized using TTS. Be conversational and use short responses to facilitate interactive conversational flow. Do not use unnecessary text formatting or emojies."
DEBUG_VAD = False

ORANGE  = "\033[38;5;208m"
WHITE   = "\033[97m"
GREEN   = "\033[92m"
RESET   = "\033[0m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

KEY_LEFT   = "\x1b[D"
KEY_RIGHT  = "\x1b[C"
KEY_DEL    = "\x7f"
KEY_DEL2   = "\x08"
KEY_FWDDEL = "\x1b[3~"
KEY_ENTER  = "\n"


class _PerThreadStream:
    """Proxy for sys.stdout/stderr with per-thread muting via a threading.local flag."""

    def __init__(self, real: object) -> None:
        self._real = real
        self._local = threading.local()

    def mute(self) -> None:
        self._local.muted = True

    def unmute(self) -> None:
        self._local.muted = False

    def write(self, data: str) -> int:
        if not getattr(self._local, 'muted', False):
            return self._real.write(data)  # type: ignore[union-attr]
        return len(data)

    def flush(self) -> None:
        if not getattr(self._local, 'muted', False):
            self._real.flush()  # type: ignore[union-attr]

    def __getattr__(self, name: str) -> object:
        return getattr(self._real, name)


def read_key(fd: int) -> str | None:
    """Non-blocking read of one key (terminal must be in cbreak mode)."""
    if not select.select([sys.stdin], [], [], 0)[0]:
        return None
    ch = os.read(fd, 1)
    if ch == b"\x1b":
        if select.select([sys.stdin], [], [], 0.05)[0]:
            ch += os.read(fd, 8)
    return ch.decode("utf-8", errors="replace")


def main() -> None:
    _real_stdout, _real_stderr = sys.stdout, sys.stderr

    print()
    print_heading("Real-time STT → LLM → TTS Demo")  
    print()

    # Init:

    Tts.init_model_type()

    state = State()
    prefs = state.prefs    
    project = state.project

    # Init static state values using project and prefs
    Tts.set_model_params_using_project(project)
    Tts.set_force_cpu(prefs.tts_force_cpu)
    Stt.set_variant(prefs.stt_variant)
    Stt.set_config(prefs.stt_config)
    
    # Instantiate STT and TTS models
    ModelsUtil.warm_up_models(state)

    # TTS prereq check
    prereq_errors = Tts.get_class().get_prereq_errors(project, Tts.get_instance(), short_format=False) 
    if prereq_errors:
        print("Prerequisite errors:")
        for err in prereq_errors:
            print(f"  - {err}")
        print("Please fix these issues and try again.")
        return  

    llm = LlmUtil(
        api_endpoint_url=API_ENDPOINT_URL,
        token=TOKEN,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT,
        extra_params=PARAMS,
        verbose=False,
    )

    chunk_queue: queue.Queue[str] = queue.Queue()
    prompt_chunks: list[str] = []
    selected_idx: int | None = None
    render_prev_lines = 0  # how many terminal lines the last render occupied
    mic_paused = False
    sound_stream: SoundDeviceStream | None = None

    def on_transcription(segments: list[Segment]) -> None:
        if mic_paused:
            return
        text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            chunk_queue.put(text)

    def term_cols() -> int:
        try:
            return os.get_terminal_size().columns
        except OSError:
            return 80

    def count_display_lines(text: str) -> int:
        plain = ANSI_RE.sub("", text)
        cols = term_cols()
        total = 0
        for line in plain.split("\n"):
            total += max(1, (len(line) + cols - 1) // cols)
        return max(1, total)

    def clear_render() -> None:
        nonlocal render_prev_lines
        if render_prev_lines > 1:
            print(f"\033[{render_prev_lines - 1}A", end="")
        if render_prev_lines > 0:
            print("\r\033[J", end="", flush=True)
        render_prev_lines = 0

    def render() -> None:
        nonlocal render_prev_lines
        parts = []
        for i, chunk in enumerate(prompt_chunks):
            parts.append(f"{WHITE}{chunk}{RESET}" if i == selected_idx else f"{COL_DIM}{chunk}{RESET}")
        content = " ".join(parts) if parts else f"{COL_DIM}{Ansi.ITALICS}Speak into mic...{RESET}"
        display = f"> {content}"
        new_lines = count_display_lines(display)
        clear_render()
        print(display, end="", flush=True)
        render_prev_lines = new_lines

    util = WhisperRealTimeUtil(
        prefs=prefs,
        on_transcription=on_transcription,
        debug_vad=DEBUG_VAD,
    )
    util.start()
    print("Listening...  ←→ select chunk  Del remove  Enter send  Ctrl-C quit\n")
    render()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    try:
        tty.setcbreak(fd)

        while True:
            updated = False
            while not chunk_queue.empty():
                chunk = chunk_queue.get_nowait()
                prompt_chunks.append(chunk)
                selected_idx = len(prompt_chunks) - 1
                updated = True
            if updated:
                render()

            key = read_key(fd)
            if key is None:
                time.sleep(0.02)
                continue

            if key == KEY_LEFT:
                if prompt_chunks:
                    selected_idx = (
                        len(prompt_chunks) - 1
                        if selected_idx is None
                        else max(0, selected_idx - 1)
                    )
            elif key == KEY_RIGHT:
                if selected_idx is not None:
                    if selected_idx >= len(prompt_chunks) - 1:
                        selected_idx = None
                    else:
                        selected_idx += 1
            elif key in (KEY_DEL, KEY_DEL2, KEY_FWDDEL):
                if selected_idx is not None and prompt_chunks:
                    prompt_chunks.pop(selected_idx)
                    if not prompt_chunks:
                        selected_idx = None
                    else:
                        selected_idx = min(selected_idx, len(prompt_chunks) - 1)
                elif prompt_chunks:
                    prompt_chunks.pop()
            elif key == KEY_ENTER and prompt_chunks:
                assembled = " ".join(prompt_chunks)
                clear_render()
                print(f"> {COL_DIM}{assembled}{RESET}")
                print()
                prompt_chunks.clear()
                selected_idx = None
                mic_paused = True
                while not chunk_queue.empty():
                    chunk_queue.get_nowait()

                use_tts = project is not None
                tts_q: queue.Queue[str | None] = queue.Queue()
                tts_buffer = ""
                render_buffer = ""
                # (sentence_text, start_sample, end_sample) for sentences whose
                # audio has been generated and queued to the sound stream.
                spoken_segments: list[tuple[str, int, int]] = []
                # Sentences enqueued to TTS but not yet rendered to audio.
                pending_sentences: list[str] = []
                state_lock = threading.Lock()
                render_lock = threading.Lock()
                response_render_prev_lines = 0
                last_render_key: tuple | None = None
                playback_done = False
                worker: threading.Thread | None = None
                render_stop = threading.Event()
                interrupt_requested = threading.Event()
                # Ctrl-C is locked out once the first LLM token arrives to avoid the
                # ambiguity of a partial assistant reply in history.
                # TODO: consider enabling interrupt here with partial-reply rollback
                #       once the UX design for that case is clearer.
                ctrlc_locked = False
                llm_content_received = False
                old_sigint = signal.signal(
                    signal.SIGINT,
                    lambda *_: (None if ctrlc_locked else interrupt_requested.set()),
                )

                class _MuteCurrentThreadOutput:
                    """
                    Temporarily silence only this worker thread.

                    The prototype used to install _PerThreadStream for the entire
                    program lifetime.  That should have passed other threads'
                    output through, but it made debugging confusing because all
                    console output still flowed through a muting proxy.  Install
                    the proxy only while a TTS inference call is active, mute only
                    the TTS worker thread, then restore the real streams.
                    """

                    def __init__(self) -> None:
                        self._installed: list[tuple[str, _PerThreadStream, object]] = []
                        self._muted_existing: list[_PerThreadStream] = []
                        self._log_filter: logging.Filter | None = None
                        self._filtered_handlers: list[logging.Handler] = []

                    def __enter__(self) -> None:
                        for name in ("stdout", "stderr"):
                            stream = getattr(sys, name)
                            if isinstance(stream, _PerThreadStream):
                                stream.mute()
                                self._muted_existing.append(stream)
                            else:
                                proxy = _PerThreadStream(stream)
                                proxy.mute()
                                setattr(sys, name, proxy)
                                self._installed.append((name, proxy, stream))

                        # Python logging handlers keep a reference to whatever
                        # sys.stderr was when the handler was created.  That
                        # means library messages like
                        # "WARNING:chatterbox.tts_turbo:..." can bypass the
                        # sys.stderr proxy above.  Drop log records emitted by
                        # this inference worker thread while the mute is active.
                        muted_thread_id = threading.get_ident()

                        class _CurrentThreadLogFilter(logging.Filter):
                            def filter(self, record: logging.LogRecord) -> bool:
                                return record.thread != muted_thread_id

                        self._log_filter = _CurrentThreadLogFilter()
                        seen: set[int] = set()
                        for logger_obj in [logging.getLogger(), *logging.Logger.manager.loggerDict.values()]:
                            if not isinstance(logger_obj, logging.Logger):
                                continue
                            for handler in logger_obj.handlers:
                                if id(handler) in seen:
                                    continue
                                seen.add(id(handler))
                                handler.addFilter(self._log_filter)
                                self._filtered_handlers.append(handler)

                    def __exit__(self, *_: object) -> None:
                        if self._log_filter is not None:
                            for handler in self._filtered_handlers:
                                handler.removeFilter(self._log_filter)
                            self._filtered_handlers.clear()
                            self._log_filter = None
                        for stream in self._muted_existing:
                            stream.unmute()
                        for name, proxy, real in reversed(self._installed):
                            proxy.unmute()
                            if getattr(sys, name) is proxy:
                                setattr(sys, name, real)

                def tts_worker() -> None:
                    nonlocal sound_stream
                    assert project is not None  # worker is only started when use_tts
                    while True:
                        text = tts_q.get()
                        if text is None:
                            break
                        # If interrupted, drain the queue without synthesizing.
                        if interrupt_requested.is_set():
                            continue
                        if not text.strip():
                            with state_lock:
                                if pending_sentences and pending_sentences[0] == text:
                                    pending_sentences.pop(0)
                            continue
                        try:
                            tts_instance = Tts.get_instance()
                            massaged = tts_instance.massage_for_inference(text)
                            with _MuteCurrentThreadOutput():
                                result = tts_instance.generate_using_project(project, [massaged])
                            if isinstance(result, str):
                                print(f"\n[TTS error: {result}]", flush=True)
                                with state_lock:
                                    if pending_sentences and pending_sentences[0] == text:
                                        pending_sentences.pop(0)
                                continue
                            sound = result[0]
                            if sound_stream is None:
                                sound_stream = SoundDeviceStream(sound.sr)
                                sound_stream.start()
                            # Don't add audio if interrupted while synthesizing.
                            if not interrupt_requested.is_set():
                                start, end = sound_stream.add_data(sound.data)
                                with state_lock:
                                    if pending_sentences and pending_sentences[0] == text:
                                        pending_sentences.pop(0)
                                    spoken_segments.append((text, start, end))
                            else:
                                with state_lock:
                                    if pending_sentences and pending_sentences[0] == text:
                                        pending_sentences.pop(0)
                        except Exception as e:
                            print(f"\n[TTS exception: {e}]", flush=True)
                            with state_lock:
                                if pending_sentences and pending_sentences[0] == text:
                                    pending_sentences.pop(0)

                def render_response() -> None:
                    nonlocal response_render_prev_lines, last_render_key
                    with render_lock:
                        with state_lock:
                            segs = list(spoken_segments)
                            pending = list(pending_sentences)
                            buf = render_buffer
                        pos = sound_stream.play_position_samples if sound_stream else 0
                        # Determine which segment is currently audible.
                        active_idx = None if playback_done else next(
                            (i for i, (_, start, end) in enumerate(segs) if start <= pos < end),
                            None,
                        )
                        render_key = (active_idx, len(segs), len(pending), buf)
                        if render_key == last_render_key:
                            return
                        last_render_key = render_key
                        parts = []
                        for i, (text, _, end) in enumerate(segs):
                            if end <= pos:
                                parts.append(f"{COL_DIM}{text}{RESET}")
                            elif i == active_idx:
                                parts.append(f"{ORANGE}{text}{RESET}")
                            else:
                                parts.append(f"{COL_DIM}{text}{RESET}")
                        for text in pending:
                            parts.append(f"{COL_DIM}{text}{RESET}")
                        if buf:
                            parts.append(f"{COL_DIM}{buf}{RESET}")
                        display = "".join(parts)
                        new_lines = count_display_lines(display)
                        if response_render_prev_lines > 1:
                            print(f"\033[{response_render_prev_lines - 1}A", end="")
                        if response_render_prev_lines > 0:
                            print("\r\033[J", end="", flush=True)
                        print(display, end="", flush=True)
                        response_render_prev_lines = new_lines

                def on_chunk(delta: str) -> None:
                    nonlocal tts_buffer, render_buffer, ctrlc_locked, llm_content_received
                    if interrupt_requested.is_set():
                        return
                    # Lock out Ctrl-C once the first token arrives to avoid a
                    # partial-reply history entry (see ctrlc_locked comment above).
                    ctrlc_locked = True
                    llm_content_received = True
                    if project is None:
                        print(delta, end="", flush=True)
                        return

                    def make_stable_phrase_preview(text: str) -> str:
                        """
                        Return only phrase-complete text for display.

                        The LLM stream arrives token-by-token, but repainting the
                        response on every token flickers while the first TTS job
                        is still being generated.  PhraseSegmenter keeps the last
                        phrase as incomplete, so the rendered tail advances only
                        at phrase boundaries while TTS state changes can still
                        repaint independently.
                        """
                        phrases = PhraseSegmenter.sentence_string_to_phrase_strings(text)
                        if len(phrases) >= 2:
                            return "".join(phrases[:-1])
                        return ""

                    to_send: list[str] = []
                    with state_lock:
                        tts_buffer += delta
                        sentences = PhraseSegmenter.string_to_sentence_strings(
                            tts_buffer, project.language_code
                        )
                        if len(sentences) >= 2:
                            for s in sentences[:-1]:
                                if s.strip():
                                    pending_sentences.append(s)
                                    to_send.append(s)
                            tts_buffer = sentences[-1]
                        if to_send:
                            render_buffer = ""
                        elif not pending_sentences and not spoken_segments:
                            # Nothing sent yet: cut at first phrase boundary rather than
                            # waiting for a full sentence, so TTS starts sooner.
                            phrases = PhraseSegmenter.sentence_string_to_phrase_strings(tts_buffer)
                            if len(phrases) >= 2:
                                first = phrases[0]
                                if first.strip():
                                    pending_sentences.append(first)
                                    to_send.append(first)
                                tts_buffer = "".join(phrases[1:])
                                render_buffer = ""
                            else:
                                render_buffer = make_stable_phrase_preview(tts_buffer)
                        else:
                            render_buffer = make_stable_phrase_preview(tts_buffer)
                    for s in to_send:
                        tts_q.put(s)

                def abort_response() -> None:
                    """Stop TTS worker and audio stream immediately on interrupt."""
                    nonlocal sound_stream
                    # Drain the TTS queue so the worker exits without synthesizing more.
                    while not tts_q.empty():
                        try:
                            tts_q.get_nowait()
                        except queue.Empty:
                            break
                    tts_q.put(None)
                    if worker is not None:
                        # TODO: revisit — if a synthesis is in progress when Ctrl-C fires,
                        #       this join blocks until that synthesis finishes (~1–5 s).
                        #       Audio is already stopped; the result is simply discarded.
                        worker.join(timeout=10.0)
                    ss = sound_stream
                    if ss is not None:
                        ss.shut_down()
                        sound_stream = None
                    render_stop.set()
                    ticker.join()

                def render_ticker() -> None:
                    while not render_stop.wait(0.05):
                        render_response()

                if use_tts:
                    worker = threading.Thread(target=tts_worker, daemon=True)
                    worker.start()

                ticker = threading.Thread(target=render_ticker, daemon=True)
                ticker.start()

                try:
                    llm.send(assembled, on_chunk=on_chunk, interrupt_event=interrupt_requested)
                except Exception as e:
                    print(f"\n[LLM error: {e}]")

                # Re-enable Ctrl-C now that the LLM call is done (ctrlc_locked was
                # set on first token to prevent partial-history ambiguity).
                ctrlc_locked = False

                was_interrupted = interrupt_requested.is_set()

                if was_interrupted and not use_tts:
                    # No TTS path: just stop the ticker.
                    render_stop.set()
                    ticker.join()
                elif was_interrupted:
                    abort_response()
                else:
                    if use_tts:
                        # Flush any remaining LLM-buffered text as one final sentence.
                        final_text = ""
                        with state_lock:
                            if tts_buffer.strip():
                                final_text = tts_buffer
                                pending_sentences.append(tts_buffer)
                            tts_buffer = ""
                        if final_text:
                            tts_q.put(final_text)
                        tts_q.put(None)
                        if worker is not None:
                            worker.join()
                        ss = sound_stream
                        if ss is not None:
                            while not ss.is_playback_complete and not interrupt_requested.is_set():
                                time.sleep(0.02)
                        if interrupt_requested.is_set():
                            abort_response()
                            was_interrupted = True
                        else:
                            render_stop.set()
                            ticker.join()
                    else:
                        render_stop.set()
                        ticker.join()

                signal.signal(signal.SIGINT, old_sigint)

                playback_done = True
                last_render_key = None  # force redraw even if active_idx was already None
                render_response()
                print()  # newline after streamed LLM output
                if was_interrupted and not llm_content_received:
                    # Turn was fully rolled back (no content received before interrupt).
                    print()
                    print("(turn erased — no reply received)")
                    print()
                else:
                    print()  # spacing before next prompt

                util.flush()
                mic_paused = False

            render()

    except KeyboardInterrupt:
        clear_render()
        print("Stopped.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        util.stop()
        if sound_stream is not None:
            sound_stream.shut_down()
        sys.stdout, sys.stderr = _real_stdout, _real_stderr


if __name__ == "__main__":
    main()
