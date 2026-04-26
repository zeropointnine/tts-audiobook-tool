#!/usr/bin/env python3

"""
STT to LLM - Minimal functionality test 

Linux
"""
import os
import queue
import re
import select
import sys
import termios
import time
import tty
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from faster_whisper.transcribe import Segment

from tts_audiobook_tool.ansi import Ansi
from tts_audiobook_tool.constants import COL_DIM
from tts_audiobook_tool.llm_util import LlmUtil
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.stt import Stt
from tts_audiobook_tool.whisper_realtime_util import WhisperRealTimeUtil

API_ENDPOINT_URL = "https://api.deepseek.com/v1/chat/completions"
TOKEN  = "xxx"
MODEL  = "deepseek-v4-flash"
PARAMS = {"thinking": {"type": "disabled"}}

ORANGE  = "\033[38;5;208m"
RESET   = "\033[0m"
ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

KEY_LEFT   = "\x1b[D"
KEY_RIGHT  = "\x1b[C"
KEY_DEL    = "\x7f"
KEY_DEL2   = "\x08"
KEY_FWDDEL = "\x1b[3~"
KEY_ENTER  = "\n"


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
    prefs = Prefs()
    Stt.set_variant(prefs.stt_variant)
    Stt.set_config(prefs.stt_config)
    print("Loading STT model ...")
    Stt.get_whisper()
    print("STT model ready.\n")

    llm = LlmUtil(
        api_endpoint_url=API_ENDPOINT_URL,
        token=TOKEN,
        model=MODEL,
        system_prompt="",
        extra_params=PARAMS,
        verbose=False,
    )

    chunk_queue: queue.Queue[str] = queue.Queue()
    prompt_chunks: list[str] = []
    selected_idx: int | None = None
    render_prev_lines = 0  # how many terminal lines the last render occupied
    mic_paused = False

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
            parts.append(f"{ORANGE}{chunk}{RESET}" if i == selected_idx else chunk)
        content = " ".join(parts) if parts else f"{COL_DIM}{Ansi.ITALICS}Speak into mic{RESET}"
        display = f"> {content}"
        visible_len = len(ANSI_RE.sub("", display))
        new_lines = max(1, (visible_len + term_cols() - 1) // term_cols())
        clear_render()
        print(display, end="", flush=True)
        render_prev_lines = new_lines

    util = WhisperRealTimeUtil(prefs=prefs, on_transcription=on_transcription)
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
                print(f"> {ORANGE}{assembled}{RESET}")
                print()
                prompt_chunks.clear()
                selected_idx = None
                mic_paused = True
                while not chunk_queue.empty():
                    chunk_queue.get_nowait()
                try:
                    llm.send(assembled, on_chunk=lambda c: print(c, end="", flush=True))
                except Exception as e:
                    print(f"\n[LLM error: {e}]")
                print() # ensure newline after LLM response
                print() # extra spacing before next prompt
                mic_paused = False

            render()

    except KeyboardInterrupt:
        clear_render()
        print("Stopped.")
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        util.stop()


if __name__ == "__main__":
    main()
