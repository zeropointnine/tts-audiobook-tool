from dataclasses import dataclass
import re
import json
import pathlib
import itertools
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

import numpy as np

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.sound_app_util import SoundAppUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.util import make_terminal_hyperlink, printt

_HERE = pathlib.Path(__file__).parent
_DEMOS_DIR = _HERE / "demos"
_DEMOS_DIR_RESOLVED = _DEMOS_DIR.resolve()

from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.l import L
from tts_audiobook_tool.server.audio_stream import AudioStream
from tts_audiobook_tool.server.audio_stream_http import AudioStreamHttp
from tts_audiobook_tool.tts import Tts


@dataclass
class PromptItem:
    phrase_group: PhraseGroup
    can_eager: bool
    use_tts_streaming: bool


class Server:

    def __init__(self):
        
        # Load current project
        prefs = Prefs.load()
        if not prefs.project_dir:
            printt(f"{COL_ERROR}Active project required")
            printt("Run tts-audiobook-tool and set up a new project. Then re-start the server.")
            exit(0)
        result = Project.load_using_dir_path(prefs.project_dir)
        if isinstance(result, str):
            printt(f"{COL_ERROR}Error: {result}")
            exit(0)
        self._project = result

        Tts.set_model_params_using_project(self._project) 

        s = f"{COL_ACCENT}Loaded tts-audiobook-tool's current active project's settings from:\n"
        s += f"{COL_ACCENT}{make_terminal_hyperlink(self._project.dir_path, is_file=True)}"
        printt(s)
        printt()

        # PriorityQueue allows for inserting items at front
        self._queue: queue.PriorityQueue[tuple[int, int, PromptItem]] = queue.PriorityQueue()

        self._queue_counter = itertools.count()
        self._prompt_currently_inferencing = ""
        self._generation_id = 0

        self._audio_stream = AudioStream()
        self._audio_http_stream = AudioStreamHttp()
        self._audio_stream.set_playback_listener(self._audio_http_stream.on_audio_played)

        self._is_initializing = True
        self._local_audio_enabled = True
        self._tts_streaming_enabled = Tts.get_info().can_stream
        self._tts_ready = threading.Event()

        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def run(self, host: str = "127.0.0.1", port: int = 5001):
        
        _server = self

        class _Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                if self.path == "/prompt":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    data = json.loads(body) if body else {}
                    self._respond(
                        _server.prompt(
                            data.get("prompt", ""), 
                            should_segment=data.get("should_segment", True),
                            can_eager=data.get("eager_first_segment", False)
                        )
                    )
                elif self.path == "/clear":
                    self._respond(_server.clear())
                elif self.path == "/local-audio":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    data = json.loads(body) if body else {}
                    self._respond(_server.local_audio(data.get("enabled", True)))
                elif self.path == "/tts-streaming":
                    length = int(self.headers.get("Content-Length", 0))
                    body = self.rfile.read(length)
                    data = json.loads(body) if body else {}
                    self._respond(_server.tts_streaming(data.get("enabled", True)))
                else:
                    self.send_error(404)

            def do_GET(self):
                parsed = urlparse(self.path)

                if parsed.path == "/status":
                    self._respond(_server.status())
                elif parsed.path.startswith("/demos/") and parsed.path.endswith(".html"):
                    demo_path = _DEMOS_DIR / pathlib.PurePosixPath(parsed.path.removeprefix("/demos/"))
                    try:
                        resolved_demo_path = demo_path.resolve()
                    except OSError:
                        self.send_error(404)
                        return
                    if not resolved_demo_path.is_file() or _DEMOS_DIR_RESOLVED not in resolved_demo_path.parents:
                        self.send_error(404)
                        return
                    body = resolved_demo_path.read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif parsed.path == "/stream":
                    q = _server._audio_http_stream.connect()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.end_headers()
                    try:
                        while True:
                            try:
                                chunk = q.get(timeout=5)
                            except queue.Empty:
                                # Heartbeat: 0-length app frame so we detect a dead connection.
                                # Client skips frames with msgLen == 0.
                                heartbeat = b'\x00\x00\x00\x00'
                                self.wfile.write(f"{len(heartbeat):x}\r\n".encode())
                                self.wfile.write(heartbeat)
                                self.wfile.write(b"\r\n")
                                self.wfile.flush()
                                continue
                            self.wfile.write(f"{len(chunk):x}\r\n".encode())
                            self.wfile.write(chunk)
                            self.wfile.write(b"\r\n")
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                    finally:
                        _server._audio_http_stream.disconnect(q)
                elif parsed.path == "/":
                    body = (_DEMOS_DIR / API_DEMO_HTML_FILE_NAME).read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.send_error(404)

            def _respond(self, data: dict):
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format, *args):
                msg = format % args
                if '/status' in msg:
                    return # bc too much chatter
                printt(f"[server] {self.address_string()} - {msg}")

        def _init_tts():
            Tts.get_instance()
            self._is_initializing = False
            self._tts_ready.set()
            printt(f"{COL_DIM_ITALICS}Ready")
            printt()

        with ThreadingHTTPServer((host, port), _Handler) as httpd:
            
            # Print some useful info
            base_url = f"http://{host}:{port}"
            demo_url_1 = f"{base_url}/demos/{API_DEMO_HTML_FILE_NAME}"
            demo_url_2 = f"{base_url}/demos/{STREAMING_CLIENT_DEMO_HTML_FILE_NAME}"
            demo_url_3 = f"{base_url}/demos/{COMBINATION_DEMO_HTML_FILE_NAME}"

            printt(f"{COL_ACCENT}Server listening on {make_terminal_hyperlink(base_url)}")
            printt()
            printt("Demo pages:")
            printt(f"- API demo: {make_terminal_hyperlink(demo_url_1)}")
            printt(f"- Streaming client demo: {make_terminal_hyperlink(demo_url_2)}")
            printt(f"- Combination demo: {make_terminal_hyperlink(demo_url_3)}")
            printt()

            # Initialize the TTS model
            threading.Thread(target=_init_tts, daemon=True).start()

            httpd.serve_forever()

    def status(self) -> dict:
        return {
            "status": "initializing" if self._is_initializing else "ready",
            "tts_model": Tts.get_type().value.ui["proper_name"],
            "inferencing": self._prompt_currently_inferencing,
            "playing": self._audio_stream.get_currently_playing(),
            "audio_buffer": self._audio_stream.get_seconds_left(),
            "num_queued": self._queue.qsize(),
            "stream_clients": self._audio_http_stream.client_count(),
            "local_audio": self._local_audio_enabled,
            "tts_streaming": self._tts_streaming_enabled,
            "tts_streaming_supported": Tts.get_info().can_stream,
        }

    def local_audio(self, enabled: bool) -> dict:
        self._local_audio_enabled = enabled
        self._audio_stream.set_is_mute(not enabled)
        return {"local_audio": self._local_audio_enabled}

    def tts_streaming(self, enabled: bool) -> dict:
        supported = Tts.get_info().can_stream
        effective_enabled = enabled and supported
        self._tts_streaming_enabled = effective_enabled

        response: dict[str, bool | str] = {
            "tts_streaming": self._tts_streaming_enabled,
            "tts_streaming_supported": supported,
        }
        if enabled and not supported:
            response["warning"] = "Current TTS engine does not support TTS streaming; using non-streaming inference."
        return response


    def prompt(self, prompt: str, should_segment: bool = True, can_eager: bool = False) -> dict:
        """
        Queues text prompt for TTS inference
        """
        prompt = prompt.strip()
        if not prompt:
            return {"error": "Empty prompt"}

        # Used for convenience response output feedback
        prompt_texts = []
        use_tts_streaming = self._tts_streaming_enabled and Tts.get_info().can_stream

        if should_segment:
            phrase_groups = PhraseGrouper.text_to_groups(prompt, self._project.max_words, self._project.segmentation_strategy, self._project.language_code)
            for phrase_group in phrase_groups:
                prompt_texts.append(phrase_group.text)
                self._queue.put((1, next(self._queue_counter), PromptItem(phrase_group, can_eager, use_tts_streaming)))
        else:
            phrase_group = PhraseGroup( [Phrase(text=prompt, reason=Reason.UNDEFINED)] )
            prompt_texts.append(prompt)
            self._queue.put((1, next(self._queue_counter), PromptItem(phrase_group, False, use_tts_streaming)))
       
        return {
            "input": prompt,
            "prompts": prompt_texts,
            "queue_length": self._queue.qsize(),
        }

    def clear(self) -> dict:
        self._generation_id += 1
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break
        self._audio_stream.clear()
        self._audio_http_stream.clear()
        return {}

    def generate_streaming_output(
        self,
        prompt_text: str,
        phrase_group: PhraseGroup,
        generation_id: int,
    ) -> bool:
        info = Tts.get_info()
        stream_started_at = time.monotonic()
        self.log_tts_inference_start(mode="streaming", text=prompt_text)
        first_audio_callback_registered = False
        streamed_audio_sample_count = 0

        def log_first_audio_latency() -> None:
            self.log_tts_first_audio_latency(
                mode="streaming",
                text=prompt_text,
                started_at=stream_started_at,
            )

        def on_stream_chunk(data) -> None:
            nonlocal first_audio_callback_registered, streamed_audio_sample_count
            if self._generation_id != generation_id:
                return
            start, end = self._audio_stream.append_data(data, info.sample_rate, prompt_text)
            streamed_audio_sample_count += end - start
            if not first_audio_callback_registered:
                first_audio_callback_registered = True
                self._audio_stream.set_first_audio_output_callback(start, log_first_audio_latency)

        def on_stream_end() -> None:
            nonlocal streamed_audio_sample_count
            if self._generation_id != generation_id:
                return
            if phrase_group.last_reason.pause_duration <= 0:
                return
            silence_sample_count = int(round(phrase_group.last_reason.pause_duration * info.sample_rate))
            if silence_sample_count <= 0:
                return
            silence = np.zeros(silence_sample_count, dtype=np.float32)
            start, end = self._audio_stream.append_data(silence, info.sample_rate, prompt_text)
            streamed_audio_sample_count += end - start

        try:
            result = Tts.generate_using_project(
                self._project,
                [prompt_text],
                force_random_seed=False,
                on_stream_chunk=on_stream_chunk,
                on_stream_end=on_stream_end,
            )
        finally:
            model = Tts.get_instance_if_exists()
            if model is not None:
                model.clear_stream_state()

        if self._generation_id != generation_id:
            return False
        if isinstance(result, str):
            printt(f"* TTS error: {result}")
            return False
        if streamed_audio_sample_count <= 0:
            printt("* No streamed audio output")
            return False
        return True

    def generate_non_streaming_output(
        self,
        prompt_text: str,
        phrase_group: PhraseGroup,
        generation_id: int,
    ) -> bool:
        started_at = time.monotonic()
        self.log_tts_inference_start(mode="non-streaming", text=prompt_text)
        result = Tts.generate_using_project(
            self._project,
            [prompt_text],
            force_random_seed=False,
        )
        if isinstance(result, str):
            printt(f"* TTS error: {result}")
            return False
        sound = result[0]

        if self._generation_id != generation_id:
            return False

        sound = SoundAppUtil.apply_generate_post_processing(sound)
        if sound.data.size == 0:
            printt("* Model output is empty or silence")
            return False

        sound = SoundAppUtil.apply_segment_post_processing(
            sound=sound,
            high_shelf=self._project.get_high_shelf(),
            limit_silence_gaps=self._project.limit_silence_gaps,
            use_upsampler=False,
        )
        assert isinstance(sound, Sound)

        if phrase_group.last_reason != Reason.UNDEFINED:
            sound = SoundUtil.append_pause_or_section_effect(
                sound, reason=phrase_group.last_reason, use_section_sound_effect=False
            )

        if self._generation_id != generation_id:
            return False

        start, _ = self._audio_stream.append(sound, prompt_text)
        self._audio_stream.set_first_audio_output_callback(
            start,
            lambda: self.log_tts_first_audio_latency(
                mode="non-streaming",
                text=prompt_text,
                started_at=started_at,
            ),
        )
        return True

    def log_tts_inference_start(self, mode: str, text: str) -> None:
        preview = re.sub(r"\s+", " ", text).strip()
        if len(preview) > 80:
            preview = preview[:80] + "..."
        L.i(
            f"TTS inference start ({mode}) | chars={len(text)} | text='{preview}'"
        )

    def log_tts_first_audio_latency(self, mode: str, text: str, started_at: float) -> None:
        elapsed_ms = (time.monotonic() - started_at) * 1000.0
        preview = re.sub(r"\s+", " ", text).strip()
        if len(preview) > 80:
            preview = preview[:80] + "..."
        L.i(
            f"TTS first-audio latency ({mode}): {elapsed_ms:.1f} ms | chars={len(text)} | text='{preview}'"
        )

    def _worker(self):
        """Background thread: pulls prompts from the queue and runs TTS inference."""

        while True:
            
            _, _, prompt_item = self._queue.get()
            self._tts_ready.wait()

            generation_id = self._generation_id

            # Prevent audio buffer from growing past buffer-max-seconds
            while self._audio_stream.get_seconds_left() > BUFFER_MAX_SECONDS:
                if self._generation_id != generation_id:
                    break
                time.sleep(1.0)

            # Discard if clear() was called while we were waiting
            if self._generation_id != generation_id:
                self._queue.task_done()
                continue

            phrase_group = prompt_item.phrase_group

            if (
                not prompt_item.use_tts_streaming
                and self._audio_stream.get_seconds_left() < EAGER_THRESHOLD
                and prompt_item.can_eager
                and len(phrase_group.phrases) > 1
            ):
                # Extract first phrase
                prompt_text = phrase_group.phrases[0].text
                # And put remainder back in the "queue", at the front
                remainder = PhraseGroup(phrase_group.phrases[1:])
                self._queue.put((0, next(self._queue_counter), PromptItem(remainder, True, prompt_item.use_tts_streaming)))

            else:
                prompt_text = prompt_item.phrase_group.text
            
            self._prompt_currently_inferencing = prompt_text

            try:
                if prompt_item.use_tts_streaming and Tts.get_info().can_stream:
                    self.generate_streaming_output(prompt_text, phrase_group, generation_id)
                else:
                    self.generate_non_streaming_output(prompt_text, phrase_group, generation_id)

            finally:
                self._prompt_currently_inferencing = ""
                self._queue.task_done()


# TODO: Make configurable maybe:

# Maximum audio buffer duration (seconds) below which eager splitting is able to be triggered
EAGER_THRESHOLD = 1.0

# Maximum audio buffer duration (seconds) before the worker pauses inferencing
BUFFER_MAX_SECONDS = 60 * 10

API_DEMO_HTML_FILE_NAME = "api.html"
STREAMING_CLIENT_DEMO_HTML_FILE_NAME = "streaming-client.html"
COMBINATION_DEMO_HTML_FILE_NAME = "combination.html"