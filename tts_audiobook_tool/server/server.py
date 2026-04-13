from dataclasses import dataclass
import json
import pathlib
import itertools
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.util import make_terminal_hyperlink, printt

_HERE = pathlib.Path(__file__).parent

from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.server.audio_stream import AudioStream
from tts_audiobook_tool.server.audio_stream_http import AudioStreamHttp
from tts_audiobook_tool.tts import Tts


@dataclass
class PromptItem:
    phrase_group: PhraseGroup
    can_eager: bool


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
        _path = self._project.dir_path
        s += f"{COL_ACCENT}{make_terminal_hyperlink(f'file://{_path}', str(_path))}"
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
                            should_segment=data.get("should_segment", False),
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
                else:
                    self.send_error(404)

            def do_GET(self):
                if self.path == "/status":
                    self._respond(_server.status())
                elif self.path == "/streaming-client-demo.html":
                    body = (_HERE / "streaming-client-demo.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif self.path == "/stream":
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
                elif self.path in ("/", "/api-demo.html"):
                    body = (_HERE / "api-demo.html").read_bytes()
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

        with ThreadingHTTPServer((host, port), _Handler) as httpd:
            
            # Print some useful info
            base_url = f"http://{host}:{port}"
            demo_url_1 = f"{base_url}/{API_DEMO_HTML_FILE_NAME}"
            demo_url_2 = f"{base_url}/{STREAMING_CLIENT_DEMO_HTML_FILE_NAME}"

            printt("-" * 80)
            printt(f"{COL_ACCENT}Server listening on {make_terminal_hyperlink(base_url)}")
            printt()
            printt(f"API demo page:\n{make_terminal_hyperlink(demo_url_1)}")
            printt()
            printt(f"Streaming client demo page:\n {make_terminal_hyperlink(demo_url_1)}")
            printt("-" * 80)
            printt()

            # Initialize the TTS model
            threading.Thread(target=_init_tts, daemon=True).start()

            httpd.serve_forever()

    def status(self) -> dict:
        return {
            "status": "initializing" if self._is_initializing else "ready",
            "inferencing": self._prompt_currently_inferencing,
            "audio_buffer": self._audio_stream.get_seconds_left(),
            "num_queued": self._queue.qsize(),
            "stream_clients": self._audio_http_stream.client_count(),
            "local_audio": self._local_audio_enabled,
        }
        # TODO: 
        # "playing": self._prompt_currently_playing
        # Requires extra logic in AudioStream

    def local_audio(self, enabled: bool) -> dict:
        self._local_audio_enabled = enabled
        self._audio_stream.set_is_mute(not enabled)
        return {"local_audio": self._local_audio_enabled}


    def prompt(self, prompt: str, should_segment: bool = False, can_eager: bool = False) -> dict:
        """
        Queues text prompt for TTS inference
        """
        prompt = prompt.strip()
        if not prompt:
            return {"error": "Empty prompt"}

        # Used for convenience response output feedback
        prompt_texts = []

        if should_segment:
            phrase_groups = PhraseGrouper.text_to_groups(prompt, self._project.max_words, self._project.segmentation_strategy, self._project.language_code)
            for phrase_group in phrase_groups:
                prompt_texts.append(phrase_group.text)
                self._queue.put((1, next(self._queue_counter), PromptItem(phrase_group, can_eager)))
        else:
            phrase_group = PhraseGroup( [Phrase(text=prompt, reason=Reason.UNDEFINED)] )
            prompt_texts.append(prompt)
            self._queue.put((1, next(self._queue_counter), PromptItem(phrase_group, False)))
       
        return {
            "input": prompt,
            "prompts": prompt_texts,
            "current_queue_length": self._queue.qsize(),
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

    def _worker(self):
        """Background thread: pulls prompts from the queue and runs TTS inference."""

        while True:
            
            ...
            
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

            if self._audio_stream.get_seconds_left() < EAGER_THRESHOLD and prompt_item.can_eager and len(phrase_group.phrases) > 1:
                # Extract first phrase
                prompt_text = phrase_group.phrases[0].text
                # And put remainder back in the "queue", at the front
                remainder = PhraseGroup(phrase_group.phrases[1:])
                self._queue.put((0, next(self._queue_counter), PromptItem(remainder, True)))

            else:
                prompt_text = prompt_item.phrase_group.text
            
            self._prompt_currently_inferencing = prompt_text

            try:

                # Generate
                result = Tts.get_instance().generate_using_project(
                    self._project, [prompt_text], force_random_seed=False
                )
                if isinstance(result, str):
                    printt(f"* TTS error: {result}")
                    continue
                sound = result[0]

                # Discard result if clear() was called during inference
                if self._generation_id != generation_id:
                    continue

                # Trim silence
                sound = SilenceUtil.trim_silence(sound)[0]
                if sound.data.size == 0:
                    printt(f"* Model output is empty or silence")
                    continue

                # Normalize
                normalized_data = SoundUtil.normalize(sound.data, headroom_db=3.0)
                sound = Sound(normalized_data, sound.sr)

                # Pad end with silence using 'phrase reason'
                if prompt_item.phrase_group.last_reason != Reason.UNDEFINED:
                    sound = SoundUtil.add_silence(sound, prompt_item.phrase_group.last_reason.pause_duration)

                # Done, add to audio stream buffer.
                # HTTP clients are fed automatically via the playback listener.
                self._audio_stream.append(sound)

            finally:
                self._prompt_currently_inferencing = ""
                self._queue.task_done()


# TODO: Make configurable maybe:

# Maximum audio buffer duration (seconds) below which eager splitting is able to be triggered
EAGER_THRESHOLD = 1.0

# Maximum audio buffer duration (seconds) before the worker pauses inferencing
BUFFER_MAX_SECONDS = 60 * 10

API_DEMO_HTML_FILE_NAME = "/api-demo.html"
STREAMING_CLIENT_DEMO_HTML_FILE_NAME = "/streaming-client-demo.html"