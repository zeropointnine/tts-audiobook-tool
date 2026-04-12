from dataclasses import dataclass
import json
import pathlib
import itertools
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from tts_audiobook_tool.app_types import Sound
from tts_audiobook_tool.constants import *
from tts_audiobook_tool.silence_util import SilenceUtil
from tts_audiobook_tool.sound_util import SoundUtil
from tts_audiobook_tool.util import printt

_HERE = pathlib.Path(__file__).parent

from tts_audiobook_tool.phrase import Phrase, PhraseGroup, Reason
from tts_audiobook_tool.phrase_grouper import PhraseGrouper
from tts_audiobook_tool.prefs import Prefs
from tts_audiobook_tool.project import Project
from tts_audiobook_tool.server.audio_stream import AudioStream
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
            printt("Run tts-audiobook-tool and set up a new project. Then, re-start server.")
            exit(0)
        result = Project.load_using_dir_path(prefs.project_dir)
        if isinstance(result, str):
            printt(f"{COL_ERROR}Error: {result}")
            exit(0)
        self._project = result

        Tts.set_model_params_using_project(self._project) 

        # PriorityQueue allows for inserting items at front
        self._queue: queue.PriorityQueue[tuple[int, int, PromptItem]] = queue.PriorityQueue()

        self._queue_counter = itertools.count()
        self._prompt_currently_inferencing = ""

        self._audio_stream = AudioStream()

        self._is_initializing = True
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
                else:
                    self.send_error(404)

            def do_GET(self):
                if self.path == "/status":
                    self._respond(_server.status())
                elif self.path in ("/", "/tester.html"):
                    body = (_HERE / "tester.html").read_bytes()
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
                ... # xxx printt(f"[server] {self.address_string()} - {format % args}")

        def _init_tts():
            Tts.get_instance()
            self._is_initializing = False
            self._tts_ready.set()

        threading.Thread(target=_init_tts, daemon=True).start()

        with ThreadingHTTPServer((host, port), _Handler) as httpd:
            printt(f"{COL_ACCENT}Server listening on http://{host}:{port}")
            printt()
            httpd.serve_forever()

    def status(self) -> dict:
        return {
            "status": "initializing" if self._is_initializing else "ready",
            "inferencing": self._prompt_currently_inferencing,
            "audio_buffer": self._audio_stream.get_seconds_left(),
            "num_queued": self._queue.qsize(),
        }
    
        # TODO: 
        # "playing": self._prompt_currently_playing
        # Requires extra logic in AudioStream

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
        self._audio_stream.clear()
        return {}

    def _worker(self):
        """Background thread: pulls prompts from the queue and runs TTS inference."""

        while True:
            _, _, prompt_item = self._queue.get()
            self._tts_ready.wait()

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

                # Done, add to audio stream buffer
                self._audio_stream.append(sound)

            finally:
                self._prompt_currently_inferencing = ""
                self._queue.task_done()


# Maximum audio buffer duration (seconds) below which eager splitting is able to be triggered
EAGER_THRESHOLD = 1.0
