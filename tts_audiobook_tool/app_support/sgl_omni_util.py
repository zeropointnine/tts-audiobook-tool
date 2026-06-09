from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import json
import httpx
import numpy as np
import soundfile

from tts_audiobook_tool.l import L
from tts_audiobook_tool.app_types import ReadinessIssue, Sound, StreamChunkCallback, StreamEndCallback
from tts_audiobook_tool.constants import COL_DIM_ITALICS
from tts_audiobook_tool.text_util import make_terminal_hyperlink
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType
from tts_audiobook_tool.util import *


class SglOmniUtil:
    """
    Utility functions for interacting with SGL-Omni server
    Maintains some static state
    """

    _base_url: str = ""
    _model_id: str = ""

    @staticmethod
    def get_base_url() -> str:
        return SglOmniUtil._base_url
    
    @staticmethod
    def set_base_url(s: str) -> None:
        SglOmniUtil._base_url = s

    @staticmethod
    def get_model_id() -> str:
        return SglOmniUtil._model_id

    @staticmethod
    def update_model_id() -> None:
        """
        Gets the model id from the models endpoint and saves it to _model_id
        """
        url = SglOmniUtil._base_url + MODELS_PATH
        try:
            response = httpx.get(url, timeout=2.0)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            SglOmniUtil._model_id = ""
            return
        if not isinstance(data, dict):
            SglOmniUtil._model_id = ""
            return

        models = data.get("data")
        if not isinstance(models, list) or len(models) == 0:
            SglOmniUtil._model_id = ""
            return
        
        # SglOmni returns a list but docs say it only ever returns one item
        result = ""
        for model in models:
            id_ = model["id"]
            if id_:
                result = id_
            break # yes rly

        SglOmniUtil._model_id = result

    @staticmethod
    def check_readiness(base_url: str) -> ReadinessIssue | None:
        """
        Verifies server is online and returns issue or None if okay
        """

        # Ping health endpoint
        url = base_url + HEALTH_PATH
        try:
            response = httpx.get(url, timeout=2.0)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return ReadinessIssue(
                "SGL-Omni server",
                f"SGL-Omni health check failed at {url}: {make_error_string(e)}"
            )
        
        if not isinstance(data, dict):
            return ReadinessIssue(
                "SGL-Omni server",
                f"SGL-Omni health check returned {type(data).__name__}, expected JSON object"
            )

        status = data.get("status", "")
        if status != "healthy":
            return ReadinessIssue(
                "SGL-Omni server",
                f"SGL-Omni health check status was {status!r}, expected 'healthy'"
            )

        # Ping models endpoint
        url = base_url + MODELS_PATH
        try:
            response = httpx.get(url, timeout=2.0)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            return ReadinessIssue(
                "SGL-Omni server",
                f"SGL-Omni models check failed at {url}: {make_error_string(e)}"
            )

        if not isinstance(data, dict):
            return ReadinessIssue(
                "SGL-Omni server",
                f"SGL-Omni models check returned {type(data).__name__}, expected JSON object"
            )

        models = data.get("data")
        if not isinstance(models, list) or len(models) == 0:
            return ReadinessIssue(
                "SGL-Omni server",
                "SGL-Omni models check returned no available models"
            )

        return None

    @staticmethod
    def generate(base_url: str, payload: dict, print: bool = False) -> Sound | str:
        """
        POSTs to sgl-omni sound generation endpoint (.../speech) and returns Sound or error string
        """
        
        url = base_url + SPEECH_PATH

        if print:
            s = f"{COL_DIM_ITALICS}Sending generation request to {make_terminal_hyperlink(url)}...{Ansi.RESET}\n"
            s += COL_DIM + pretty_json_string(payload)
            printt(s)

        try:
            with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
                with client.stream("POST", url, json=payload) as response:
                    SglOmniUtil.raise_for_response_error(response)

                    content = response.read()

            return SglOmniUtil.sound_from_encoded_audio(content)

        except Exception as e:
            return make_error_string(e)

    @staticmethod
    def generate_streaming(
            base_url: str,
            payload: dict,
            on_stream_chunk: StreamChunkCallback | None = None,
            on_stream_end: StreamEndCallback | None = None,
            should_print: bool = False,
    ) -> Sound | str:
        """
        POSTs to sgl-omni sound generation endpoint (.../speech), consumes Higgs SSE audio chunks,
        invokes app streaming callbacks, and returns the fully accumulated Sound.
        """

        url = base_url + SPEECH_PATH
        payload = {**payload, "stream": True}

        if should_print:
            s = f"{COL_DIM_ITALICS}Sending streaming generation request to {make_terminal_hyperlink(url)}...{Ansi.RESET}\n"
            s += COL_DIM + pretty_json_string(payload)
            printt(s)

        chunks: list[np.ndarray] = []
        sample_rate = 0

        try:
            with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
                with client.stream("POST", url, json=payload) as response:
                    SglOmniUtil.raise_for_response_error(response)

                    for line in response.iter_lines():
                        if not line:
                            continue
                        if line == "data: [DONE]":
                            break
                        if not line.startswith("data: "):
                            continue

                        event = json.loads(line[len("data: "):])
                        if event.get("finish_reason") == "stop":
                            break

                        audio = event.get("audio") or {}
                        audio_data = audio.get("data")
                        if not audio_data:
                            continue

                        sound = SglOmniUtil.sound_from_encoded_audio(base64.b64decode(audio_data))
                        L.i(f"received streaming audio chunk: {sound.data.size} samples at {sound.sr} Hz")
                        sample_rate = sound.sr or sample_rate
                        chunks.append(sound.data)
                        if on_stream_chunk is not None:
                            on_stream_chunk(sound.data)

            if not chunks:
                return "No audio output"

            if on_stream_end is not None:
                on_stream_end()

            return Sound(np.concatenate(chunks), sample_rate or TtsModelType.SERVER_HIGGS_V3.value.sample_rate)

        except Exception as e:
            return make_error_string(e)

    @staticmethod
    def raise_for_response_error(response: httpx.Response) -> None:
        if response.is_error:
            content = response.read()
            server_error = content.decode("utf-8", errors="replace").strip()
            message = f"{response.status_code} {response.reason_phrase} for url '{response.url}'"
            if server_error:
                message += f"\nServer response:\n{server_error}"
            raise httpx.HTTPStatusError(message, request=response.request, response=response)

    @staticmethod
    def sound_from_encoded_audio(content: bytes) -> Sound:
        data, sample_rate = soundfile.read(
            BytesIO(content),
            dtype="float32",
            always_2d=False,
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = np.asarray(data, dtype=np.float32)

        sr = int(sample_rate)
        if not sr:
            from tts_audiobook_tool.tts import Tts
            sr = Tts.get_type().value.sample_rate
            L.i(f"Samplerate unknown, falling back to TtsInfo value {sr}")

        return Sound(data, sr or TtsModelType.SERVER_HIGGS_V3.value.sample_rate)

    @staticmethod
    def generate_concurrent(endpoint: str, payloads: list[dict], print_request: bool = False) -> list[Sound] | str:

        if len(payloads) == 0:
            return []

        num_workers = len(payloads)
        executor = ThreadPoolExecutor(max_workers=num_workers)
        futures = {
            executor.submit(SglOmniUtil.generate, endpoint, payload, print=print_request): index
            for index, payload in enumerate(payloads)
        }
        results: list[Sound | None] = [None] * len(payloads)

        try:
            for future in as_completed(futures):
                index = futures[future]
                try:
                    result = future.result()
                except Exception as e:
                    executor.shutdown(wait=False, cancel_futures=True)
                    return make_error_string(e)

                if isinstance(result, str):
                    # On any error, immediately return error string 
                    executor.shutdown(wait=False, cancel_futures=True)
                    return result

                results[index] = result

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        return [result for result in results if result is not None]

# ---

GENERATE_TIMEOUT = httpx.Timeout(
    connect=5.0,
    write=10.0,
    read=300.0, # max duration for response or between response updates
    pool=5.0,
)

HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"
SPEECH_PATH = "/v1/audio/speech"
