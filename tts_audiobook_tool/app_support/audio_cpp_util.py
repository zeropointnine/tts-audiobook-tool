"""audio.cpp HTTP and in-memory WAV conversion for offline voice cloning."""
from __future__ import annotations

import base64
from io import BytesIO
import os

import httpx
import numpy as np
import soundfile

from tts_audiobook_tool.app_types import ReadinessIssue, Sound
from tts_audiobook_tool.constants import COL_DIM_ITALICS
from tts_audiobook_tool.util import print_generation_request, print_init

MAX_REFERENCE_WAV_BYTES = 5 * 1024 * 1024
SPEECH_PATH = "/v1/audio/speech"
MODELS_PATH = "/v1/models"
UNLOAD_ALL_MODELS_PATH = "/v1/tasks/unload_all_models"
# Unload can wait for ongoing inference; don't time out while awaiting teardown.
UNLOAD_TIMEOUT = httpx.Timeout(connect=5.0, write=30.0, read=None, pool=5.0)
GENERATE_TIMEOUT = httpx.Timeout(connect=5.0, write=30.0, read=300.0, pool=5.0)


class AudioCppUtil:
    _base_url = ""

    @staticmethod
    def set_base_url(url: str) -> None:
        AudioCppUtil._base_url = url.strip().rstrip("/")

    @staticmethod
    def get_base_url() -> str:
        return AudioCppUtil._base_url

    @staticmethod
    def unload_all_models(base_url: str) -> str | None:
        """Block until all server models are unloaded; return an error or None."""
        base_url = base_url.strip().rstrip("/")
        if not base_url:
            return "audio.cpp server URL is not configured"
        try:
            response = httpx.post(base_url + UNLOAD_ALL_MODELS_PATH, timeout=UNLOAD_TIMEOUT)
            if response.is_success:
                return None
            try:
                payload = response.json()
            except ValueError:
                payload = None
            if isinstance(payload, dict):
                error = payload.get("error")
                if isinstance(error, dict):
                    message = error.get("message")
                    if isinstance(message, str) and message.strip():
                        return message
            detail = response.text.strip() or response.reason_phrase
            return f"audio.cpp unload failed (HTTP {response.status_code}): {detail}"
        except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
            return f"audio.cpp unload failed: {exc}"

    @staticmethod
    def check_readiness(base_url: str, server_model_id: str = "") -> ReadinessIssue | None:
        if not base_url:
            return ReadinessIssue("audio.cpp server", "audio.cpp server URL is not configured")
        try:
            response = httpx.get(base_url.rstrip("/") + MODELS_PATH, timeout=2.0)
            response.raise_for_status()
            data = response.json()
            ids = [model.get("id") for model in data.get("data", []) if isinstance(model, dict)]
            if not ids or server_model_id and server_model_id not in ids:
                return ReadinessIssue("audio.cpp server", "Configured audio.cpp model not available")
        except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError) as exc:
            return ReadinessIssue("audio.cpp server", f"audio.cpp model discovery failed: {exc}")
        return None

    @staticmethod
    def print_model_init_if_unloaded(
        base_url: str, server_model_id: str, model_description: str,
    ) -> None:
        """Best-effort startup notice; querying metadata never loads a model.

        Residency can change after this snapshot, and False also includes a
        load already in progress. Don't cache it or treat it as readiness.
        """
        if not base_url or not server_model_id:
            return
        try:
            response = httpx.get(base_url.rstrip("/") + MODELS_PATH, timeout=2.0)
            response.raise_for_status()
            payload = response.json()
        except (httpx.HTTPError, httpx.InvalidURL, ValueError):
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return
        matches = [model for model in payload["data"]
                   if isinstance(model, dict) and model.get("id") == server_model_id]
        if len(matches) != 1 or matches[0].get("loaded") is not False:
            return
        path = matches[0].get("path")
        if not isinstance(path, str) or not path.strip():
            return
        # This path belongs to the server; preserve it rather than resolving it
        # against the client's filesystem.
        print_init(f"audio.cpp is loading {model_description}\n{COL_DIM_ITALICS}server path: {path}")

    @staticmethod
    def make_voice_ref(path: str) -> str:
        """Decode FLAC or any libsndfile input, then encode *mono PCM16 WAV* in memory."""
        if not os.path.isfile(path):
            raise ValueError(f"Voice clone sample file not found: {path}")
        try:
            data, sample_rate = soundfile.read(path, dtype="float32", always_2d=True)
        except soundfile.SoundFileError as exc:
            raise ValueError(f"Voice clone sample could not be decoded: {exc}") from exc
        if data.size == 0 or sample_rate <= 0:
            raise ValueError("Voice clone sample contains no audio")
        if not np.isfinite(data).all():
            raise ValueError("Voice clone sample contains non-finite samples")
        samples = np.mean(data, axis=1, dtype=np.float32)
        # Refuse an oversized decoded WAV before spending memory encoding it.
        if samples.size * 2 + 1024 > MAX_REFERENCE_WAV_BYTES:
            raise ValueError("Voice clone WAV exceeds 5 MiB")
        buffer = BytesIO()
        soundfile.write(buffer, samples, sample_rate, format="WAV", subtype="PCM_16")
        wav = buffer.getvalue()
        if len(wav) > MAX_REFERENCE_WAV_BYTES:
            raise ValueError("Voice clone WAV exceeds 5 MiB")
        return "data:audio/wav;base64," + base64.b64encode(wav).decode("ascii")

    @staticmethod
    def sound_from_wav(content: bytes) -> Sound:
        with soundfile.SoundFile(BytesIO(content)) as audio:
            if audio.format != "WAV":
                raise ValueError("audio.cpp response is not WAV")
            rate = audio.samplerate
            data = audio.read(dtype="float32", always_2d=True)
        if data.size == 0:
            raise ValueError("audio.cpp response contains no audio")
        return Sound(np.mean(data, axis=1, dtype=np.float32), rate)

    @staticmethod
    def generate(base_url: str, payload: dict, print_request: bool = False) -> Sound | str:
        # One request per call, deliberately: the server serializes requests on a
        # per-model lock, so pipelining extra ones only queues them (and can trip
        # its busy-timeout/503 guard) without improving throughput.
        if not base_url:
            return "audio.cpp server URL is not configured"
        url = base_url.rstrip("/") + SPEECH_PATH
        if print_request:
            print_generation_request(url, payload)
        try:
            with httpx.Client(timeout=GENERATE_TIMEOUT) as client:
                response = client.post(url, json=payload, headers={"Accept": "audio/wav"})
                if response.is_error:
                    return f"audio.cpp request failed ({response.status_code}): {response.text[:1000]}"
                return AudioCppUtil.sound_from_wav(response.content)
        except (httpx.HTTPError, ValueError, OSError, soundfile.SoundFileError) as exc:
            return f"audio.cpp request failed: {exc}"

