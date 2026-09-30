"""Explicit, cached remote-TTS endpoint discovery (never per-sentence polling)."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.parse import urlsplit

import httpx

from tts_audiobook_tool.tts_models.model_spec import TtsBackendKind
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType


@dataclass(frozen=True)
class RemoteTtsIssue:
    code: str
    message: str


@dataclass(frozen=True)
class RemoteTtsSnapshot:
    backend_kind: TtsBackendKind | None = None
    models: tuple[dict[str, Any], ...] = ()
    candidates: tuple[tuple[TtsModelType, str], ...] = ()
    issue: RemoteTtsIssue | None = None


@dataclass(frozen=True)
class _Observation:
    """A snapshot plus the monotonic time it was observed."""

    snapshot: RemoteTtsSnapshot
    checked_at: float


class RemoteTtsDiscovery:
    """Process-wide snapshot per URL; callers choose among candidate pairs.

    An observation stays valid for its time-to-live so that render and action
    paths can re-ask cheaply: within the TTL this is pure cache, after it the
    next ``refresh()`` re-probes with the short revalidation timeout. A cached
    *failure* backs off longer, so an offline server costs at most one quick
    probe per backoff window instead of one stall per menu render.

    ``force=True`` ignores both the TTL and the backoff and uses the full
    timeout, because a caller asking explicitly is about to act on the answer.
    """

    _base_url: ClassVar[str] = ""
    _snapshots: ClassVar[dict[str, _Observation]] = {}
    _timeout: ClassVar[httpx.Timeout] = httpx.Timeout(connect=2.0, read=4.0, write=2.0, pool=2.0)
    # Revalidation happens on render/action paths, so it must fail fast: a host
    # that black-holes packets costs a fraction of a second, not seconds.
    _revalidate_timeout: ClassVar[httpx.Timeout] = httpx.Timeout(connect=0.5, read=2.0, write=2.0, pool=2.0)
    _ttl_seconds: ClassVar[float] = 10.0
    _failure_ttl_seconds: ClassVar[float] = 30.0

    @classmethod
    def _now(cls) -> float:
        return time.monotonic()

    @classmethod
    def _store(cls, snapshot: RemoteTtsSnapshot, url: str | None = None) -> None:
        cls._snapshots[url or cls._base_url] = _Observation(snapshot, cls._now())

    @classmethod
    def set_base_url(cls, url: str | None) -> None:
        cls._base_url = (url or "").strip().rstrip("/")

    @staticmethod
    def validate_url(url: str) -> RemoteTtsIssue | None:
        """Validate endpoint syntax without changing configuration or probing."""
        try:
            # Use the transport's parser, plus urlsplit's bracket/port checks.
            parsed = httpx.URL(url)
            parts = urlsplit(url)
            if parsed.scheme not in ("http", "https") or not parts.hostname:
                raise ValueError("an HTTP or HTTPS URL with a hostname is required")
            if any(character.isspace() for character in parts.hostname):
                raise ValueError("the hostname must not contain whitespace")
            _ = parts.port
        except (httpx.InvalidURL, ValueError) as exc:
            return RemoteTtsIssue("invalid_url", f"Invalid remote TTS server URL: {exc}.")
        return None

    @classmethod
    def get_base_url(cls) -> str:
        return cls._base_url

    @classmethod
    def get_snapshot(cls) -> RemoteTtsSnapshot:
        """Return the last observation without network I/O."""
        if not cls._base_url:
            return RemoteTtsSnapshot(issue=RemoteTtsIssue("not_configured", "Remote TTS server URL is not configured."))
        observation = cls._snapshots.get(cls._base_url)
        if observation is None:
            return RemoteTtsSnapshot(
                issue=RemoteTtsIssue("not_probed", "Remote TTS server has not been checked yet."))
        return observation.snapshot

    @classmethod
    def refresh(cls, force: bool = False) -> RemoteTtsSnapshot:
        """Return the cached observation while fresh, else probe for a new one."""
        url = cls._base_url
        if not url:
            return cls.get_snapshot()
        observation = cls._snapshots.get(url)
        if observation is not None and not force:
            ttl = cls._ttl_seconds if observation.snapshot.issue is None else cls._failure_ttl_seconds
            if cls._now() - observation.checked_at < ttl:
                return observation.snapshot
        snapshot = cls._probe(url, cls._timeout if force else cls._revalidate_timeout)
        cls._store(snapshot, url)
        return snapshot

    @classmethod
    def _probe(cls, url: str, timeout: httpx.Timeout | None = None) -> RemoteTtsSnapshot:
        issue = cls.validate_url(url)
        if issue is not None:
            return RemoteTtsSnapshot(issue=issue)
        try:
            with httpx.Client(timeout=timeout or cls._timeout) as client:
                health_response = client.get(f"{url}/health")
                health_response.raise_for_status()
                health = health_response.json()
                if not isinstance(health, dict):
                    return _issue("invalid_health", "Server health response is not an object.")
                backend = _identify_backend(health)
                if backend is None:
                    return _issue("unrecognized_server", "Server health response does not identify a supported TTS backend.")
                params = {"include_session_options": "true"} if backend is TtsBackendKind.AUDIO_CPP else None
                models_response = client.get(f"{url}/v1/models", params=params)
                models_response.raise_for_status()
                payload = models_response.json()
        except httpx.TimeoutException:
            return _issue("timeout", "Timed out checking the remote TTS server")
        except httpx.HTTPStatusError as exc:
            return _issue("http_error", f"Remote TTS server returned HTTP {exc.response.status_code} for {exc.request.url.path}")
        except httpx.InvalidURL as exc:
            return _issue("invalid_url", f"Invalid remote TTS server URL: {exc}")
        except httpx.RequestError as exc:
            return _issue("unavailable", f"Could not connect to the remote TTS server: {exc}")
        except (ValueError, UnicodeError):
            return _issue("invalid_response", "Remote TTS server returned invalid JSON")

        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            return _issue("invalid_models", "Server model list must contain a data array", backend)
        models = payload["data"]
        if not models:
            return _issue("no_models", "Remote TTS server has no configured models", backend)
        if backend is TtsBackendKind.AUDIO_CPP:
            if not all(_valid_audio_model(model) for model in models):
                return _issue("invalid_models", "audio.cpp model entries require id, family, task, and mode strings and valid session options", backend)
            from tts_audiobook_tool.tts_models.audio_cpp_detection import detect_audio_cpp_models
            candidates = detect_audio_cpp_models(models)
        else:
            if len(models) > 1:
                return _issue("invalid_models", "SGL-Omni must advertise exactly one served model", backend)
            if not all(isinstance(model, dict) and isinstance(model.get("id"), str) and model["id"].strip() for model in models):
                return _issue("invalid_models", "SGL-Omni model entries require nonempty id strings", backend)
            from tts_audiobook_tool.tts_models.sgl_omni_detection import detect_sgl_omni_models
            candidates = detect_sgl_omni_models(models)
        issue = None if candidates else RemoteTtsIssue("no_supported_models", "No configured server models match supported catalog variants")
        return RemoteTtsSnapshot(backend, tuple(models), tuple(candidates), issue)


def _issue(code: str, message: str, backend: TtsBackendKind | None = None) -> RemoteTtsSnapshot:
    return RemoteTtsSnapshot(backend_kind=backend, issue=RemoteTtsIssue(code, message))


def _identify_backend(health: dict[str, Any]) -> TtsBackendKind | None:
    if health.get("status") == "healthy":
        return TtsBackendKind.SGL_OMNI
    if (health.get("status") == "ok"
        and isinstance(health.get("backend"), str) and bool(health["backend"])
        and type(health.get("models")) is int and health["models"] >= 0
        and type(health.get("ui")) is bool
        and type(health.get("ui_management")) is bool):
        return TtsBackendKind.AUDIO_CPP
    return None


def _valid_audio_model(model: Any) -> bool:
    if not isinstance(model, dict):
        return False
    if not all(isinstance(model.get(key), str) and model[key].strip() for key in ("id", "family", "task", "mode")):
        return False
    options = model.get("session_options", {})
    return isinstance(options, dict) and all(
        isinstance(key, str) and isinstance(value, str) for key, value in options.items())
