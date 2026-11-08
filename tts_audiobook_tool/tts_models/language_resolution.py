"""
How a TTS model resolves the project language code, and the user-facing
messages derived from that.

Every model answers one question — "given this project, what language value
does the request carry, and how sure are we?" — with a `LanguageResolution`.
The request uses `.value`; the pre-flight note (`make_language_note`) and the
readiness blocker (`make_language_blocker`) are built from the same object, so
what the user is told cannot drift from what is sent. See
docs-dev/language-handling.md §6.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum

from tts_audiobook_tool.app_types import ReadinessIssue


class LanguageStatus(Enum):
    # The code maps to a value the model is known to accept
    MAPPED = "mapped"
    # A value is sent, but the app does not know the model's language list
    UNVERIFIED = "unverified"
    # The project code is empty; `value` is the model's default (None or eg "auto")
    NOT_SET = "not_set"
    # The code is set but the model has no equivalent; `value` is the fallback sent
    NO_MATCH = "no_match"
    # The model cannot run with this code (readiness blocks); nothing is sent
    UNSUPPORTED = "unsupported"
    # The model never receives a language
    NOT_USED = "not_used"


@dataclass(frozen=True)
class LanguageResolution:
    status: LanguageStatus
    # Exactly what the request carries; None means the field is omitted
    value: str | None = None
    # Optional model-specific qualifier for the note, eg "applies to IndexTTS2.5 only"
    detail: str = ""

    def with_detail(self, detail: str) -> LanguageResolution:
        return replace(self, detail=detail) if detail else self


NOT_USED = LanguageResolution(LanguageStatus.NOT_USED)


def resolve_from_table(value: str | None, language_code: str, fallback: str | None = None) -> LanguageResolution:
    """
    Resolution for a table lookup result `value` (eg from `language_util.map_language`):
    MAPPED when found, else NOT_SET / NO_MATCH carrying `fallback`.
    """
    if value:
        return LanguageResolution(LanguageStatus.MAPPED, value)
    if not (language_code or "").strip():
        return LanguageResolution(LanguageStatus.NOT_SET, fallback)
    return LanguageResolution(LanguageStatus.NO_MATCH, fallback)


def resolve_unverified(value: str | None) -> LanguageResolution:
    """Resolution for a value sent without knowing the model's language list."""
    if not value:
        return LanguageResolution(LanguageStatus.NOT_SET)
    return LanguageResolution(LanguageStatus.UNVERIFIED, value)


def _source(language_code: str) -> str:
    code = (language_code or "").strip()
    return f'project language code = "{code}"' if code else "project language code not set"


def make_language_note(resolution: LanguageResolution, language_code: str) -> str | None:
    """
    The informational pre-flight note for `resolution`, or None when there is
    nothing to say, eg::

        Language value passed to model: English (project language code = "en-US")
        Language value passed to model: auto (project language code = "xx"; no matching model language)
        Language value passed to model: xx (project language code = "xx"; not checked against the model's language list)
        Language: not used by this model (project language code = "de")

    UNSUPPORTED has no note (the readiness blocker reports it), and NOT_USED
    has none when the project code is empty.
    """
    status = resolution.status
    if status is LanguageStatus.UNSUPPORTED:
        return None
    if status is LanguageStatus.NOT_USED:
        if not (language_code or "").strip():
            return None
        return f"Language: not used by this model ({_source(language_code)})"
    parts = [_source(language_code)]
    if status is LanguageStatus.NO_MATCH:
        parts.append("no matching model language")
    elif status is LanguageStatus.UNVERIFIED:
        parts.append("not checked against the model's language list")
    if resolution.detail:
        parts.append(resolution.detail)
    return f"Language value passed to model: {resolution.value or 'none'} ({'; '.join(parts)})"


def make_language_blocker(
        resolution: LanguageResolution, language_code: str, model_name: str
) -> ReadinessIssue | None:
    """The readiness blocker for an UNSUPPORTED resolution, else None."""
    if resolution.status is not LanguageStatus.UNSUPPORTED:
        return None
    code = (language_code or "").strip()
    if code:
        return ReadinessIssue("language", f'{model_name} does not support project language "{code}"')
    return ReadinessIssue("language", f"{model_name} requires a project language code")


def append_language_note(warnings: list[str], resolution: LanguageResolution, language_code: str) -> list[str]:
    """Appends the note for `resolution` to `warnings` when there is one; returns `warnings`."""
    note = make_language_note(resolution, language_code)
    if note:
        warnings.append(note)
    return warnings
