"""The informational pre-flight note naming the language value a model receives.

Every model that passes a language to inference reports it the same way: the
value actually sent (after any mapping from the project language code), with
the project code it was derived from, eg::

    Language value passed to model: English (project language code = "en")

A value of None or "" means no language reaches the model ("none").
"""
from __future__ import annotations


def make_language_value_note(value: str | None, language_code: str | None, detail: str = "") -> str:
    """The note for `value` derived from project `language_code`, plus an optional
    qualifying `detail` (eg "applies to IndexTTS2.5 only")."""
    code = (language_code or "").strip()
    source = f'project language code = "{code}"' if code else "project language code not set"
    if detail:
        source += f"; {detail}"
    return f"Language value passed to model: {value or 'none'} ({source})"
