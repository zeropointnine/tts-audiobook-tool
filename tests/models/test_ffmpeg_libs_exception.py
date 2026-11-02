"""
The per-model FFmpeg shared-library exception (`requires_ffmpeg_libs`): the
startup check runs only for declaring models, its message is model-specific,
and the catalog requires each declaration to carry its reason and README section.

No shipped model declares the exception since Phase 8 of the pip-only audio
plan (MOSS dropped it), but the mechanism stays, so most tests use a
synthetic declaring model built from a real catalog entry.
"""

import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from catalog_toml_support import read_catalog
from tts_audiobook_tool.constants import APP_URL
from tts_audiobook_tool.sound.ffmpeg_util import FfmpegUtil, github_heading_anchor
from tts_audiobook_tool.start import Start
from tts_audiobook_tool.tts import Tts
from tts_audiobook_tool.tts_models.model_catalog import CATALOG_PATH, parse_spec
from tts_audiobook_tool.tts_models.tts_model_type import TtsModelType

README_PATH = Path(__file__).resolve().parents[2] / "README.md"

DECLARING = [model for model in TtsModelType.all() if model.value.requires_ffmpeg_libs]

SYNTHETIC_REASON = "its decoder loads the system's FFmpeg shared libraries"
# Any real README heading works for link-building tests
SYNTHETIC_SECTION = "Virtual environment for MOSS-TTS v1.5 (local inference)"


def _entry(data, entry_id):
    return next(model for model in data["models"] if model["id"] == entry_id)


def _declaring_entry() -> dict:
    """The MOSS catalog entry, turned back into a declared exception."""
    entry = _entry(read_catalog(CATALOG_PATH), "moss_local")
    entry["spec"]["requires_ffmpeg_libs"] = True
    entry["spec"]["ui"]["ffmpeg_libs_reason"] = SYNTHETIC_REASON
    entry["spec"]["ui"]["readme_install_section"] = SYNTHETIC_SECTION
    return entry


def _use_local_model(monkeypatch, model) -> None:
    monkeypatch.setattr(Tts, "get_local_model_type", staticmethod(lambda: model))


def _use_declaring_model(monkeypatch):
    spec = parse_spec(_declaring_entry())
    _use_local_model(monkeypatch, SimpleNamespace(id=spec.id, value=spec))
    return spec


def _libs_available(monkeypatch, available: bool, calls: list[str]) -> None:
    def check() -> bool:
        calls.append("check")
        return available

    monkeypatch.setattr(FfmpegUtil, "are_ffmpeg_libraries_available", staticmethod(check))


def test_no_shipped_model_declares_the_exception():
    # Baseline after Phase 8 (MOSS uses waveform/code inputs only). Update this
    # when a model adds an exception, together with its reason and README section.
    assert [model.id for model in DECLARING] == []


@pytest.mark.parametrize("model_id", ["chatterbox_local", "moss_local", "none"])
def test_check_is_a_no_op_for_models_without_the_exception(monkeypatch, model_id):
    # Models that do not declare the exception never probe for system FFmpeg,
    # so a machine without it starts them normally.
    _use_local_model(monkeypatch, TtsModelType.require_by_id(model_id))
    calls: list[str] = []
    _libs_available(monkeypatch, False, calls)
    object.__new__(Start).exit_on_missing_ffmpeg_libs()
    assert calls == []


def test_declaring_model_starts_when_libraries_are_present(monkeypatch):
    _use_declaring_model(monkeypatch)
    calls: list[str] = []
    _libs_available(monkeypatch, True, calls)
    object.__new__(Start).exit_on_missing_ffmpeg_libs()
    assert calls


def test_declaring_model_exits_with_model_specific_message(monkeypatch, capsys):
    _use_declaring_model(monkeypatch)
    _libs_available(monkeypatch, False, [])
    with pytest.raises(SystemExit):
        object.__new__(Start).exit_on_missing_ffmpeg_libs()
    out = capsys.readouterr().out
    assert "MOSS-TTS needs FFmpeg shared libraries" in out
    assert SYNTHETIC_REASON in out
    assert "only MOSS-TTS does" in out
    assert f"{APP_URL}#virtual-environment-for-moss-tts-v15-local-inference" in out


@pytest.mark.parametrize("platform, expected", [
    ("win32", "avcodec-*.dll"),
    ("linux", "apt install ffmpeg"),
    ("darwin", "brew install ffmpeg"),
])
def test_message_gives_the_step_for_the_current_os(platform, expected):
    spec = parse_spec(_declaring_entry())
    message = FfmpegUtil.make_missing_libs_message(spec, platform)
    assert expected in message
    other_steps = {"avcodec-*.dll", "apt install ffmpeg", "brew install ffmpeg"} - {expected}
    assert not any(step in message for step in other_steps)


@pytest.mark.parametrize("model", DECLARING, ids=lambda model: model.id)
def test_readme_section_of_each_exception_exists_and_covers_every_os(model):
    # The startup message links to this heading; it must exist and hold the
    # per-OS FFmpeg instructions the message refers to. (Empty while no model
    # declares the exception.)
    section = model.value.ui["readme_install_section"]
    readme = README_PATH.read_text(encoding="utf-8")
    match = re.search(rf"^#+ {re.escape(section)}\n(.*?)(?=^# |^### )", readme, re.MULTILINE | re.DOTALL)
    assert match, f"README has no heading {section!r}"
    body = match.group(1)
    assert "FFmpeg" in body
    for os_name in ("Windows", "Linux", "macOS"):
        assert os_name in body


def test_github_heading_anchor():
    assert github_heading_anchor("Virtual environment for MOSS-TTS v1.5 (local inference)") == \
        "virtual-environment-for-moss-tts-v15-local-inference"
    assert github_heading_anchor("Step 4 (Windows only)") == "step-4-windows-only"


@pytest.mark.parametrize("key", ["ffmpeg_libs_reason", "readme_install_section"])
def test_catalog_requires_reason_and_readme_section_for_the_exception(key):
    entry = _declaring_entry()
    del entry["spec"]["ui"][key]
    with pytest.raises(ValueError, match="required when requires_ffmpeg_libs is true"):
        parse_spec(entry)

    entry["spec"]["ui"][key] = "  "
    with pytest.raises(ValueError, match="must not be empty"):
        parse_spec(entry)


def test_catalog_rejects_exception_fields_on_server_models():
    entry = _entry(read_catalog(CATALOG_PATH), "fish_s2_audiocpp")
    entry["spec"]["ui"]["ffmpeg_libs_reason"] = "x"
    with pytest.raises(ValueError, match="unsupported field"):
        parse_spec(entry)
