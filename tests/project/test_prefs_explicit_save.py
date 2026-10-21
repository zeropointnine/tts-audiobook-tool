from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tts_audiobook_tool.app_support.JsonSaveUtil import JsonArtifactType, JsonSaveUtil
from tts_audiobook_tool.constants_config import PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
from tts_audiobook_tool.prefs import PREFS_FILE_NAME, Prefs


def test_prefs_assignment_does_not_save_until_explicitly_requested(
    prefs_path: Path,
) -> None:
    prefs = Prefs()

    prefs.llm_url = "https://example.com"
    assert not prefs_path.exists()

    assert prefs.save() == ""
    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["llm_url"] == "https://example.com"


def test_compound_prefs_change_is_persisted_with_one_explicit_save(
    prefs_path: Path,
    monkeypatch,
) -> None:
    prefs = Prefs()
    writes: list[JsonArtifactType] = []
    real_save = JsonSaveUtil.save

    def record_save(artifact_type, path, payload_factory):
        writes.append(artifact_type)
        return real_save(artifact_type, path, payload_factory)

    monkeypatch.setattr(JsonSaveUtil, "save", record_save)

    prefs.llm_system_prompt = "Be concise."
    prefs.system_prompt_preset = ""

    assert writes == []
    assert prefs.save() == ""
    assert writes == [JsonArtifactType.PREFS]

    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["llm_system_prompt"] == "Be concise."
    assert payload["system_prompt_preset"] == ""


def test_enhance_audio_path_defaults_and_requires_explicit_save(
    prefs_path: Path,
) -> None:
    prefs = Prefs()

    assert prefs.enhance_audio_path == ""
    prefs.enhance_audio_path = "/missing/book.m4b"
    assert not prefs_path.exists()

    assert prefs.save() == ""
    assert json.loads(prefs_path.read_text(encoding="utf-8"))["enhance_audio_path"] == "/missing/book.m4b"
    assert Prefs.load().enhance_audio_path == "/missing/book.m4b"


def test_enhance_audio_path_invalid_value_is_normalized_and_rewritten(
    prefs_path: Path,
) -> None:
    prefs_path.write_text(json.dumps({"enhance_audio_path": 42}), encoding="utf-8")

    prefs = Prefs.load()

    assert prefs.enhance_audio_path == ""
    assert json.loads(prefs_path.read_text(encoding="utf-8"))["enhance_audio_path"] == ""


def test_enhance_audio_path_retains_nonexistent_string(
    prefs_path: Path,
    tmp_path: Path,
) -> None:
    missing = str(tmp_path / "gone" / "book.mp3")
    prefs_path.write_text(json.dumps({"enhance_audio_path": missing}), encoding="utf-8")

    assert Prefs.load().enhance_audio_path == missing


def test_chat_echo_override_defaults_false_and_round_trips(
    prefs_path: Path,
) -> None:
    assert Prefs().chat_echo_override is False

    prefs = Prefs(chat_echo_override=True)
    prefs.save()
    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["chat_echo_override"] is True
    assert Prefs.load().chat_echo_override is True


def test_chat_echo_override_invalid_value_is_normalized_to_false(
    prefs_path: Path,
) -> None:
    prefs_path.write_text(json.dumps({"chat_echo_override": "yes"}), encoding="utf-8")

    prefs = Prefs.load()

    assert prefs.chat_echo_override is False
    assert json.loads(prefs_path.read_text(encoding="utf-8"))["chat_echo_override"] is False


def test_save_gen_log_defaults_false_and_round_trips(
    prefs_path: Path,
) -> None:
    assert Prefs().save_gen_log is False

    prefs = Prefs()
    prefs.save_gen_log = True
    prefs.save()
    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["save_gen_log"] is True

    assert Prefs.load().save_gen_log is True


def test_save_gen_log_invalid_value_is_normalized_to_false(
    prefs_path: Path,
) -> None:
    prefs_path.write_text(json.dumps({"save_gen_log": "yes"}), encoding="utf-8")

    prefs = Prefs.load()

    assert prefs.save_gen_log is False
    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert normalized["save_gen_log"] is False


def test_import_limit_silence_defaults_and_round_trips(
    prefs_path: Path,
) -> None:
    prefs = Prefs()
    assert prefs.import_limit_silence_enabled is False
    assert prefs.import_limit_silence_duration == 0.5
    # Disabled means "don't limit", regardless of the stored duration
    assert prefs.import_max_silence_gap is None

    prefs.import_limit_silence_enabled = True
    prefs.import_limit_silence_duration = 0.25
    prefs.save()

    loaded = Prefs.load()
    assert loaded.import_limit_silence_enabled is True
    assert loaded.import_limit_silence_duration == 0.25
    assert loaded.import_max_silence_gap == 0.25


@pytest.mark.parametrize(
    ("enabled", "duration"),
    [("yes", 0.3), (True, "0.3"), (True, 1.5), (True, -0.1), (True, True)],
)
def test_import_limit_silence_invalid_values_are_normalized(
    prefs_path: Path, enabled, duration,
) -> None:
    # Each invalid field independently falls back to its default; valid ones are kept
    prefs_path.write_text(
        json.dumps({"import_limit_silence_enabled": enabled, "import_limit_silence_duration": duration}),
        encoding="utf-8",
    )

    prefs = Prefs.load()

    assert prefs.import_limit_silence_enabled is (enabled if isinstance(enabled, bool) else False)
    expected_duration = 0.3 if duration == 0.3 and not isinstance(duration, bool) else 0.5
    assert prefs.import_limit_silence_duration == expected_duration
    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert normalized["import_limit_silence_duration"] == expected_duration


def test_hint_mutations_require_explicit_save(
    prefs_path: Path,
) -> None:
    prefs = Prefs()

    prefs.set_hint_true("example")
    assert not prefs_path.exists()
    assert prefs.get_hint("example")

    assert prefs.save() == ""
    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["hints"] == {"example": True}


def test_malformed_json_is_quarantined_before_defaults_are_created(
    prefs_path: Path,
    capsys,
) -> None:
    malformed = '{"llm_url": "https://recover-me.example"'
    prefs_path.write_text(malformed, encoding="utf-8")

    prefs = Prefs.load()

    quarantined = list(prefs_path.parent.glob(f"{PREFS_FILE_NAME}.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == malformed
    assert json.loads(prefs_path.read_text(encoding="utf-8"))["llm_url"] == ""
    assert prefs.llm_url == ""
    output = capsys.readouterr().out
    assert "Preferences recovery warning" in output
    assert str(quarantined[0]) in output


@pytest.mark.parametrize("root_value", [[], "text", 7, True, None])
def test_non_object_json_root_is_quarantined(
    prefs_path: Path,
    root_value,
) -> None:
    original = json.dumps(root_value)
    prefs_path.write_text(original, encoding="utf-8")

    Prefs.load()

    quarantined = list(prefs_path.parent.glob(f"{PREFS_FILE_NAME}.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == original
    assert isinstance(json.loads(prefs_path.read_text(encoding="utf-8")), dict)


def test_invalid_field_is_salvaged_without_quarantining_parseable_object(
    prefs_path: Path,
) -> None:
    prefs_path.write_text(
        json.dumps({
            "llm_url": "https://valid.example",
            "hints": ["invalid"],
        }),
        encoding="utf-8",
    )

    prefs = Prefs.load()

    assert prefs.llm_url == "https://valid.example"
    assert not prefs.get_hint("invalid")
    assert list(prefs_path.parent.glob(f"{PREFS_FILE_NAME}.*.corrupt")) == []
    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert normalized["llm_url"] == "https://valid.example"
    assert normalized["hints"] == {}


def test_read_error_does_not_rename_or_overwrite_preferences(
    prefs_path: Path,
    monkeypatch,
) -> None:
    original = '{"llm_url": "https://valid.example"}'
    prefs_path.write_text(original, encoding="utf-8")

    def fail_open(*_args, **_kwargs):
        raise PermissionError("read denied")

    monkeypatch.setattr("builtins.open", fail_open)

    with pytest.raises(RuntimeError, match="Error reading preferences file"):
        Prefs.load()

    assert prefs_path.read_text(encoding="utf-8") == original
    assert list(prefs_path.parent.glob(f"{PREFS_FILE_NAME}.*.corrupt")) == []


def test_quarantine_failure_preserves_original_and_stops_recovery(
    prefs_path: Path,
    monkeypatch,
) -> None:
    malformed = "{bad json"
    prefs_path.write_text(malformed, encoding="utf-8")

    def fail_quarantine(_file_path: str) -> str:
        raise PermissionError("rename denied")

    monkeypatch.setattr(Prefs, "quarantine_file", staticmethod(fail_quarantine))

    with pytest.raises(RuntimeError, match="could not be preserved"):
        Prefs.load()

    assert prefs_path.read_text(encoding="utf-8") == malformed
    assert list(prefs_path.parent.glob(f"{PREFS_FILE_NAME}.*.corrupt")) == []


def test_default_save_failure_keeps_quarantine_and_uses_in_memory_defaults(
    prefs_path: Path,
    monkeypatch,
    capsys,
) -> None:
    malformed = "{bad json"
    prefs_path.write_text(malformed, encoding="utf-8")
    monkeypatch.setattr(Prefs, "save", lambda _self: "disk full")

    prefs = Prefs.load()

    assert prefs.llm_url == ""
    assert not prefs_path.exists()
    quarantined = list(prefs_path.parent.glob(f"{PREFS_FILE_NAME}.*.corrupt"))
    assert len(quarantined) == 1
    assert quarantined[0].read_text(encoding="utf-8") == malformed
    assert "continuing with in-memory defaults" in capsys.readouterr().out


def test_quarantine_path_is_unique_when_timestamp_collides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    destination = tmp_path / PREFS_FILE_NAME

    class FixedDateTime:
        @classmethod
        def now(cls, _timezone):
            return datetime(2026, 8, 1, 17, 0, tzinfo=timezone.utc)

    monkeypatch.setattr("tts_audiobook_tool.prefs.datetime", FixedDateTime)
    destination.write_text("first", encoding="utf-8")
    first = Prefs.quarantine_file(str(destination))
    destination.write_text("second", encoding="utf-8")
    second = Prefs.quarantine_file(str(destination))

    assert first != second
    assert first.endswith(".corrupt")
    assert second.endswith("-1.corrupt")
    assert Path(first).read_text(encoding="utf-8") == "first"
    assert Path(second).read_text(encoding="utf-8") == "second"


def test_remote_tts_url_defaults_to_empty() -> None:
    assert Prefs().remote_tts_url == ""


def test_remote_tts_url_load_normalizes_unset_values(
    prefs_path: Path,
) -> None:
    # Missing file => missing key => empty
    assert Prefs.load().remote_tts_url == ""

    prefs_path.write_text(json.dumps({"remote_tts_url": ""}), encoding="utf-8")
    assert Prefs.load().remote_tts_url == ""

    prefs_path.write_text(json.dumps({"remote_tts_url": "   "}), encoding="utf-8")
    assert Prefs.load().remote_tts_url == ""

    prefs_path.write_text(json.dumps({"remote_tts_url": 42}), encoding="utf-8")
    assert Prefs.load().remote_tts_url == ""

    prefs_path.write_text(
        json.dumps({"remote_tts_url": "http://example.test:9009"}),
        encoding="utf-8",
    )
    assert Prefs.load().remote_tts_url == "http://example.test:9009"


def test_remote_tts_url_setter_strips_and_allows_empty() -> None:
    prefs = Prefs()
    prefs.remote_tts_url = "  http://example.test:9009  "
    assert prefs.remote_tts_url == "http://example.test:9009"

    prefs.remote_tts_url = "   "
    assert prefs.remote_tts_url == ""


def test_remote_tts_url_round_trips_empty(
    prefs_path: Path,
) -> None:
    prefs = Prefs()
    assert prefs.save() == ""

    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["remote_tts_url"] == ""
    assert "remote_tts_type" not in payload
    assert "remote_tts_model_id" not in payload
    assert "sgl_omni_url" not in payload
    assert Prefs.load().remote_tts_url == ""


def test_remote_connection_preferences_have_no_model_selection(prefs_path: Path) -> None:
    prefs = Prefs(remote_tts_url="http://example.test")
    assert prefs.save() == ""
    payload = json.loads(prefs_path.read_text())
    assert payload["remote_tts_url"] == "http://example.test"
    for key in ("remote_tts_type", "remote_tts_model_id", "remote_tts_selection_url", "sgl_omni_type"):
        assert key not in payload
    assert not hasattr(Prefs.load(), "remote_tts_type")
    prefs.remote_tts_url = "http://other.test"
    assert prefs.save() == ""
    assert Prefs.load().remote_tts_url == "http://other.test"


def test_llm_api_key_env_var_defaults_and_saves_no_secret(
    prefs_path: Path,
    monkeypatch,
) -> None:
    prefs = Prefs()
    assert prefs.llm_api_key_env_var == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR

    monkeypatch.setenv(PREFS_DEFAULT_LLM_API_KEY_ENV_VAR, "sk-test-123")
    assert prefs.llm_api_key == "sk-test-123"

    assert prefs.save() == ""
    payload = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert payload["llm_api_key_env_var"] == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
    assert "llm_api_key" not in payload
    assert "api_key" not in payload


def test_llm_api_key_env_var_round_trips(
    prefs_path: Path,
    monkeypatch,
) -> None:
    prefs = Prefs()
    prefs.llm_api_key_env_var = "MY_PROVIDER_KEY"
    assert prefs.save() == ""

    monkeypatch.delenv("MY_PROVIDER_KEY", raising=False)
    loaded = Prefs.load()
    assert loaded.llm_api_key_env_var == "MY_PROVIDER_KEY"
    assert loaded.llm_api_key == ""

    monkeypatch.setenv("MY_PROVIDER_KEY", "sk-456")
    assert loaded.llm_api_key == "sk-456"


def test_legacy_plaintext_api_key_is_scrubbed_with_warning(
    prefs_path: Path,
    monkeypatch,
    capsys,
) -> None:
    prefs_path.write_text(
        json.dumps({"llm_api_key": "sk-666", "llm_url": "https://keep.example"}),
        encoding="utf-8",
    )

    prefs = Prefs.load()

    # The user warning is deferred to the first main-menu show, not printed here
    assert capsys.readouterr().out == ""
    assert prefs.legacy_llm_api_key_removed is True
    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert "llm_api_key" not in normalized
    assert "api_key" not in normalized
    assert normalized["llm_url"] == "https://keep.example"
    assert normalized["llm_api_key_env_var"] == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
    monkeypatch.delenv(PREFS_DEFAULT_LLM_API_KEY_ENV_VAR, raising=False)
    assert prefs.llm_api_key == ""


def test_legacy_llm_api_key_removed_flag_defaults_false(
    prefs_path: Path,
) -> None:
    assert Prefs().legacy_llm_api_key_removed is False
    assert Prefs.load().legacy_llm_api_key_removed is False


def test_legacy_plaintext_api_key_older_name_is_scrubbed(
    prefs_path: Path,
) -> None:
    prefs_path.write_text(json.dumps({"api_key": "sk-old"}), encoding="utf-8")

    Prefs.load()

    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert "api_key" not in normalized
    assert "llm_api_key" not in normalized


def test_invalid_llm_api_key_env_var_is_normalized_to_default(
    prefs_path: Path,
) -> None:
    prefs_path.write_text(json.dumps({"llm_api_key_env_var": 42}), encoding="utf-8")

    prefs = Prefs.load()

    assert prefs.llm_api_key_env_var == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR
    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert normalized["llm_api_key_env_var"] == PREFS_DEFAULT_LLM_API_KEY_ENV_VAR


def test_saved_directories_are_normalized_or_dropped_on_load(
    prefs_path: Path,
    tmp_path: Path,
) -> None:
    """
    "Last used" directories are machine-local. One saved on another operating
    system cannot exist here, and leaving it in place would send the next file
    dialog somewhere that has no meaning.
    """
    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    prefs_path.write_text(json.dumps({
        "last_voice_dir": str(voice_dir) + os.sep,
        "last_project_dir": "C:\\Users\\lee\\mybook",
        "last_text_dir": str(tmp_path / "deleted-projects"),
    }), encoding="utf-8")

    prefs = Prefs.load()

    assert prefs.last_voice_dir == str(voice_dir)
    assert prefs.last_project_dir == ""
    assert prefs.last_text_dir == ""

    normalized = json.loads(prefs_path.read_text(encoding="utf-8"))
    assert normalized["last_voice_dir"] == str(voice_dir)
    assert normalized["last_project_dir"] == ""
    assert normalized["last_text_dir"] == ""


def test_saved_project_dir_from_another_os_is_left_alone_so_it_can_be_reported(
    prefs_path: Path,
) -> None:
    """
    Unlike the "last used" directories, `project_dir` names the directory the
    app is being asked to open. Blanketing it to empty would hide the problem;
    the app reporting that the directory doesn't exist is what the user needs.
    """
    saved_dir = "C:\\Users\\lee\\mybook"
    prefs_path.write_text(json.dumps({"project_dir": saved_dir}), encoding="utf-8")

    prefs = Prefs.load()

    assert prefs.project_dir == saved_dir
