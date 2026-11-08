"""
TTS-model language mappings resolve free-form project codes through
`language_util.map_language`, so region/alias forms reach the same entry as
the base code. Each table must therefore be keyed by base code only.
"""

from types import SimpleNamespace

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_ops import language_util
from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.tts_models import audio_cpp_behavior_fireredtts3
from tts_audiobook_tool.tts_models.audio_cpp_behavior_qwen3 import Qwen3Behavior
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppModelSupport
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel
from tts_audiobook_tool.tts_models.dots_base_model import DotsBaseModel
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel
from tts_audiobook_tool.tts_models.omnivoice_language import resolve_omnivoice_language
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.qwen3_language import QWEN3_LANGUAGE_NAMES


@pytest.mark.parametrize(
    "table",
    [
        MossBaseModel.LANGUAGE_NAMES_BY_CODE,
        QWEN3_LANGUAGE_NAMES,
        audio_cpp_behavior_fireredtts3._TAG_BY_CODE,
    ],
    ids=["moss", "qwen3", "fireredtts3"],
)
def test_model_tables_are_keyed_by_base_code(table: dict[str, str]) -> None:
    # A non-base key ("eng", "zh-cn") would be unreachable: lookups normalize first
    assert all(normalize_language_code(key) == key for key in table)


@pytest.mark.parametrize(
    ("code", "expected"),
    [("en-US", "English"), ("jpn", "Japanese"), ("zh_CN", "Chinese"), ("xx", ""), ("", "")],
)
def test_moss_language_name(code: str, expected: str) -> None:
    # Shared by local MOSS and its audio.cpp / SGLang-Omni variants
    assert MossBaseModel.get_language_name(code) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("zh-TW", ("chinese", "")),
        ("German", ("german", "")),
        # Zhuang: formerly matched the old startswith("zh") special case
        ("zha", ("auto", "no matching Qwen3-TTS language")),
        ("", ("auto", "")),
    ],
)
def test_qwen3_local_language_value(code: str, expected: tuple[str, str]) -> None:
    assert Qwen3BaseModel.resolve_language_value(SimpleNamespace(), code) == expected  # type: ignore[arg-type]


def test_qwen3_audio_cpp_shares_local_table() -> None:
    behavior = Qwen3Behavior.__new__(Qwen3Behavior)
    for code, name in QWEN3_LANGUAGE_NAMES.items():
        assert behavior.resolve_language(code) == name
    assert behavior.resolve_language("xx") == "auto"


class _ChatterboxStub(ChatterboxBaseModel):
    """Loaded-instance stand-in exposing the library's base-code language list."""

    def supported_languages_multi(self) -> list[str]:
        return ["en", "es", "zh"]

    def generate_using_project(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def kill(self) -> None:  # pragma: no cover - unused
        pass


@pytest.mark.parametrize(("code", "blocked"), [("en-US", False), ("spa", False), ("fr", True)])
def test_chatterbox_multilingual_readiness_accepts_region_and_alias_codes(code: str, blocked: bool) -> None:
    project = Project(tts_model_type="chatterbox_local", language_code=code)
    instance = _ChatterboxStub.__new__(_ChatterboxStub)
    assert bool(ChatterboxBaseModel.get_blocking_issues(project, instance)) is blocked
    assert ChatterboxBaseModel.resolve_language_id(code) == normalize_language_code(code)


# --- Former raw-passthrough models ---


@pytest.mark.parametrize(
    ("code", "expected"),
    [
        ("en-US", "en"),  # OmniVoice has no region subtags; audio.cpp rejected "en-US"
        ("zho", "zh"),
        ("ar", "arb"),  # OmniVoice spells Arabic and Filipino differently
        ("arabic", "arb"),
        ("tagalog", "fil"),
        ("nb-NO", "nb"),  # Bokmål stays distinct, though the shared aliases fold it into "no"
        ("nob", "nb"),
        ("no", "no"),
        ("nan", "nan"),  # OmniVoice-only ISO 639-3 ids pass through
        ("", None),
    ],
)
def test_omnivoice_language(code: str, expected: str | None) -> None:
    # Shared by local OmniVoice and the audio.cpp OmniVoice behavior
    assert resolve_omnivoice_language(code) == expected


def test_omnivoice_resolved_ids_exist_in_library() -> None:
    # Every base code the shared alias table produces must resolve to an id the
    # OmniVoice library knows (run under venv-omni)
    lang_map = pytest.importorskip("omnivoice.utils.lang_map")
    for base in language_util._ALIASES_BY_BASE:
        assert resolve_omnivoice_language(base) in lang_map.LANG_IDS, base


def _audio_cpp_support(model_id: str) -> AudioCppModelSupport:
    return AudioCppModelSupport(load_audio_cpp_definitions().models[model_id])


@pytest.mark.parametrize(("code", "blocked"), [("en_US", False), ("", False), ("zh", True), ("ja-JP", True)])
def test_chatterbox_audio_cpp_readiness_checks_server_language_list(code: str, blocked: bool) -> None:
    # The server raises mid-run for languages outside its list (narrower than the
    # Python library's), so readiness blocks them first
    support = _audio_cpp_support("chatterbox_audiocpp")
    project = Project(tts_model_type="chatterbox_audiocpp", language_code=code)
    assert bool(support.behavior.get_blocking_issues(project, {})) is blocked


def test_chatterbox_audio_cpp_sends_base_code() -> None:
    # The server only cuts at "-", so "en_US"/"eng" used to be generation errors
    support = _audio_cpp_support("chatterbox_audiocpp")
    assert support.resolve_request_language("en_US") == "en"
    assert support.resolve_request_language("eng") == "en"


@pytest.mark.parametrize("model_id", ["breeze_tts_2_audiocpp", "echo_tts_audiocpp", "higgs_v3_audiocpp"])
def test_families_that_ignore_language_send_none(model_id: str) -> None:
    assert _audio_cpp_support(model_id).resolve_request_language("en-US") is None


def test_dots_local_sends_base_code() -> None:
    assert DotsBaseModel.resolve_language("en-US") == "en"
    assert DotsBaseModel.resolve_language("") is None
