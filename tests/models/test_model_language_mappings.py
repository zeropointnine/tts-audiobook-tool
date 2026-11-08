"""
How TTS models resolve the free-form project language code (a
`LanguageResolution`), and the pre-flight note / readiness blocker built from
it. Lookups normalize first, so region/alias forms reach the same entry as the
base code; model tables must therefore be keyed by base code only.
"""

import pytest

from tts_audiobook_tool.project import Project
from tts_audiobook_tool.text_ops import language_util
from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.tts_models import audio_cpp_behavior_fireredtts3
from tts_audiobook_tool.tts_models.audio_cpp_configured import AudioCppModelSupport
from tts_audiobook_tool.tts_models.audio_cpp_definition import load_audio_cpp_definitions
from tts_audiobook_tool.tts_models.chatterbox_base_model import ChatterboxBaseModel
from tts_audiobook_tool.tts_models.dots_base_model import DotsBaseModel
from tts_audiobook_tool.tts_models.language_resolution import (
    NOT_USED, LanguageResolution, LanguageStatus, make_language_blocker, make_language_note,
)
from tts_audiobook_tool.tts_models.moss_base_model import MossBaseModel
from tts_audiobook_tool.tts_models.omnivoice_language import (
    OMNIVOICE_LANGUAGE_IDS, get_omnivoice_language_id, resolve_omnivoice_language,
)
from tts_audiobook_tool.tts_models.qwen3_base_model import Qwen3BaseModel
from tts_audiobook_tool.tts_models.qwen3_language import QWEN3_LANGUAGE_NAMES

S = LanguageStatus


# --- Messages: one formatter for every model ---


@pytest.mark.parametrize(
    ("resolution", "code", "expected"),
    [
        (LanguageResolution(S.MAPPED, "English"), "en-US",
         'Language value passed to model: English (project language code = "en-US")'),
        (LanguageResolution(S.UNVERIFIED, "xx"), "xx",
         'Language value passed to model: xx (project language code = "xx"; '
         "not checked against the model's language list)"),
        (LanguageResolution(S.NO_MATCH, "auto"), "xx",
         'Language value passed to model: auto (project language code = "xx"; no matching model language)'),
        (LanguageResolution(S.NOT_SET), "",
         "Language value passed to model: none (project language code not set)"),
        (LanguageResolution(S.MAPPED, "es", "applies to IndexTTS2.5 only"), "es",
         'Language value passed to model: es (project language code = "es"; applies to IndexTTS2.5 only)'),
        (NOT_USED, "de", 'Language: not used by this model (project language code = "de")'),
        (NOT_USED, "", None),  # nothing to say when there is no code to ignore
        (LanguageResolution(S.UNSUPPORTED), "xx", None),  # the blocker reports it instead
    ],
)
def test_language_note(resolution: LanguageResolution, code: str, expected: str | None) -> None:
    assert make_language_note(resolution, code) == expected


@pytest.mark.parametrize(
    ("code", "expected"),
    [("xx", 'Model does not support project language "xx"'), ("", "Model requires a project language code")],
)
def test_language_blocker(code: str, expected: str) -> None:
    blocker = make_language_blocker(LanguageResolution(S.UNSUPPORTED), code, "Model")
    assert blocker is not None and blocker.short == "language" and blocker.verbose == expected
    assert make_language_blocker(LanguageResolution(S.NO_MATCH), code, "Model") is None


# --- Tables ---


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


# --- Local models ---


@pytest.mark.parametrize(
    ("code", "value", "status"),
    [("en-US", "English", S.MAPPED), ("jpn", "Japanese", S.MAPPED), ("zh_CN", "Chinese", S.MAPPED),
     ("xx", None, S.NO_MATCH), ("", None, S.NOT_SET)],
)
def test_moss_language(code: str, value: str | None, status: LanguageStatus) -> None:
    # Shared by local MOSS and its audio.cpp / SGLang-Omni variants
    assert MossBaseModel.resolve_language_code(code) == LanguageResolution(status, value)


@pytest.mark.parametrize(
    ("code", "value", "status"),
    [
        ("zh-TW", "chinese", S.MAPPED),
        ("German", "german", S.MAPPED),
        # Zhuang: formerly matched the old startswith("zh") special case
        ("zha", "auto", S.NO_MATCH),
        ("", "auto", S.NOT_SET),
    ],
)
def test_qwen3_local_language(code: str, value: str, status: LanguageStatus) -> None:
    project = Project(tts_model_type="qwen3tts_local", language_code=code)
    assert Qwen3BaseModel.resolve_language(project) == LanguageResolution(status, value)


def test_qwen3_audio_cpp_shares_local_table() -> None:
    support = _audio_cpp_support("qwen3tts_audiocpp")
    for code, name in QWEN3_LANGUAGE_NAMES.items():
        assert support.resolve_language_code(code).value == name
    assert support.resolve_language_code("xx") == LanguageResolution(S.NO_MATCH, "auto")


class _ChatterboxStub(ChatterboxBaseModel):
    """Loaded-instance stand-in exposing the library's base-code language list."""

    def supported_languages_multi(self) -> list[str]:
        return ["en", "es", "zh"]

    def generate_using_project(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def kill(self) -> None:  # pragma: no cover - unused
        pass


@pytest.mark.parametrize(
    ("code", "blocked"), [("en-US", False), ("spa", False), ("fr", True), ("", True)]
)
def test_chatterbox_multilingual_readiness_accepts_region_and_alias_codes(code: str, blocked: bool) -> None:
    project = Project(tts_model_type="chatterbox_local", language_code=code)
    assert project.get_model_setting("chatterbox_local", "type").is_multilingual
    instance = _ChatterboxStub.__new__(_ChatterboxStub)
    assert bool(ChatterboxBaseModel.get_blocking_issues(project, instance)) is blocked
    # Without a loaded instance the model's list is unknown: unverified, not blocked
    assert ChatterboxBaseModel.get_blocking_issues(project, None) == []


def test_dots_local_sends_base_code_unverified() -> None:
    def resolve(code: str) -> LanguageResolution:
        return DotsBaseModel.resolve_language(Project(tts_model_type="dots_local", language_code=code))
    assert resolve("en-US") == LanguageResolution(S.UNVERIFIED, "en")
    assert resolve("") == LanguageResolution(S.NOT_SET)


# --- OmniVoice (local and audio.cpp share the resolver) ---


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
        ("xx", None),  # not an OmniVoice id
        ("", None),
    ],
)
def test_omnivoice_language_id(code: str, expected: str | None) -> None:
    assert get_omnivoice_language_id(code) == expected


def test_omnivoice_unknown_code_sends_no_language() -> None:
    # Automatic language selection instead of the audio.cpp generation error
    assert resolve_omnivoice_language("xx") == LanguageResolution(S.NO_MATCH)
    assert _audio_cpp_support("omnivoice_audiocpp").resolve_language_code("xx").value is None
    assert resolve_omnivoice_language("en-US") == LanguageResolution(S.MAPPED, "en")


def test_omnivoice_id_list_matches_library() -> None:
    # The app's copy must match the library's table, and every base code the
    # shared alias table produces must resolve (run under venv-omni)
    lang_map = pytest.importorskip("omnivoice.utils.lang_map")
    assert OMNIVOICE_LANGUAGE_IDS == lang_map.LANG_IDS
    for base in language_util._ALIASES_BY_BASE:
        assert get_omnivoice_language_id(base) is not None, base


# --- audio.cpp families ---


def _audio_cpp_support(model_id: str) -> AudioCppModelSupport:
    return AudioCppModelSupport(load_audio_cpp_definitions().models[model_id])


@pytest.mark.parametrize(
    ("code", "resolution"),
    [
        ("en_US", LanguageResolution(S.MAPPED, "en")),  # the server only cuts at "-"
        ("eng", LanguageResolution(S.MAPPED, "en")),
        ("", LanguageResolution(S.NOT_SET)),  # the session treats it as English
        ("zh", LanguageResolution(S.UNSUPPORTED)),  # not in the server's list
        ("ja-JP", LanguageResolution(S.UNSUPPORTED)),
    ],
)
def test_chatterbox_audio_cpp_language(code: str, resolution: LanguageResolution) -> None:
    support = _audio_cpp_support("chatterbox_audiocpp")
    assert support.resolve_language_code(code) == resolution
    project = Project(tts_model_type="chatterbox_audiocpp", language_code=code)
    blocked = resolution.status is S.UNSUPPORTED
    assert bool(support.get_language_blocking_issues(project)) is blocked


@pytest.mark.parametrize("model_id", ["breeze_tts_2_audiocpp", "echo_tts_audiocpp", "higgs_v3_audiocpp"])
def test_families_that_ignore_language_send_none(model_id: str) -> None:
    assert _audio_cpp_support(model_id).resolve_language_code("en-US") == NOT_USED
