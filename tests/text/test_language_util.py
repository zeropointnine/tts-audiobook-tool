import pytest

from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.text_ops.whitelist import Whitelist
from tts_audiobook_tool.text_ops.word_equivalence import WordEquivalence


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("en", "en"),
        ("EN", "en"),
        (" en ", "en"),
        ("en-US", "en"),
        ("en_GB", "en"),
        ("eng", "en"),
        ("English", "en"),
        ("es-419", "es"),
        ("spa", "es"),
        ("Spanish", "es"),
        ("fr", "fr"),
        ("zh-Hant", "zh"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_language_code(raw: str, expected: str) -> None:
    assert normalize_language_code(raw) == expected


@pytest.mark.parametrize(
    "language_code", ["en-US", "en_GB", "eng", "English", "es-419", "spa"]
)
def test_subsystems_share_normalization(language_code: str) -> None:
    assert Whitelist.normalize_language_code(language_code) == normalize_language_code(language_code)
    assert WordEquivalence.normalize_language_code(language_code) == normalize_language_code(language_code)
    assert Whitelist.supports_language(language_code) is True
