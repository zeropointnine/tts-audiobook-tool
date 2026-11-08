from types import SimpleNamespace

import pytest

from tts_audiobook_tool.app_types import Strictness
from tts_audiobook_tool.model_worker import ModelWorker
from tts_audiobook_tool.text_ops import language_util
from tts_audiobook_tool.text_ops.language_util import normalize_language_code
from tts_audiobook_tool.text_ops.phrase_segmenter import PhraseSegmenter
from tts_audiobook_tool.text_ops.prompt_normalizer import PromptNormalizer
from tts_audiobook_tool.text_ops.text_normalizer import TextNormalizer
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
        ("zho", "zh"),
        ("Japanese", "ja"),
        ("kor", "ko"),
        ("yue", "yue"),
        ("", ""),
        ("   ", ""),
    ],
)
def test_normalize_language_code(raw: str, expected: str) -> None:
    assert normalize_language_code(raw) == expected


# One row per free-form code, pinning every language-class predicate at once so
# they cannot drift apart. Columns: is_english, is_spanish, is_cjk,
# is_validation_supported. Includes codes a prefix (startswith) match would get
# wrong: Javanese "jav" and Konkani "kok" are not Japanese/Korean.
@pytest.mark.parametrize(
    ("code", "english", "spanish", "cjk", "validation"),
    [
        ("en", True, False, False, True),
        ("en-us", True, False, False, True),
        ("en_GB", True, False, False, True),
        ("eng", True, False, False, True),
        ("es", False, True, False, True),
        ("es-mx", False, True, False, True),
        ("spa", False, True, False, True),
        ("zh", False, False, True, False),
        ("zh-cn", False, False, True, False),
        ("zh_CN", False, False, True, False),
        ("zho", False, False, True, False),
        ("yue", False, False, True, False),
        ("ja-jp", False, False, True, False),
        ("ko", False, False, True, False),
        ("jav", False, False, False, True),
        ("kok", False, False, False, True),
        ("fr", False, False, False, True),
        ("", False, False, False, True),
    ],
)
def test_language_class_predicates(
    code: str, english: bool, spanish: bool, cjk: bool, validation: bool
) -> None:
    assert language_util.is_english(code) is english
    assert language_util.is_spanish(code) is spanish
    assert language_util.is_cjk(code) is cjk
    assert language_util.is_validation_supported(code) is validation


@pytest.mark.parametrize(
    "language_code", ["en-US", "en_GB", "eng", "English", "es-419", "spa"]
)
def test_subsystems_share_normalization(language_code: str) -> None:
    assert Whitelist.normalize_language_code(language_code) == normalize_language_code(language_code)
    assert WordEquivalence.normalize_language_code(language_code) == normalize_language_code(language_code)
    assert Whitelist.supports_language(language_code) is True


# --- Consumer wiring: a region-qualified code behaves like its base code ---


def test_text_normalizer_applies_english_rules_for_region_code() -> None:
    # The English-specific pass (contractions, numbers) must not be skipped
    # just because the project code carries a region suffix
    source = "I can't pay 25 dollars"
    assert TextNormalizer.normalize_source(source, "en-US") == TextNormalizer.normalize_source(source, "en")
    assert TextNormalizer.normalize_source(source, "en-US") != TextNormalizer.normalize_source(source, "")


def test_recommended_strictness_for_region_code() -> None:
    assert Strictness.get_recommended_default("en-US") == Strictness.get_recommended_default("en")


def test_pysbd_receives_base_code() -> None:
    # Spanish pysbd rules keep "Sra. López" together; the English fallback
    # (formerly used for any region-qualified code) splits after "Sra."
    text = "Vino con la Sra. López. Luego se fue."
    spanish = PhraseSegmenter.string_to_sentence_strings(text, "es")
    assert PhraseSegmenter.string_to_sentence_strings(text, "es-MX") == spanish
    assert PhraseSegmenter.string_to_sentence_strings(text, "en") != spanish


def test_prompt_number_expansion_resolves_alias_code() -> None:
    # num2words' own fallback would truncate "spa" to the unknown "sp"
    assert PromptNormalizer.normalize_prompt("Tengo 12 gatos", "spa") == "Tengo doce gatos"


def test_whisper_command_receives_base_code(monkeypatch: pytest.MonkeyPatch) -> None:
    # Whisper only recognizes base codes; "en-US" used to be dropped as
    # unsupported, silently falling back to auto-detect
    queued: list = []
    monkeypatch.setattr(ModelWorker, "_start_with_console_handler", classmethod(lambda cls, handler: ""))
    monkeypatch.setattr(ModelWorker, "_command_queue", SimpleNamespace(put=queued.append))
    monkeypatch.setattr(ModelWorker, "_cancellation_event", None)
    monkeypatch.setattr(ModelWorker, "_active_operation_id", None)
    monkeypatch.setattr(
        ModelWorker, "_wait_for_blocking_result_with_handler",
        classmethod(lambda cls, *args, **kwargs: None),
    )
    prefs = SimpleNamespace(stt_variant=SimpleNamespace(id="v"), stt_config=SimpleNamespace(id="c"))

    ModelWorker.transcribe_audio_blocking(prefs, object(), language="en-US")

    assert queued[0].language == "en"
