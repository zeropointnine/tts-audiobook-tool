"""Target duration for sample-less AuK speech, ported from upstream AuK.

Without reference audio, SGL-Omni's AuK speech route requires an explicit
`stage_params.auk_engine.gen_seconds` (its own estimate needs a reference clip
plus its transcript), and upstream's core inference (`get_gen_duration()` in
`src/auk/infer/infer_auk.py`) has no text-only estimate either. Upstream's
Prompt Enhancer does: `_estimate_f5_instruct_duration()` in
`src/auk/infer/pe.py`, a fixed baseline "calibrated for ordinary speaking",
which the Prompt Enhancer itself falls back to when its LLM duration-adjust
step is unavailable. This module is that baseline only (no LLM step), so the
app adds no estimation rule of its own.

Source: Tencent-Hunyuan/AuK, `src/auk/infer/pe.py` @ 944895c and constants from
`src/auk/infer/pe.config.yaml` (`runtime.duration`) @ 903002c. Keep this a
faithful port; re-sync with upstream rather than tuning values here.
"""
from __future__ import annotations

import re

# pe.config.yaml runtime.duration.seconds_per_utf8_byte
SECONDS_PER_UTF8_BYTE: dict[str, float] = {"en": 0.0656, "zh": 0.0803}
# pe.config.yaml runtime.duration.f5
F5_SAMPLE_RATE = 24000
F5_HOP_LENGTH = 256
F5_SHORT_TEXT_BYTE_THRESHOLD = 10
F5_SHORT_TEXT_SPEED = 0.3

_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_EN_WORD_RE = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)*")


def _normalize_language(language: object) -> str | None:
    value = str(language or "").strip().lower()
    if value.startswith("en"):
        return "en"
    if value.startswith(("zh", "cn")) or "chinese" in value:
        return "zh"
    return None


def _resolve_text_language(text: str, declared_language: object = None) -> str:
    num_zh = len(_CJK_RE.findall(text))
    num_en = len(_EN_WORD_RE.findall(text))
    if num_zh and not num_en:
        return "zh"
    if num_en and not num_zh:
        return "en"
    if num_zh and num_en:
        return "zh" if num_zh * 0.21 >= num_en * 0.30 else "en"
    return _normalize_language(declared_language) or "zh"


def _utf8_byte_count(text: str) -> int:
    return len(text.encode("utf-8")) if text.strip() else 0


def _tts_utf8_weight(text: str, language: object) -> float:
    """Seconds: each character's UTF-8 bytes at its script language's rate.
    Characters that are neither CJK nor ASCII letters (punctuation, digits,
    accented letters) take the language of the nearest preceding, else
    following, classified character, else the text's overall language."""
    if not text.strip():
        return 0.0
    fallback = _resolve_text_language(text, declared_language=language)
    script_languages = [
        "zh" if _CJK_RE.fullmatch(char) else ("en" if char.isascii() and char.isalpha() else None) for char in text
    ]
    next_languages: list[str | None] = [None] * len(text)
    next_language = None
    for index in range(len(text) - 1, -1, -1):
        if script_languages[index]:
            next_language = script_languages[index]
        next_languages[index] = next_language
    weight = 0.0
    previous_language = None
    for index, char in enumerate(text):
        char_language = script_languages[index]
        if char_language is None:
            char_language = previous_language or next_languages[index] or fallback
        else:
            previous_language = char_language
        weight += len(char.encode("utf-8")) * SECONDS_PER_UTF8_BYTE[char_language]
    return weight


def estimate_instruct_tts_seconds(text: str, language: object = None) -> float:
    """Upstream's ordinary-speaking duration for `text` (seconds, before any
    speed adjustment). `language` is only the fallback for text with no CJK
    characters or English words; it is normalized to "en"/"zh" as upstream does,
    and anything else falls back to the Chinese rate, also as upstream does."""
    weight = _tts_utf8_weight(text, language)
    local_speed = F5_SHORT_TEXT_SPEED if _utf8_byte_count(text) < F5_SHORT_TEXT_BYTE_THRESHOLD else 1.0
    frames = int(weight * F5_SAMPLE_RATE / F5_HOP_LENGTH / local_speed)
    return frames * F5_HOP_LENGTH / F5_SAMPLE_RATE
