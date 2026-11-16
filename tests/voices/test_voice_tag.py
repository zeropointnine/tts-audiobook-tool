"""
Sound segment file name voice tags (ProjectVoiceUtil.make_voice_tag): concat
output names summarize voices by these tags, so distinct voices should yield
distinct, non-empty tags.
"""
from __future__ import annotations

from tts_audiobook_tool.project_support.project_voice_util import (
    VOICE_TAG_MAX_LEN, ProjectVoiceUtil,
)
from tts_audiobook_tool.project_support.sound_segment_util import SoundSegmentUtil

tag = ProjectVoiceUtil.make_voice_tag


def test_short_names_are_plainly_sanitized():
    assert tag("alice (calm)") == "alice_calm"
    assert tag("v1.2.3") == "v1_2_3"


def test_accents_fold_to_ascii():
    assert tag("Zoë") == "Zoe"
    assert tag("Ångström_é") == "Angstrom_e"


def test_long_names_sharing_a_prefix_stay_distinct():
    # Plain truncation made both "narrator_chapter_one_clean_tak"
    a = tag("narrator_chapter_one_clean_take_A")
    b = tag("narrator_chapter_one_clean_take_B")
    assert a != b
    assert len(a) <= VOICE_TAG_MAX_LEN and len(b) <= VOICE_TAG_MAX_LEN
    assert a.startswith("narrator_chapter_one_cle_")


def test_name_exactly_at_the_limit_is_not_hashed():
    name = "a" * VOICE_TAG_MAX_LEN
    assert tag(name) == name


def test_unsanitizable_names_get_a_distinct_non_empty_tag():
    a, b = tag("日本語"), tag("中文")
    assert a and b and a != b
    assert a.startswith("v_")


def test_unsanitizable_voice_keeps_segment_file_name_parsable():
    # An empty "[]" voice tag used to stop tag parsing, losing the voice and
    # the word-error count that follows it
    voice = ProjectVoiceUtil.make_voice_file_name_tag("日本語.wav", "qwen3")
    segment = SoundSegmentUtil.make_from_file_name(
        f"[00001] [0123456789abcdef] [qwen3] [{voice}] [3] Hello.flac"
    )
    assert segment is not None
    assert (segment.voice, segment.num_errors) == (voice, 3)
