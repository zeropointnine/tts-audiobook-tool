"""Subheading text owns its internal layout, not the trailing blank line."""
from importlib import import_module

import pytest

from tts_audiobook_tool.text_util import strip_ansi_codes


@pytest.mark.parametrize("module_name, name", [
    ("constants", "ROLLING_CONTINUATION_DESC"),
    ("enhance.enhance_menu", "INTRO"),
    ("menus.options_menu", "DEBUG_SUBHEADING"),
    ("menus.text_menu", "SEG_SUBHEADING"),
    ("menus.text_menu", "DIALOG_SEGMENTATION_DESC"),
    ("menus.text_menu", "SUBSTITUTIONS_DESC"),
    ("menus.text_menu", "UNCOMMON_WORDS_DESC"),
    ("menus.generate_menu", "STRICTNESS_DESC"),
    ("menus.real_time_playback_menu", "REAL_TIME_SUBHEADING"),
    ("menus.section_markers_menu", "SUBLABEL"),
    ("menus.section_markers_menu", "LIMITED_SUBLABEL"),
    ("menus.concat_menu", "LOUDNORM_SUBHEADING"),
    ("menus.concat_menu", "SUBDIVIDE_SUBHEADING"),
    ("menus.concat_menu", "SECTION_BREAK_SUBHEADING"),
    ("menus.concat_menu", "OPEN_AUDIOBOOK_SUBHEADING"),
    ("menus.concat_menu", "HIGH_SHELF_SUBHEADING"),
    ("menus.concat_menu", "LIMIT_SILENCE_GAPS_SUBHEADING"),
    ("menus.concat_menu", "REASON_PAUSES_SUBHEADING"),
    ("menus.concat_menu", "UPSAMPLE_SUBHEADING"),
])
def test_shared_subheading_text_has_no_trailing_linefeed(module_name, name):
    module = import_module(f"tts_audiobook_tool.{module_name}")
    text = strip_ansi_codes(getattr(module, name))
    assert text
    assert not text.endswith(("\r", "\n"))
