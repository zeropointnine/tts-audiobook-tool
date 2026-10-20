"""Subheading text owns its internal layout, not the trailing blank line."""
from importlib import import_module

from tts_audiobook_tool.text_util import strip_ansi_codes


# Every module-level *SUBHEADING constant in these modules is discovered automatically.
SUBHEADING_MODULES = [
    "menus.options_menu",
    "menus.text_menu",
    "menus.real_time_playback_menu",
    "menus.concat_menu",
]

# Subheading/sublabel texts that do not follow the *SUBHEADING naming convention.
# (Not discovered by suffix: e.g. generate_menu.RETRIES_DESC is printed directly and keeps its linefeed.)
OTHER_SUBHEADINGS = [
    ("constants", "ROLLING_CONTINUATION_DESC"),
    ("enhance.enhance_menu", "INTRO"),
    ("menus.text_menu", "DIALOG_SEGMENTATION_DESC"),
    ("menus.text_menu", "SUBSTITUTIONS_DESC"),
    ("menus.text_menu", "UNCOMMON_WORDS_DESC"),
    ("menus.generate_menu", "STRICTNESS_DESC"),
    ("menus.section_markers_menu", "SUBLABEL"),
    ("menus.section_markers_menu", "LIMITED_SUBLABEL"),
]


def test_shared_subheading_text_has_no_trailing_linefeed():
    names = list(OTHER_SUBHEADINGS)
    for module_name in SUBHEADING_MODULES:
        module = import_module(f"tts_audiobook_tool.{module_name}")
        discovered = [name for name in vars(module) if name.isupper() and name.endswith("SUBHEADING")]
        assert discovered, module_name
        names += [(module_name, name) for name in discovered]

    bad = []
    for module_name, name in names:
        text = strip_ansi_codes(getattr(import_module(f"tts_audiobook_tool.{module_name}"), name))
        if not text or text.endswith(("\r", "\n")):
            bad.append(f"{module_name}.{name}")
    assert not bad
