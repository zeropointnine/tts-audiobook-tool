"""Shared UI ellipsizing removes whitespace adjacent to inserted dots only."""
import pytest

from tts_audiobook_tool.util import ellipsize, ellipsize_path_for_menu


@pytest.mark.parametrize("value,length,from_start,expected", [
    ("abcdef", 5, False, "ab..."),
    ("abcdef", 5, True, "...ef"),
    ("ab \t cdef", 8, False, "ab..."),
    ("abcd \t ef", 8, True, "...ef"),
    ("ab\ncdef", 6, False, "ab..."),
    ("abcd\nef", 6, True, "...ef"),
    (" ab cdef", 7, False, " ab..."),
    ("abcdef ", 6, True, "...ef "),
    (" ab ", 4, False, " ab "),
    (" ab ", 5, True, " ab "),
    ("", 5, False, ""),
])
def test_ellipsize_trims_only_next_to_inserted_dots(value, length, from_start, expected):
    # Cut-boundary whitespace is dropped, but spaces within or outside the retained text stay.
    assert ellipsize(value, length, from_start) == expected


def test_menu_path_has_no_space_after_dots():
    # This path cuts immediately before a space, as in a voice list's LoRA label.
    path = "/mnt/x/w/_loras/brina palencia - spice and wolf/vol 14/lora/checkpoint-1000"
    assert ellipsize_path_for_menu(path) == "...and wolf/vol 14/lora/checkpoint-1000"
