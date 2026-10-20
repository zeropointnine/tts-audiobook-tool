import os
import posixpath
import ntpath

import pytest

from tts_audiobook_tool.app_support.path_norm import (
    DEFAULT_LOCAL_DIR_NAMES,
    has_drive_or_unc,
    is_absolute_any_grammar,
    join_project_relative,
    looks_foreign,
    make_relative_from_base,
    normalize_native_path,
    normalize_stored_relative_path,
    split_relative,
)


# --- Grammar divergence this module exists to neutralize -------------------

@pytest.mark.parametrize(
    "value",
    [
        r"C:\Users\lee\mybook",
        r"\\fileserver\books\mybook",
        "/home/lee/mybook",
        "voice/narrator.flac",
        "",
    ],
)
def test_is_absolute_any_grammar_never_misses_a_value(value: str) -> None:
    assert is_absolute_any_grammar(value) == (
        posixpath.isabs(value) or ntpath.isabs(value)
    )


@pytest.mark.parametrize("value", [r"C:\Users\lee\mybook", r"\\fileserver\books\mybook"])
def test_is_absolute_any_grammar_catches_what_posixpath_isabs_misses(value: str) -> None:
    # The asymmetry that makes `os.path.isabs` unsafe on serialized values:
    # POSIX reads these as ordinary relative names.
    assert is_absolute_any_grammar(value)
    assert not posixpath.isabs(value)
    assert ntpath.isabs(value)


@pytest.mark.parametrize(
    "value, expected",
    [
        (r"C:\Users\lee\mybook", True),
        (r"\\fileserver\books\mybook", True),
        ("//fileserver/books/mybook", True),
        ("/home/lee/mybook", False),
        ("voice/narrator.flac", False),
        # `ntpath.splitdrive` would report a drive here; we must not.
        ("foo:bar.flac", False),
        ("", False),
    ],
)
def test_has_drive_or_unc(value: str, expected: bool) -> None:
    assert has_drive_or_unc(value) is expected


@pytest.mark.parametrize(
    "value, is_windows, expected",
    [
        # Windows-authored value read on Linux/macOS
        (r"C:\Users\lee\mybook", False, True),
        (r"\\fileserver\books\mybook", False, True),
        # POSIX-authored value read on Windows
        ("/home/lee/mybook", True, True),
        # Same-grammar values are not foreign
        ("/home/lee/mybook", False, False),
        (r"C:\Users\lee\mybook", True, False),
        # Relative values are never foreign
        ("voice/narrator.flac", False, False),
        ("voice/narrator.flac", True, False),
        ("", False, False),
    ],
)
def test_looks_foreign_is_grammar_and_platform_relative(
        value: str, is_windows: bool, expected: bool
) -> None:
    assert looks_foreign(value, is_windows=is_windows) is expected


# --- Project-local relative canonical form ---------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        # Already canonical: unchanged, and reported unchanged
        ("voice/narrator.flac", "voice/narrator.flac"),
        ("narrator.flac", "narrator.flac"),
        # Other grammar's separators
        (r"voice\narrator.flac", "voice/narrator.flac"),
        # Redundant components
        ("./voice//narrator.flac", "voice/narrator.flac"),
        ("voice/narrator.flac/", "voice/narrator.flac"),
        # Legacy absolute voice paths that leaked into portable fields
        (r"C:\Users\lee\mybook\voice\narrator.flac", "narrator.flac"),
        ("//fileserver/books/mybook/voices/narrator.flac", "narrator.flac"),
        ("/home/lee/mybook/voice/narrator.flac", "narrator.flac"),
        (r"C:\Users\lee\mybook\narrator.flac", "narrator.flac"),
        ("/home/lee/mybook/narrator.flac", "narrator.flac"),
        # A bare drive root reduces to nothing rather than to `C:`
        ("C:\\", ""),
        ("C:", ""),
    ],
)
def test_normalize_stored_relative_path(value: str, expected: str) -> None:
    canonical, _changed = normalize_stored_relative_path(value)
    assert canonical == expected


@pytest.mark.parametrize(
    "value, expected_changed",
    [
        ("voice/narrator.flac", False),
        ("narrator.flac", False),
        (r"voice\narrator.flac", True),
        (r"C:\Users\lee\mybook\voice\narrator.flac", True),
        ("", False),
    ],
)
def test_normalize_stored_relative_path_changed_flag(
        value: str, expected_changed: bool
) -> None:
    _canonical, changed = normalize_stored_relative_path(value)
    assert changed is expected_changed


@pytest.mark.parametrize(
    "value",
    [
        "voice/narrator.flac",
        r"voice\narrator.flac",
        r"C:\Users\lee\mybook\voice\narrator.flac",
        "/home/lee/mybook/narrator.flac",
        "./voice//narrator.flac",
        "",
    ],
)
def test_normalize_stored_relative_path_is_idempotent(value: str) -> None:
    once, _ = normalize_stored_relative_path(value)
    twice, changed = normalize_stored_relative_path(once)
    assert twice == once
    assert changed is False


def test_normalize_stored_relative_path_accepts_custom_local_dir_names() -> None:
    canonical, changed = normalize_stored_relative_path(
        r"C:\books\mybook\sfx\ding.wav",
        local_dir_names=("sfx",),
    )
    assert canonical == "ding.wav"
    assert changed


def test_normalize_stored_relative_path_passes_through_non_strings() -> None:
    assert normalize_stored_relative_path(None) == ("", False)  # type: ignore[arg-type]
    assert normalize_stored_relative_path(123) == ("", False)  # type: ignore[arg-type]


# --- split_relative / join_project_relative --------------------------------

@pytest.mark.parametrize(
    "value, expected",
    [
        ("voice/narrator.flac", ["voice", "narrator.flac"]),
        (r"voice\narrator.flac", ["voice", "narrator.flac"]),
        ("./voice//narrator.flac", ["voice", "narrator.flac"]),
        ("narrator.flac", ["narrator.flac"]),
        ("", []),
    ],
)
def test_split_relative(value: str, expected: list[str]) -> None:
    assert split_relative(value) == expected


def test_join_project_relative_produces_native_paths_from_canonical_input() -> None:
    joined = join_project_relative(os.path.join("some", "dir"), "voice/narrator.flac")
    assert joined == os.path.join("some", "dir", "voice", "narrator.flac")


def test_join_project_relative_neutralizes_foreign_grammar_input() -> None:
    # A raw Windows-form relative name must not become one bogus component,
    # and must not discard the base directory.
    joined = join_project_relative(os.path.join("some", "dir"), r"voice\narrator.flac")
    assert joined == os.path.join("some", "dir", "voice", "narrator.flac")


def test_join_project_relative_cannot_discard_base_with_absolute_component() -> None:
    # `os.path.join(base, "/abs/x")` silently returns "/abs/x".
    joined = join_project_relative(os.path.join("some", "dir"), "/abs/x.flac")
    assert joined == os.path.join("some", "dir", "abs", "x.flac")


def test_join_project_relative_with_empty_relative_returns_base() -> None:
    assert join_project_relative(os.path.join("some", "dir"), "") == os.path.join("some", "dir")


# --- normalize_native_path -------------------------------------------------

def test_normalize_native_path_expands_user_before_abspath() -> None:
    result = normalize_native_path("~/books/mybook")
    assert not result.startswith("~")
    assert result == os.path.join(os.path.expanduser("~"), "books", "mybook")


def test_normalize_native_path_is_absolute_and_normalized() -> None:
    result = normalize_native_path("books/./mybook")
    assert os.path.isabs(result)
    assert result == os.path.join(os.getcwd(), "books", "mybook")


def test_normalize_native_path_leaves_foreign_value_untouched() -> None:
    # No cross-OS translation: callers must use `looks_foreign` to reject it.
    assert normalize_native_path(r"C:\Users\lee\mybook") != ""


@pytest.mark.parametrize("value", [""])
def test_normalize_native_path_empty(value: str) -> None:
    assert normalize_native_path(value) == value


# --- make_relative_from_base -----------------------------------------------

def test_make_relative_from_base_returns_canonical_relative_path() -> None:
    base = os.path.join("some", "dir")
    candidate = os.path.join(base, "voice", "narrator.flac")
    assert make_relative_from_base(base, candidate) == "voice/narrator.flac"


def test_make_relative_from_base_rejects_outside_and_equal() -> None:
    base = os.path.join("some", "dir")
    assert make_relative_from_base(base, os.path.join("other", "x.flac")) == ""
    assert make_relative_from_base(base, base) == ""
    assert make_relative_from_base("", os.path.join("some", "x")) == ""


def test_make_relative_from_base_does_not_raise_on_mixed_drives() -> None:
    # `os.path.commonpath` raises ValueError for these; this must not.
    assert make_relative_from_base(r"C:\books\mybook", "/home/lee/mybook/x") == ""
