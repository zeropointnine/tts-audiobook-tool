"""
App-standard path normalization.

Paths that cross a persistence boundary (project.json, prefs.json, the ABR
metadata snapshot) are written by one operating system and may later be read
by another. `os.path` is `posixpath` on Linux/macOS and `ntpath` on Windows,
so a path string stored in the writing OS's grammar is silently re-interpreted
with the reading OS's grammar. The two grammars disagree on basic questions:

    value                         isabs(posix)  isabs(nt)
    C:\\Users\\lee\\mybook         False         True
    \\\\fileserver\\books\\mybook  False         True
    /home/lee/mybook              True          True

This module keeps both grammars visible and provides one place to decide how a
stored path is interpreted. It is deliberately stdlib-only and imports nothing
from the app, so it can be used from `util`, `prefs`, and the project
serialization layer without circular imports.

Two different kinds of stored path, two different normalizers:

- **Project-local relative paths** (voice sample file names, and anything else
  resolved relative to `Project.dir_path`) have a canonical *portable* form:
  `/`-separated, no drive, no leading separator. See
  `normalize_stored_relative_path`.
- **Machine-local absolute paths** (`Project.dir_path`, prefs directories) are
  only ever meaningful on the machine that wrote them. They are normalized for
  the current machine, never translated between machines. See
  `normalize_native_path`.

Documented tradeoff: in a *relative* stored path, a backslash is treated as a
separator. A POSIX file literally named `a\\b.flac` would be reinterpreted.
The app only ever creates such names as `f"{stem}.flac"`, and
`ProjectVoiceUtil.resolve_voice_file_path` keeps a verbatim fallback so the
literal name still resolves.
"""

from __future__ import annotations

import os
import posixpath
import ntpath
import re

# Names that mark the boundary between "machine-local prefix" and
# "project-local remainder" when reducing a foreign absolute path.
DEFAULT_LOCAL_DIR_NAMES = ("voice", "voices")

_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_DRIVE_ROOT_RE = re.compile(r"^[A-Za-z]:$")
_UNC_RE = re.compile(r"^(\\\\|//)")


def is_absolute_any_grammar(value: str) -> bool:
    """
    True if the value is absolute under *either* path grammar.

    Never use `os.path.isabs` on a value that came from a serialized file: it
    answers using the current platform's grammar only.
    """
    if not isinstance(value, str) or not value:
        return False
    return posixpath.isabs(value) or ntpath.isabs(value)


def has_drive_or_unc(value: str) -> bool:
    """
    True for `C:`-style drives and `\\\\server\\share` / `//server/share` UNC roots.

    Deliberately narrower than `ntpath.splitdrive`, which reports a drive for
    any `a:b` string and so would misclassify a POSIX file named `foo:bar.flac`.
    """
    if not isinstance(value, str) or not value:
        return False
    if _DRIVE_RE.match(value):
        return True
    return bool(_UNC_RE.match(value))


def looks_foreign(value: str, *, is_windows: bool | None = None) -> bool:
    """
    True when the value is absolute in a grammar that is *not* the current
    platform's, or carries a drive/UNC root the current platform cannot use.

    Diagnostic only: used to warn the user and to drop machine-local values that
    cannot resolve here. Never used to rewrite a path into a guess.

    `is_windows` is injectable so the behavior is testable on any host.
    """
    if not isinstance(value, str) or not value:
        return False

    on_windows = (os.name == "nt") if is_windows is None else is_windows

    if has_drive_or_unc(value):
        return not on_windows

    if on_windows:
        # `/home/lee/book` is absolute to ntpath but drive-less, so it resolves
        # against the current drive of the process cwd: a different volume.
        return posixpath.isabs(value) and not ntpath.splitdrive(value)[0]

    # A backslash-grammar absolute path is merely a relative name on POSIX.
    return ntpath.isabs(value) and not posixpath.isabs(value)


def split_relative(relative_path: str) -> list[str]:
    """
    Components of a project-local relative path, in either grammar.

    Empty components and `.` are dropped, so `voice//x.flac` and
    `./voice/x.flac` both yield `["voice", "x.flac"]`.
    """
    if not isinstance(relative_path, str) or not relative_path:
        return []
    return [part for part in relative_path.replace("\\", "/").split("/") if part and part != "."]


def normalize_stored_relative_path(
        value: str,
        *,
        local_dir_names: tuple[str, ...] = DEFAULT_LOCAL_DIR_NAMES,
) -> tuple[str, bool]:
    """
    Canonicalize a project-local path as stored in a serialized file.

    Canonical form is `/`-separated, with no drive, UNC root, or leading
    separator. Returns `(canonical, changed)`; `changed` lets the caller warn
    or log that a legacy value was upgraded.

    A value that is absolute or drive-rooted under *either* grammar is a
    machine-local path that leaked into a portable field (older projects do
    this for voice sample names). It is reduced to its project-local remainder:
    the components following the last `local_dir_names` component when present,
    otherwise the final component.
    """
    if not isinstance(value, str) or not value:
        return (value if isinstance(value, str) else ""), False

    unified = value.replace("\\", "/")
    components = split_relative(unified)

    if not is_absolute_any_grammar(value) and not has_drive_or_unc(value):
        canonical = "/".join(components)
        return canonical, canonical != value

    lowered = [component.lower() for component in components]
    boundary = -1
    for index, name in enumerate(lowered):
        if name in local_dir_names:
            boundary = index

    if boundary >= 0 and boundary < len(components) - 1:
        remainder = components[boundary + 1:]
    elif components:
        remainder = components[-1:]
    else:
        return "", True

    # A bare drive root (`C:`) is not a file name, and reducing a path such as
    # `C:\` must not turn it into one.
    remainder = [component for component in remainder if not _DRIVE_ROOT_RE.match(component)]
    if not remainder:
        return "", True

    canonical = "/".join(remainder)
    return canonical, canonical != value


def normalize_native_path(value: str) -> str:
    """
    Normalize a machine-local path for the *current* machine.

    Expands a leading `~` *before* making the path absolute; `abspath` first
    would leave `~` as a literal directory component under the cwd, which a
    later `expanduser` cannot undo.

    Performs no cross-OS translation: a foreign-grammar value stays foreign,
    and callers that need to reject one should use `looks_foreign` first.
    """
    if not isinstance(value, str) or not value:
        return value if isinstance(value, str) else ""
    return os.path.normpath(os.path.abspath(os.path.expanduser(value)))


def join_project_relative(dir_path: str, relative_path: str) -> str:
    """
    Join a canonical project-local relative path onto a native directory path.

    Components are joined one at a time, so a relative path that still carries
    the other grammar's separators cannot collapse into a single bogus filename
    component, and an embedded absolute component cannot silently discard
    `dir_path`.
    """
    parts = split_relative(relative_path)
    if not parts:
        return dir_path
    path = dir_path
    for part in parts:
        path = os.path.join(path, part)
    return path


def make_relative_from_base(base_dir: str, candidate_path: str) -> str:
    """
    Express `candidate_path` relative to `base_dir` in canonical form, or return
    an empty string when it is not underneath it.

    Replaces `os.path.commonpath` comparisons, which raise `ValueError` for
    mixed drives and answer using the current grammar only.
    """
    if not isinstance(base_dir, str) or not base_dir:
        return ""
    if not isinstance(candidate_path, str) or not candidate_path:
        return ""

    base_parts = split_relative(os.path.normpath(base_dir))
    candidate_parts = split_relative(os.path.normpath(candidate_path))

    if len(candidate_parts) <= len(base_parts):
        return ""
    if candidate_parts[:len(base_parts)] != base_parts:
        return ""
    return "/".join(candidate_parts[len(base_parts):])
