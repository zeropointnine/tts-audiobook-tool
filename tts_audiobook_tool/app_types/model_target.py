"""
`ModelTarget`: where a model (or LoRA adapter) comes from — app nomenclature "target".

A target is one of two kinds:

- `TargetKind.HF`: a Hugging Face repo id (`owner/repo`), or a directory inside a
  repo, written `owner/repo/sub/dir` (the `HfFileSystem` path form, minus `hf://`).
  A plain repo id never has more than two segments, so the forms cannot collide.
- `TargetKind.LOCAL`: a local directory, stored as entered (absolute when entered
  through the app).

On disk a target is a pair of sibling strings: `target` (the value) and
`target_type` (the kind id, `"hf"` | `"local"`). A missing or invalid `target_type`
is legacy data from before the kind was recorded; `ModelTarget.from_stored` then
infers the kind the way the app used to decide it at runtime
(`infer_legacy_kind`) and marks the result `inferred`.

Instances are immutable. `inferred` is transient: never serialized, and ignored by
equality and hashing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import os
from pathlib import Path
import re
from typing import Any


TARGET_FIELD = "target"
TARGET_TYPE_FIELD = "target_type"

# Files whose presence makes a directory look like a model or adapter directory
DEFAULT_LOCAL_FILE_MARKERS: tuple[str, ...] = ("config.json", "model.safetensors", "adapter_config.json")


class TargetKind(StrEnum):
    HF = "hf"
    LOCAL = "local"

    @property
    def label(self) -> str:
        """User-facing name, eg for "(currently: ...)" lines."""
        return "Hugging Face repo" if self is TargetKind.HF else "local directory"

    @staticmethod
    def get_by_id(value: Any) -> TargetKind | None:
        try:
            return TargetKind(value)
        except ValueError:
            return None


def is_hf_repo_id_syntax(s: str) -> bool:
    """Whether `s` is a syntactically valid plain repo id (`owner/repo` or `repo`)."""
    from huggingface_hub.utils._validators import validate_repo_id
    from huggingface_hub.errors import HFValidationError

    try:
        validate_repo_id(s)
    except HFValidationError:
        return False
    return True


def split_hf_subfolder(s: str) -> tuple[str, str] | None:
    """
    Splits `owner/repo/sub/dir` into ("owner/repo", "sub/dir").
    Returns None for anything else, including a plain repo id. Syntax only.
    """
    if "\\" in s:
        return None
    parts = s.split("/")
    if len(parts) < 3 or any(part in ("", ".", "..") for part in parts):
        return None
    repo_id = "/".join(parts[:2])
    if not is_hf_repo_id_syntax(repo_id):
        return None
    return repo_id, "/".join(parts[2:])


def is_hf_target_syntax(s: str) -> bool:
    """Whether `s` is a valid hf target value: a plain repo id or `owner/repo/sub/dir`."""
    return is_hf_repo_id_syntax(s) or split_hf_subfolder(s) is not None


_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")


def looks_like_local_path(s: str) -> bool:
    """Whether `s` is unmistakably written as a filesystem path (absolute, home- or
    dot-relative, or with backslashes), as opposed to something repo-shaped."""
    return (
        s.startswith(("/", "~", "."))
        or "\\" in s
        or bool(_WINDOWS_DRIVE_RE.match(s))
    )


_HF_URL_PREFIXES = ("https://huggingface.co/", "http://huggingface.co/", "huggingface.co/")
_HF_PATH_PREFIX = "hf://"


def parse_hf_input(text: str) -> tuple[str, str]:
    """
    An hf target value from user input, as (value, error): `owner/repo` or
    `owner/repo/sub/dir`, also accepting the same as a pasted `hf://` path or
    huggingface.co URL (including a `/tree/main/sub/dir` directory URL).
    Other revisions aren't supported: a target always uses the main branch.
    """
    value = text.strip().strip("\"'").strip()
    lowered = value.lower()
    is_url = False
    if lowered.startswith(_HF_PATH_PREFIX):
        value = value[len(_HF_PATH_PREFIX):].lstrip("/")
    else:
        for prefix in _HF_URL_PREFIXES:
            if lowered.startswith(prefix):
                value, is_url = value[len(prefix):].lstrip("/"), True
                break
    # Only a trailing slash is dropped: a leading one makes an absolute path, never a repo id
    value = value.rstrip("/")
    if is_url:
        parts = value.split("/")
        if len(parts) >= 3 and parts[2] in ("tree", "blob", "resolve"):
            if parts[2] != "tree":
                return "", "Enter a repo or a directory in it, not a file"
            if len(parts) < 4 or parts[3] != "main":
                return "", "Only the main revision of a Hugging Face repo can be used"
            value = "/".join(parts[:2] + parts[4:])
    if not value:
        return "", ""
    if not is_hf_target_syntax(value):
        return "", f"Not a valid Hugging Face repo id: {value}"
    return value, ""


def infer_legacy_kind(value: str) -> TargetKind:
    """
    The kind of an untyped (legacy) target string, decided the way the app used to
    decide it at runtime, so inferring never changes what a saved target loads:
    an existing local directory wins; otherwise hf syntax (plain or subfolder form)
    means hf; anything else is a local path.
    """
    if os.path.isdir(value):
        return TargetKind.LOCAL
    if is_hf_target_syntax(value):
        return TargetKind.HF
    return TargetKind.LOCAL


@dataclass(frozen=True)
class ModelTarget:
    """A typed target. See the module docstring."""

    kind: TargetKind
    value: str
    # The kind was inferred from a legacy untyped string (see `from_stored`)
    inferred: bool = field(default=False, compare=False)

    def __post_init__(self) -> None:
        kind = TargetKind.get_by_id(self.kind)
        if kind is None:
            raise ValueError(f"Invalid target kind: {self.kind!r}")
        object.__setattr__(self, "kind", kind)
        if not isinstance(self.value, str) or not self.value.strip():
            raise ValueError("Target must be a non-empty string")
        if kind is TargetKind.HF and not is_hf_target_syntax(self.value):
            raise ValueError(f"Invalid Hugging Face repo id: {self.value}")

    @staticmethod
    def hf(value: str) -> ModelTarget:
        return ModelTarget(TargetKind.HF, value)

    @staticmethod
    def local(path: str) -> ModelTarget:
        return ModelTarget(TargetKind.LOCAL, path)

    @staticmethod
    def from_stored(value: Any, kind: Any) -> ModelTarget | None:
        """
        The target of a stored `target` / `target_type` pair. None when `value` is
        missing, not a string, or blank (ie, no target). A missing or invalid
        `kind`, or one that does not fit the value (eg "hf" for a Windows path), is
        replaced by `infer_legacy_kind`, and the result is marked `inferred`.
        """
        if not isinstance(value, str) or not value.strip():
            return None
        typed_kind = TargetKind.get_by_id(kind)
        if typed_kind is not None:
            try:
                return ModelTarget(typed_kind, value)
            except ValueError:
                pass
        return ModelTarget(infer_legacy_kind(value), value, inferred=True)

    @staticmethod
    def infer(value: str) -> ModelTarget:
        """A target typed by `infer_legacy_kind`, for untyped input such as a command-line
        argument. Raises ValueError for a blank value."""
        result = ModelTarget.from_stored(value, None)
        if result is None:
            raise ValueError("Target must be a non-empty string")
        return result

    def to_stored(self) -> tuple[str, str]:
        """The stored (`target`, `target_type`) pair."""
        return self.value, self.kind.value

    @property
    def is_hf(self) -> bool:
        return self.kind is TargetKind.HF

    @property
    def is_local(self) -> bool:
        return self.kind is TargetKind.LOCAL

    @property
    def repo_id(self) -> str:
        """The repo id of an hf target ("" for a local one)."""
        if not self.is_hf:
            return ""
        split = split_hf_subfolder(self.value)
        return split[0] if split else self.value

    @property
    def subfolder(self) -> str:
        """The directory inside the repo of an hf target ("" for a plain repo id or a local target)."""
        if not self.is_hf:
            return ""
        split = split_hf_subfolder(self.value)
        return split[1] if split else ""

    def describe(self) -> str:
        """eg `k2-fsa/OmniVoice (Hugging Face repo)`."""
        return f"{self.value} ({self.kind.label})"

    def display_name(self) -> str:
        """
        Short name: the directory name, qualified by its parent when it is a training
        checkpoint directory (eg `checkpoint-900`) or a generic name like `lora`. A
        generic parent is skipped for the nearest non-generic ancestor
        (eg `fw-leda/lora/checkpoint-step-000500` -> `fw-leda-checkpoint-step-000500`).
        For hf targets: the repo name, and for a directory inside the repo, the repo
        name joined with that directory's name by the same rules, so that two
        directories in one repo get different names
        (eg `owner/repo/voices/herman` -> `repo-herman`, `owner/PaigeVoice/lora` -> `PaigeVoice-lora`).
        """
        if self.is_local:
            return _qualified_dir_name(self.value)
        repo_name = self.repo_id.split("/")[-1]
        if not self.subfolder:
            return repo_name
        # The repo is the topmost ancestor, so a generic or checkpoint name is qualified by it
        name = _qualified_dir_name(f"{repo_name}/{self.subfolder}")
        return name if name.startswith(f"{repo_name}-") else f"{repo_name}-{name}"

    def resolve_local_dir(self, file_names: list[str] | None = None, revision: str | None = None) -> str:
        """
        A local directory holding the target: a local target's own path, or the hf
        repo (or its subfolder) downloaded into the hf cache. `file_names` limits the
        download to those files (relative to the subfolder); otherwise the whole
        subfolder (or repo) is downloaded. Raises on download failure.
        """
        if self.is_local:
            return self.value
        from huggingface_hub import snapshot_download

        subfolder = self.subfolder
        prefix = f"{subfolder}/" if subfolder else ""
        if file_names is not None:
            allow_patterns: list[str] | None = [prefix + name for name in file_names]
        else:
            allow_patterns = [f"{prefix}*"] if subfolder else None
        kwargs: dict[str, Any] = {}
        if allow_patterns is not None:
            kwargs["allow_patterns"] = allow_patterns
        if revision:
            kwargs["revision"] = revision
        root = snapshot_download(self.repo_id, **kwargs)
        return os.path.join(root, *subfolder.split("/")) if subfolder else root

    def exist_test(self, local_file_markers: list[str] | tuple[str, ...] | None = None) -> tuple[str, str]:
        """
        Whether the target exists and looks like a model directory (has one of
        `local_file_markers`, default `DEFAULT_LOCAL_FILE_MARKERS`).

        Returns mutually exclusive (source, error): source is `local_path`,
        `hf_cache` or `hf_remote`.
        """
        markers: list[str] = list(local_file_markers if local_file_markers is not None else DEFAULT_LOCAL_FILE_MARKERS)
        if self.is_local:
            return _local_exist_test(self.value, markers)
        return _hf_exist_test(self.repo_id, self.subfolder, markers)

    def is_same(self, other: ModelTarget | None) -> bool:
        """Equality, except that two local targets naming the same directory are the same."""
        if other is None or other.kind is not self.kind:
            return False
        if self.value == other.value:
            return True
        if self.is_hf:
            return False
        try:
            return Path(self.value).exists() and Path(self.value).samefile(Path(other.value))
        except OSError:
            return False


def _is_generic_dir_name(name: str) -> bool:
    return name.lower() in ("lora", "adapter", "adapters")


def _qualified_dir_name(path_value: str) -> str:
    path = os.path.normpath(path_value)
    name = os.path.basename(path)
    if name.startswith("checkpoint-") or _is_generic_dir_name(name):
        parent_path = os.path.dirname(path)
        parent = os.path.basename(parent_path)
        while _is_generic_dir_name(parent):
            ancestor_path = os.path.dirname(parent_path)
            ancestor = os.path.basename(ancestor_path)
            if not ancestor or ancestor_path == parent_path:
                break
            parent_path, parent = ancestor_path, ancestor
        if parent:
            return f"{parent}-{name}"
    return name


def _local_exist_test(path: str, markers: list[str]) -> tuple[str, str]:
    if not os.path.exists(path):
        return "", f"No such directory: {path}"
    if not os.path.isdir(path):
        return "", f"Not a directory: {path}"
    if any(os.path.isfile(os.path.join(path, marker)) for marker in markers):
        return "local_path", ""
    return "", f"Directory has none of the expected files ({', '.join(markers)}): {path}"


def _hf_exist_test(repo_id: str, subfolder: str, markers: list[str]) -> tuple[str, str]:
    """The markers must be in the subfolder (or repo root), not merely somewhere in the repo."""
    from huggingface_hub import model_info, try_to_load_from_cache
    from huggingface_hub.errors import RepositoryNotFoundError

    prefix = f"{subfolder}/" if subfolder else ""
    paths = [prefix + marker for marker in markers]
    for path in paths:
        if isinstance(try_to_load_from_cache(repo_id, path), str):
            return "hf_cache", ""
    try:
        info = model_info(repo_id)
    except RepositoryNotFoundError:
        return "", f"Hugging Face repo not found or gated: {repo_id}"
    except Exception as e:
        return "", f"Network or other error: {e}"
    if not subfolder:
        return "hf_remote", ""
    files = {sibling.rfilename for sibling in info.siblings or []}
    if any(path in files for path in paths):
        return "hf_remote", ""
    return "", f"Hugging Face repo {repo_id} has no \"{subfolder}\" directory with typical model files"
