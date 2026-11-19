import os
from pathlib import Path


"""
"Target" is app nomenclature for:
1) a huggingface repo id or
2) a local directory path to a model presumably downloaded from a huggingface repo

Some targets (currently OmniVoice LoRAs) also accept a directory inside a repo,
written `owner/repo/sub/dir` (the `HfFileSystem` path form, minus `hf://`). A
plain repo id never has more than two segments, so the form cannot collide with
one; an existing local directory always takes precedence.
"""


def is_hf_repo_id_syntax(s: str) -> bool:
    from huggingface_hub.utils._validators import validate_repo_id
    from huggingface_hub.errors import HFValidationError

    try:
        validate_repo_id(s)
    except HFValidationError:
        return False
    return True


def split_hf_subfolder_target(s: str) -> tuple[str, str] | None:
    """
    Splits `owner/repo/sub/dir` into ("owner/repo", "sub/dir").
    Returns None for anything else, including a plain repo id. Syntax only;
    does not check the local filesystem.
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


def resolve_to_local_dir(target: str, file_names: list[str]) -> str:
    """
    Returns a local directory holding `file_names` for a target: the directory
    itself, or the (possibly subfolder-qualified) hf repo, downloading only
    those files into the hf cache. Raises on download failure.
    """
    if os.path.isdir(target):
        return target
    from huggingface_hub import snapshot_download
    split = split_hf_subfolder_target(target)
    repo_id, subfolder = split if split else (target, "")
    prefix = f"{subfolder}/" if subfolder else ""
    root = snapshot_download(repo_id, allow_patterns=[prefix + name for name in file_names])
    return os.path.join(root, *subfolder.split("/")) if subfolder else root


def make_target_display_name(target: str) -> str:
    """
    Short name for a target: the repo name for hf targets, else the directory
    name, qualified by its parent when it is a training checkpoint directory
    (eg `checkpoint-900`) or a generic name like `lora`.
    """
    if not os.path.isdir(target):
        split = split_hf_subfolder_target(target)
        if split or is_hf_repo_id_syntax(target):
            repo_id = split[0] if split else target
            return repo_id.split("/")[-1]
    path = os.path.normpath(target)
    name = os.path.basename(path)
    if name.startswith("checkpoint-") or name.lower() in ("lora", "adapter", "adapters"):
        parent = os.path.basename(os.path.dirname(path))
        if parent:
            return f"{parent}-{name}"
    return name


def exist_test(
        target: str,
        local_file_markers=["config.json", "model.safetensors", "adapter_config.json"]
) -> tuple[str, str]:
    """
    Tests if hf target exists

    Returns
        Tuple with mutually exclusive values:
            source type (local_path | hf_cache | hf_remote)
            error string
    """

    from huggingface_hub import model_info, try_to_load_from_cache
    from huggingface_hub.errors import RepositoryNotFoundError

    # If value is an existing local directory, check for existence a 'local file marker'
    if os.path.exists(target):
        if not os.path.isdir(target):
            return "", "Path is not a directory"
        is_hf_dir = any(os.path.isfile(os.path.join(target, marker)) for marker in local_file_markers)
        if is_hf_dir:
            return "local_path", ""
        else:
            return "", "Directory is missing typical hf file markers"

    error = "No such local directory;\n"

    split = split_hf_subfolder_target(target)
    if split:
        return _subfolder_exist_test(*split, local_file_markers=local_file_markers, error=error)

    # Test for correct hf repo id string format
    if not is_hf_repo_id_syntax(target):
        error += "Invalid string format for hf repo id"
        return "", error

    # Check local hf cache
    for file_marker in local_file_markers:
        cached_path = try_to_load_from_cache(target, file_marker)
        if isinstance(cached_path, str):
            return "hf_cache", ""

    # Test for hf repo remote reachability
    try:
        model_info(target)
        return "hf_remote", ""
    except RepositoryNotFoundError:
        error += "Repo not found or gated"
        return "", error
    except Exception as e:
        error += f"Network or other error: {e}"
        return "", error


def _subfolder_exist_test(
        repo_id: str,
        subfolder: str,
        local_file_markers: list[str],
        error: str,
) -> tuple[str, str]:
    """`exist_test` for `owner/repo/sub/dir`: the markers must be in the subfolder,
    not merely somewhere in the repo."""
    from huggingface_hub import model_info, try_to_load_from_cache
    from huggingface_hub.errors import RepositoryNotFoundError

    markers = [f"{subfolder}/{marker}" for marker in local_file_markers]
    for marker in markers:
        if isinstance(try_to_load_from_cache(repo_id, marker), str):
            return "hf_cache", ""
    try:
        info = model_info(repo_id)
    except RepositoryNotFoundError:
        return "", error + f"Repo not found or gated: {repo_id}"
    except Exception as e:
        return "", error + f"Network or other error: {e}"
    files = {sibling.rfilename for sibling in info.siblings or []}
    if any(marker in files for marker in markers):
        return "hf_remote", ""
    return "", error + f"Repo {repo_id} has no \"{subfolder}\" directory with typical hf file markers"


def is_same_target(a: str, b: str) -> bool:
    if is_hf_repo_id_syntax(b) or (split_hf_subfolder_target(b) and not os.path.exists(b)):
        is_same = (a == b)
    else:
        # Treat as dir path
        try:
            is_same = Path(b).exists() and Path(b).samefile(Path(a))
        except Exception:
            is_same = False
    return is_same


