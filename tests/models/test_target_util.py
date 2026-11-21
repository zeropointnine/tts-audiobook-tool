"""`owner/repo/sub/dir` targets: parsing, existence check, download and display name."""
import os
from types import SimpleNamespace

import huggingface_hub
import pytest

from tts_audiobook_tool import target_util


@pytest.mark.parametrize("value,expected", [
    ("owner/repo/lora", ("owner/repo", "lora")),
    ("owner/repo/adapters/checkpoint-900", ("owner/repo", "adapters/checkpoint-900")),
    ("owner/repo", None),  # a plain repo id
    ("/abs/path/to/dir", None),  # leading slash: empty first segment
    ("owner/repo/../escape", None),
    ("owner/repo//double", None),
    ("C:\\models\\lora", None),
    ("bad name/repo/sub", None),  # invalid repo id
])
def test_split_hf_subfolder_target(value, expected):
    assert target_util.split_hf_subfolder_target(value) == expected


def test_resolve_downloads_only_the_named_files_from_the_subfolder(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(repo_id, allow_patterns):
        calls.append((repo_id, allow_patterns))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    files = ["adapter_config.json", "adapter_model.safetensors"]

    assert target_util.resolve_to_local_dir("owner/repo/a/b", files) == os.path.join(str(tmp_path), "a", "b")
    assert target_util.resolve_to_local_dir("owner/repo", files) == str(tmp_path)
    # An existing local directory is used as is
    assert target_util.resolve_to_local_dir(str(tmp_path), files) == str(tmp_path)
    assert calls == [
        ("owner/repo", ["a/b/adapter_config.json", "a/b/adapter_model.safetensors"]),
        ("owner/repo", files),
    ]


@pytest.mark.parametrize("cached,siblings,expected_source,expected_error", [
    (True, [], "hf_cache", ""),
    (False, ["lora/adapter_config.json"], "hf_remote", ""),
    # The marker exists elsewhere in the repo, but not in the subfolder
    (False, ["adapter_config.json", "lorax/adapter_config.json"], "", 'no "lora" directory'),
])
def test_subfolder_exist_test_requires_markers_in_the_subfolder(
        monkeypatch, cached, siblings, expected_source, expected_error):
    looked_up = []

    def try_to_load_from_cache(repo_id, filename):
        looked_up.append((repo_id, filename))
        return "/cache/file" if cached else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "model_info", lambda repo_id: SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name) for name in siblings]))

    source, error = target_util.exist_test("owner/repo/lora", local_file_markers=["adapter_config.json"])

    assert looked_up[0] == ("owner/repo", "lora/adapter_config.json")
    assert source == expected_source
    assert (expected_error in error) if expected_error else error == ""


def test_subfolder_exist_test_reports_missing_repo(monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError

    def model_info(repo_id):
        # Built without __init__, which wants a real HTTP response
        raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_: None)
    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    _, error = target_util.exist_test("owner/repo/lora")
    assert "Repo not found" in error


def test_is_same_target_compares_subfolder_targets_as_strings():
    assert target_util.is_same_target("owner/repo/lora", "owner/repo/lora")
    assert not target_util.is_same_target("owner/repo/lora", "owner/repo/other")


@pytest.mark.parametrize("target,expected", [
    ("My-Weird-Prompts/Herman-LoRA-V3", "Herman-LoRA-V3"),
    ("wjcwqc/PaigeVoice/lora", "PaigeVoice"),
    ("/exp/herman/checkpoint-900", "herman-checkpoint-900"),
    ("/exp/herman_v3", "herman_v3"),
    # A generic parent (as Instavar's run layout) is skipped for the nearest meaningful ancestor
    ("/runs/fw-leda/lora/checkpoint-step-000500", "fw-leda-checkpoint-step-000500"),
    ("/runs/fw-leda/adapters/checkpoint-step-000500", "fw-leda-checkpoint-step-000500"),
    ("/lora/checkpoint-step-000500", "lora-checkpoint-step-000500"),
])
def test_make_target_display_name(target, expected):
    assert target_util.make_target_display_name(target) == expected
