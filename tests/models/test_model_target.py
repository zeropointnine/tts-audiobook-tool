"""`ModelTarget`: typed hf / local targets, legacy kind inference, resolution and existence checks."""
import os
import pickle
from types import SimpleNamespace

import huggingface_hub
import pytest

from tts_audiobook_tool.app_types.model_target import (
    ModelTarget,
    TargetKind,
    infer_legacy_kind,
    looks_like_local_path,
    split_hf_subfolder,
)


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
def test_split_hf_subfolder(value, expected):
    assert split_hf_subfolder(value) == expected


def test_construction_validates_hf_syntax_but_not_existence():
    # An hf target must be a repo id or subfolder form; a local one is any non-empty string
    assert ModelTarget.hf("owner/repo").repo_id == "owner/repo"
    assert ModelTarget.local("/no/such/dir").is_local
    for bad in ("/abs/path", "C:\\models", "bad name/repo"):
        with pytest.raises(ValueError):
            ModelTarget.hf(bad)
    for blank in ("", "  "):
        with pytest.raises(ValueError):
            ModelTarget.local(blank)
    with pytest.raises(ValueError):
        ModelTarget("bogus", "owner/repo")  # type: ignore[arg-type]


def test_kind_accepts_its_string_id():
    target = ModelTarget("hf", "owner/repo")  # type: ignore[arg-type]
    assert target.kind is TargetKind.HF


def test_repo_id_and_subfolder():
    plain = ModelTarget.hf("owner/repo")
    nested = ModelTarget.hf("owner/repo/a/b")
    local = ModelTarget.local("/x/y")
    assert (plain.repo_id, plain.subfolder) == ("owner/repo", "")
    assert (nested.repo_id, nested.subfolder) == ("owner/repo", "a/b")
    assert (local.repo_id, local.subfolder) == ("", "")


def test_infer_legacy_kind_follows_the_old_runtime_rule(tmp_path, monkeypatch):
    # An existing directory wins even when it is repo-shaped (relative to the cwd)
    (tmp_path / "owner" / "repo").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    assert infer_legacy_kind("owner/repo") is TargetKind.LOCAL
    # Otherwise hf syntax means hf, including the subfolder form
    assert infer_legacy_kind("other/repo") is TargetKind.HF
    assert infer_legacy_kind("other/repo/sub/dir") is TargetKind.HF
    # Everything else is a (possibly missing) local path
    for value in ("/missing/abs/dir", "C:\\models\\lora", "C:/models/lora", "~/loras/x"):
        assert infer_legacy_kind(value) is TargetKind.LOCAL


@pytest.mark.parametrize("value,kind,expected_kind,inferred", [
    ("owner/repo", "hf", TargetKind.HF, False),
    ("/some/dir", "local", TargetKind.LOCAL, False),
    # A repo-shaped string can be stored as local
    ("owner/repo", "local", TargetKind.LOCAL, False),
    ("owner/repo", None, TargetKind.HF, True),  # legacy: no kind
    ("/some/dir", "bogus", TargetKind.LOCAL, True),  # unknown kind
    ("C:\\models\\lora", "hf", TargetKind.LOCAL, True),  # kind that cannot fit the value
])
def test_from_stored(value, kind, expected_kind, inferred):
    target = ModelTarget.from_stored(value, kind)
    assert target is not None
    assert (target.kind, target.value, target.inferred) == (expected_kind, value, inferred)


@pytest.mark.parametrize("value", [None, "", "   ", 42])
def test_from_stored_without_a_value_is_none(value):
    assert ModelTarget.from_stored(value, "hf") is None


def test_inferred_flag_is_ignored_by_equality_and_hashing_and_survives_pickling():
    inferred = ModelTarget.from_stored("owner/repo", None)
    typed = ModelTarget.hf("owner/repo")
    assert inferred == typed and hash(inferred) == hash(typed)
    assert pickle.loads(pickle.dumps(typed)) == typed
    assert typed.to_stored() == ("owner/repo", "hf")


def test_is_same(tmp_path):
    a = tmp_path / "a"
    a.mkdir()
    assert ModelTarget.hf("owner/repo").is_same(ModelTarget.hf("owner/repo"))
    assert not ModelTarget.hf("owner/repo/x").is_same(ModelTarget.hf("owner/repo/y"))
    assert not ModelTarget.hf("owner/repo").is_same(None)
    # Same string, different kinds: different targets
    assert not ModelTarget.hf("owner/repo").is_same(ModelTarget.local("owner/repo"))
    # Two spellings of one local directory
    assert ModelTarget.local(str(a)).is_same(ModelTarget.local(str(tmp_path / "." / "a")))


def test_resolve_local_dir(tmp_path, monkeypatch):
    calls = []

    def snapshot_download(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)
    files = ["adapter_config.json", "adapter_model.safetensors"]

    # Named files are fetched from the subfolder
    assert ModelTarget.hf("owner/repo/a/b").resolve_local_dir(files) == os.path.join(str(tmp_path), "a", "b")
    # Without names: the whole subfolder, or the whole repo
    ModelTarget.hf("owner/repo/a").resolve_local_dir(revision="abc")
    ModelTarget.hf("owner/repo").resolve_local_dir()
    # A local target never downloads, even when its directory is missing
    assert ModelTarget.local("/missing/dir").resolve_local_dir(files) == "/missing/dir"
    assert calls == [
        ("owner/repo", {"allow_patterns": ["a/b/adapter_config.json", "a/b/adapter_model.safetensors"]}),
        ("owner/repo", {"allow_patterns": ["a/*"], "revision": "abc"}),
        ("owner/repo", {}),
    ]


def test_local_exist_test(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (tmp_path / "file.txt").write_text("x")
    assert ModelTarget.local(str(model_dir)).exist_test()[1].startswith("Directory has none")
    (model_dir / "config.json").write_text("{}")
    assert ModelTarget.local(str(model_dir)).exist_test() == ("local_path", "")
    assert ModelTarget.local(str(model_dir)).exist_test(["adapter_config.json"])[0] == ""
    assert "Not a directory" in ModelTarget.local(str(tmp_path / "file.txt")).exist_test()[1]
    assert "No such directory" in ModelTarget.local(str(tmp_path / "missing")).exist_test()[1]


@pytest.mark.parametrize("value,cached,siblings,expected_source,expected_error", [
    ("owner/repo/lora", True, [], "hf_cache", ""),
    ("owner/repo/lora", False, ["lora/adapter_config.json"], "hf_remote", ""),
    # The marker exists elsewhere in the repo, but not in the subfolder
    ("owner/repo/lora", False, ["adapter_config.json", "lorax/adapter_config.json"], "", 'no "lora" directory'),
    # A reachable plain repo is accepted without listing its files
    ("owner/repo", False, [], "hf_remote", ""),
])
def test_hf_exist_test(monkeypatch, value, cached, siblings, expected_source, expected_error):
    looked_up = []

    def try_to_load_from_cache(repo_id, filename):
        looked_up.append((repo_id, filename))
        return "/cache/file" if cached else None

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", try_to_load_from_cache)
    monkeypatch.setattr(huggingface_hub, "model_info", lambda repo_id: SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name) for name in siblings]))

    source, error = ModelTarget.hf(value).exist_test(["adapter_config.json"])

    prefix = "lora/" if value.count("/") > 1 else ""
    assert looked_up[0] == ("owner/repo", f"{prefix}adapter_config.json")
    assert source == expected_source
    assert (expected_error in error) if expected_error else error == ""


def test_hf_exist_test_reports_missing_repo(monkeypatch):
    from huggingface_hub.errors import RepositoryNotFoundError

    def model_info(repo_id):
        # Built without __init__, which wants a real HTTP response
        raise RepositoryNotFoundError.__new__(RepositoryNotFoundError)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", lambda *_: None)
    monkeypatch.setattr(huggingface_hub, "model_info", model_info)
    _, error = ModelTarget.hf("owner/repo/lora").exist_test()
    assert "not found" in error and "owner/repo" in error


@pytest.mark.parametrize("target,expected", [
    (ModelTarget.hf("My-Weird-Prompts/Herman-LoRA-V3"), "Herman-LoRA-V3"),
    (ModelTarget.hf("wjcwqc/PaigeVoice/lora"), "PaigeVoice"),
    (ModelTarget.local("/exp/herman/checkpoint-900"), "herman-checkpoint-900"),
    (ModelTarget.local("/exp/herman_v3"), "herman_v3"),
    # A generic parent (as Instavar's run layout) is skipped for the nearest meaningful ancestor
    (ModelTarget.local("/runs/fw-leda/lora/checkpoint-step-000500"), "fw-leda-checkpoint-step-000500"),
    (ModelTarget.local("/runs/fw-leda/adapters/checkpoint-step-000500"), "fw-leda-checkpoint-step-000500"),
    (ModelTarget.local("/lora/checkpoint-step-000500"), "lora-checkpoint-step-000500"),
    # The kind decides, not the shape: a repo-shaped local path uses directory naming
    (ModelTarget.local("owner/repo"), "repo"),
])
def test_display_name(target, expected):
    assert target.display_name() == expected


@pytest.mark.parametrize("value,expected", [
    ("/abs/dir", True),
    ("~/loras", True),
    ("./rel", True),
    ("C:/models", True),
    ("C:\\models", True),
    ("owner/repo", False),
    ("owner/repo/sub", False),
])
def test_looks_like_local_path(value, expected):
    assert looks_like_local_path(value) is expected


@pytest.mark.parametrize("text,expected", [
    ("owner/repo", ("owner/repo", "")),
    ("  'owner/repo/sub'  ", ("owner/repo/sub", "")),
    ("hf://owner/repo/sub/dir", ("owner/repo/sub/dir", "")),
    ("https://huggingface.co/owner/repo", ("owner/repo", "")),
    ("https://huggingface.co/owner/repo/tree/main/a/b", ("owner/repo/a/b", "")),
    ("", ("", "")),
])
def test_parse_hf_input_accepts_ids_hf_paths_and_urls(text, expected):
    from tts_audiobook_tool.app_types.model_target import parse_hf_input
    assert parse_hf_input(text) == expected


@pytest.mark.parametrize("text,error", [
    ("https://huggingface.co/owner/repo/tree/v2/a", "main revision"),
    ("https://huggingface.co/owner/repo/blob/main/config.json", "not a file"),
    ("/home/lee/model", "Not a valid"),
    ("C:\\models\\x", "Not a valid"),
])
def test_parse_hf_input_rejects_other_revisions_files_and_paths(text, error):
    from tts_audiobook_tool.app_types.model_target import parse_hf_input
    value, message = parse_hf_input(text)
    assert value == "" and error in message
