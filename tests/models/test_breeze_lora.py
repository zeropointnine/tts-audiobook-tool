"""Breeze TTS 2 LoRA merge (`breeze_lora`, Instavar format): directory recognition, base
verification, arithmetic, and all-or-nothing rejection.

Runs in venv-breeze (needs torch and safetensors). A real-weights check is
`testx/breeze_lora_poc.py`.
"""
import hashlib
import json
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models import breeze_lora
from tts_audiobook_tool.tts_models.breeze_lora import LoraLoadError, load_lora

RANK = 2
ALPHA = 4.0
Q = "backbone_model.layers.0.self_attn.q_proj"


class TinyBreeze(torch.nn.Module):
    """Same module paths as some of the Breeze layers Instavar's LoRA targets, plus a tied pair."""

    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 6, bias=False)
        layer.self_attn.k_proj = torch.nn.Linear(4, 6, bias=False)
        self.backbone_model = torch.nn.Module()
        self.backbone_model.layers = torch.nn.ModuleList([layer])
        self.backbone_model.norm = torch.nn.LayerNorm(4)
        self.lm_head = torch.nn.Linear(4, 5, bias=False)
        self.tied_a = torch.nn.Linear(4, 4, bias=False)
        self.tied_b = torch.nn.Linear(4, 4, bias=False)
        self.tied_b.weight = self.tied_a.weight


def make_model() -> TinyBreeze:
    torch.manual_seed(0)
    return TinyBreeze().to(torch.bfloat16)


def adapter_tensors() -> dict:
    g = torch.Generator().manual_seed(1)
    return {
        f"{Q}.lora_A": torch.randn(RANK, 4, generator=g).bfloat16(),
        f"{Q}.lora_B": torch.randn(6, RANK, generator=g).bfloat16(),
        "lm_head.lora_A": torch.randn(RANK, 4, generator=g).bfloat16(),
        "lm_head.lora_B": torch.randn(5, RANK, generator=g).bfloat16(),
    }


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def make_base_dir(tmp_path) -> str:
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"model": "breeze"}')
    return str(base)


def write_manifest_dir(directory, base_dir, tensors=None, **overrides) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    save_file(adapter_tensors() if tensors is None else tensors, str(directory / "adapter.safetensors"))
    manifest = {
        "schema_version": 1,
        "artifact_type": "breeze_lora_adapter",
        "base_model": {"id": "BreezeBlue/Breeze-TTS-2", "revision": "anything",
                       "files": {"config.json": sha256(os.path.join(base_dir, "config.json"))}},
        "adapter": {"file": "adapter.safetensors", "sha256": sha256(directory / "adapter.safetensors")},
        "lora": {"variant": "backbone_depth_projection", "rank": RANK, "alpha": ALPHA, "seed": 42},
    }
    for key, value in overrides.items():
        section, _, field = key.partition("__")
        if field:
            manifest[section][field] = value
        else:
            manifest[section] = value
    (directory / "adapter_config.json").write_text(json.dumps(manifest))
    return str(directory)


def write_checkpoint_dir(run_root, tensors=None, run_config=None, status="five_role_checkpoint_complete") -> str:
    checkpoint = run_root / "checkpoint-step-000500"
    checkpoint.mkdir(parents=True, exist_ok=True)
    save_file(adapter_tensors() if tensors is None else tensors, str(checkpoint / "adapter.safetensors"))
    receipt = {"schema_version": 1, "status": status,
               "roles": {"adapter.safetensors": sha256(checkpoint / "adapter.safetensors")}}
    (checkpoint / "checkpoint-receipt.json").write_text(json.dumps(receipt))
    if run_config is not False:
        (run_root / "run-config.json").write_text(json.dumps(run_config or {"rank": RANK, "alpha": ALPHA}))
    return str(checkpoint)


def state_of(model) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("form", ["manifest", "checkpoint"])
@pytest.mark.parametrize("strength", [1.0, 0.5])
def test_merge_adds_scaled_delta_rounded_once(tmp_path, form, strength):
    # W' = bf16(fp32(W) + strength * alpha / r * B @ A) for both directory forms;
    # untouched weights stay bit-identical
    base_dir = make_base_dir(tmp_path)
    target = (write_manifest_dir(tmp_path / "export", base_dir) if form == "manifest"
              else write_checkpoint_dir(tmp_path / "run"))
    model = make_model()
    before = state_of(model)
    tensors = adapter_tensors()

    plan, source = load_lora(model, base_dir, ModelTarget.local(target), strength)

    assert source.kind == form
    scale = strength * ALPHA / RANK
    assert plan.scale == pytest.approx(scale)
    after = state_of(model)
    for module in (Q, "lm_head"):
        a, b = tensors[f"{module}.lora_A"], tensors[f"{module}.lora_B"]
        expected = (before[f"{module}.weight"].float() + (b.float() @ a.float()) * scale).bfloat16()
        assert torch.equal(after[f"{module}.weight"], expected)
    for untouched in ("backbone_model.layers.0.self_attn.k_proj.weight", "tied_a.weight"):
        assert torch.equal(after[untouched], before[untouched])


def _without(key):
    return lambda tensors: {k: v for k, v in tensors.items() if k != key}


def _with(key, value):
    return lambda tensors: {**tensors, key: value}


@pytest.mark.parametrize("edit,message", [
    (_without(f"{Q}.lora_B"), "Incomplete LoRA pair"),
    (_with("lm_head.weight", torch.ones(5, 4)), "Unsupported adapter tensor"),
    (_with(f"{Q}.lora_A", torch.ones(3, 4)), "Shape mismatch"),
    (_with("missing_proj.lora_A", torch.ones(RANK, 4)), "Incomplete LoRA pair|doesn't have"),
    (lambda t: {**t, "missing_proj.lora_A": torch.ones(RANK, 4), "missing_proj.lora_B": torch.ones(4, RANK)},
     "doesn't have"),
    (lambda t: {**t, "backbone_model.norm.lora_A": torch.ones(RANK, 4),
                "backbone_model.norm.lora_B": torch.ones(4, RANK)}, "not a linear layer"),
    (lambda t: {**t, "tied_a.lora_A": torch.ones(RANK, 4), "tied_a.lora_B": torch.ones(4, RANK)},
     "shares its weight"),
    (lambda t: {}, "no tensors"),
])
def test_rejected_tensors_leave_model_unchanged(tmp_path, edit, message):
    # Validation happens before any weight is written, so an adapter is
    # rejected as a whole even when its other tensors would merge fine
    base_dir = make_base_dir(tmp_path)
    target = write_manifest_dir(tmp_path / "export", base_dir, edit(adapter_tensors()))
    model = make_model()
    before = state_of(model)
    with pytest.raises(LoraLoadError, match=message):
        load_lora(model, base_dir, ModelTarget.local(target))
    after = state_of(model)
    assert all(torch.equal(after[name], value) for name, value in before.items())


@pytest.mark.parametrize("overrides,message", [
    ({"artifact_type": "something_else"}, "not a Breeze LoRA manifest"),
    ({"schema_version": 2}, "schema_version"),
    ({"base_model__id": "Other/Model"}, "not BreezeBlue/Breeze-TTS-2"),
    ({"base_model__files": {}}, "pins no base model files"),
    ({"base_model__files": {"../config.json": "0" * 64}}, "Unsafe"),
    ({"adapter__sha256": "0" * 64}, "doesn't match the manifest checksum"),
    ({"adapter__file": "../adapter.safetensors"}, "Invalid adapter file name"),
    ({"lora__rank": 0}, "Invalid LoRA rank"),
    ({"lora__alpha": True}, "Invalid LoRA alpha"),
    ({"lora__rank": 3}, "Shape mismatch"),
])
def test_invalid_manifest_is_rejected(tmp_path, overrides, message):
    base_dir = make_base_dir(tmp_path)
    target = write_manifest_dir(tmp_path / "export", base_dir, **overrides)
    with pytest.raises(LoraLoadError, match=message):
        load_lora(make_model(), base_dir, ModelTarget.local(target))


def test_peft_adapter_is_named_as_unsupported(tmp_path):
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "LORA", "r": 8}))
    with pytest.raises(LoraLoadError, match="PEFT"):
        breeze_lora.resolve_adapter(ModelTarget.local(str(tmp_path)))


@pytest.mark.parametrize("kwargs,message", [
    ({"status": "partial"}, "incomplete"),
    ({"run_config": False}, "run-config.json"),
    ({"run_config": {"rank": RANK}}, "Invalid LoRA alpha"),
])
def test_invalid_checkpoint_is_rejected(tmp_path, kwargs, message):
    target = write_checkpoint_dir(tmp_path / "run", **kwargs)
    with pytest.raises(LoraLoadError, match=message):
        breeze_lora.resolve_adapter(ModelTarget.local(target))


def test_modified_checkpoint_weights_are_rejected(tmp_path):
    # The receipt's checksum catches a partly written or edited adapter file
    target = write_checkpoint_dir(tmp_path / "run")
    save_file({f"{Q}.lora_A": torch.zeros(RANK, 4)}, os.path.join(target, "adapter.safetensors"))
    with pytest.raises(LoraLoadError, match="doesn't match the checkpoint receipt"):
        breeze_lora.resolve_adapter(ModelTarget.local(target))


def test_unrecognized_directory_names_both_forms(tmp_path):
    (tmp_path / "latest.json").write_text("{}")
    with pytest.raises(LoraLoadError, match="adapter_config.json.*checkpoint-receipt.json"):
        breeze_lora.resolve_adapter(ModelTarget.local(str(tmp_path)))


def test_base_mismatch_and_missing_file_are_rejected(tmp_path):
    # Fails closed when a pinned base file differs or is absent; the revision string isn't compared
    base_dir = make_base_dir(tmp_path)
    target = write_manifest_dir(tmp_path / "export", base_dir, base_model__revision="different-revision")
    load_lora(make_model(), base_dir, ModelTarget.local(target))  # revision alone doesn't matter

    (tmp_path / "base" / "config.json").write_text('{"model": "changed"}')
    with pytest.raises(LoraLoadError, match="different Breeze TTS 2 base"):
        load_lora(make_model(), base_dir, ModelTarget.local(target))

    os.remove(tmp_path / "base" / "config.json")
    with pytest.raises(LoraLoadError, match="missing"):
        load_lora(make_model(), base_dir, ModelTarget.local(target))


def test_hf_cache_blob_name_stands_in_for_the_hash(tmp_path):
    # A snapshot file linking to blobs/<sha256> is identified by its blob name (not re-hashed)
    digest = "ab" * 32
    blobs = tmp_path / "hub" / "blobs"
    blobs.mkdir(parents=True)
    (blobs / digest).write_bytes(b"not really the shard")
    snapshot = tmp_path / "hub" / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    os.symlink(blobs / digest, snapshot / "model-00001-of-00002.safetensors")
    source = breeze_lora.BreezeAdapterSource(
        kind="manifest", weights_path="", rank=RANK, alpha=ALPHA,
        base_files={"model-00001-of-00002.safetensors": digest})
    breeze_lora.verify_base(str(snapshot), source)
    with pytest.raises(LoraLoadError, match="differs"):
        breeze_lora.verify_base(str(snapshot), breeze_lora.BreezeAdapterSource(
            kind="manifest", weights_path="", rank=RANK, alpha=ALPHA,
            base_files={"model-00001-of-00002.safetensors": "cd" * 32}))


@pytest.mark.parametrize("form", ["manifest", "checkpoint"])
def test_prepare_lora_validates_without_merging(tmp_path, form):
    # The switcher's planner validates without changing weights and gives unchanged
    # tensor contents and merge metadata a stable identity.
    base_dir = make_base_dir(tmp_path)
    target = (write_manifest_dir(tmp_path / "export", base_dir) if form == "manifest"
              else write_checkpoint_dir(tmp_path / "run"))
    model = make_model()
    before = state_of(model)
    prepared = breeze_lora.prepare_lora(model, base_dir, ModelTarget.local(target), 0.5)
    assert all(torch.equal(state_of(model)[name], value) for name, value in before.items())
    assert prepared.identity == breeze_lora.prepare_lora(model, base_dir, ModelTarget.local(target), 0.5).identity
    assert prepared.plan.scale == pytest.approx(0.5 * ALPHA / RANK)
    assert prepared.plan.target_names == {f"{Q}.weight", "lm_head.weight"}


@pytest.mark.parametrize("form", ["manifest", "checkpoint"])
def test_alpha_only_change_is_remerged_after_a_run(tmp_path, form):
    # Alpha lives outside the tensor file; changing only that metadata must still
    # restore and remerge the adapter rather than reusing the previous scale.
    from pathlib import Path
    from tts_audiobook_tool.app_types.voice import VoiceLora
    from tts_audiobook_tool.tts_models.lora_merge import apply_merge
    from tts_audiobook_tool.tts_models.lora_switch import BaseWeightCache, LoraSwitcher, SafetensorsBaseWeights

    base_dir = make_base_dir(tmp_path)
    target = (write_manifest_dir(tmp_path / "export", base_dir) if form == "manifest"
              else write_checkpoint_dir(tmp_path / "run"))
    model = make_model()
    save_file(state_of(model), os.path.join(base_dir, "model.safetensors"))
    switcher = LoraSwitcher(
        model, BaseWeightCache(SafetensorsBaseWeights(base_dir)),
        lambda lora: breeze_lora.prepare_lora(model, base_dir, lora.target, lora.strength),
        log=lambda _: None,
    )
    lora = VoiceLora(ModelTarget.local(target))
    old = breeze_lora.prepare_lora(model, base_dir, ModelTarget.local(target))
    weights_sha = sha256(os.path.join(target, "adapter.safetensors"))
    assert switcher.activate(lora) == ""
    switcher.end_run()

    metadata_path = (Path(target) / "adapter_config.json" if form == "manifest"
                     else Path(target).parent / "run-config.json")
    metadata = json.loads(metadata_path.read_text())
    if form == "manifest":
        metadata["lora"]["alpha"] *= 2
    else:
        metadata["alpha"] *= 2
    metadata_path.write_text(json.dumps(metadata))

    updated = breeze_lora.prepare_lora(model, base_dir, ModelTarget.local(target))
    assert sha256(os.path.join(target, "adapter.safetensors")) == weights_sha
    assert updated.identity != old.identity
    assert switcher.activate(lora) == ""
    expected = make_model()
    apply_merge(expected, updated.plan, delta_device="target")
    assert all(torch.equal(state_of(model)[name], value) for name, value in state_of(expected).items())


def test_breeze_model_reports_each_voices_rejected_lora(tmp_path):
    # Pre-flight names the voice and the reason; voices without a LoRA aren't checked
    pytest.importorskip("breeze_tts")
    from pathlib import Path
    from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
    from tts_audiobook_tool.project import Project
    from tts_audiobook_tool.tts_models.breeze_model import BreezeModel

    base_dir = make_base_dir(tmp_path)
    good = write_manifest_dir(tmp_path / "export", base_dir)
    model = BreezeModel.__new__(BreezeModel)
    model._model = make_model()  # type: ignore[assignment]
    # The switcher only accepts adapters whose targets it can restore from the base checkpoint
    save_file(state_of(model._model), os.path.join(base_dir, "model.safetensors"))
    model._loras = BreezeModel._make_lora_switcher(model._model, Path(base_dir))
    model_id = "breeze_tts_2_local"
    project = Project(tts_model_type=model_id, voice_references=[
        Voice(), Voice(loras={model_id: VoiceLora(ModelTarget.local(good))}), Voice(loras={model_id: VoiceLora(ModelTarget.local(str(tmp_path / "base")))}),
    ])
    problems = model.prepare_voice_loras(project, [0, 1, 2])
    assert len(problems) == 1 and problems[0].startswith("Voice 3 LoRA") and "Not a Breeze LoRA" in problems[0]
    assert model.validate_voice_lora(VoiceLora(ModelTarget.local(good))) == ""


def test_breeze_menu_validation_rechecks_a_repaired_or_changed_adapter(tmp_path):
    # Repeated explicit menu checks must see on-disk repairs and later corruption,
    # without applying any weights during validation.
    pytest.importorskip("breeze_tts")
    from pathlib import Path
    from tts_audiobook_tool.app_types.voice import VoiceLora
    from tts_audiobook_tool.tts_models.breeze_model import BreezeModel

    base_dir = make_base_dir(tmp_path)
    target = write_manifest_dir(tmp_path / "export", base_dir, adapter__sha256="0" * 64)
    model = BreezeModel.__new__(BreezeModel)
    model._model = make_model()  # type: ignore[assignment]
    # The switcher only accepts adapters whose targets it can restore from the base checkpoint
    save_file(state_of(model._model), os.path.join(base_dir, "model.safetensors"))
    model._loras = BreezeModel._make_lora_switcher(model._model, Path(base_dir))
    before = state_of(model._model)
    lora = VoiceLora(ModelTarget.local(target))
    assert "checksum" in model.validate_voice_lora(lora)

    manifest_path = Path(target) / "adapter_config.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["adapter"]["sha256"] = sha256(os.path.join(target, "adapter.safetensors"))
    manifest_path.write_text(json.dumps(manifest))
    assert model.validate_voice_lora(lora) == ""

    manifest["adapter"]["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))
    # The successful probe must not seed a stale plan for subsequent generation either.
    assert "checksum" in model._loras.prepare(lora)
    assert "checksum" in model.validate_voice_lora(lora)
    assert all(torch.equal(state_of(model._model)[name], value) for name, value in before.items())
