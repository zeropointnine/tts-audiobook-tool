"""OmniVoice LoRA merge (`omnivoice_lora`): arithmetic, and all-or-nothing rejection.

Runs in venv-omni (needs torch and safetensors). Parity with peft's own merge on
real adapters is checked separately by `testx/omnivoice_lora_peft_parity.py`.
"""
import json
import math

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")
from safetensors.torch import save_file

from tts_audiobook_tool.app_types.model_target import ModelTarget
from tts_audiobook_tool.tts_models import omnivoice_lora
from tts_audiobook_tool.tts_models.omnivoice_lora import LoraLoadError, load_lora

RANK = 2


class TinyOmniVoice(torch.nn.Module):
    """Same module paths as the parts of OmniVoice an adapter touches."""

    def __init__(self):
        super().__init__()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(4, 6, bias=False)
        layer.self_attn.k_proj = torch.nn.Linear(4, 6, bias=False)
        self.llm = torch.nn.Module()
        self.llm.layers = torch.nn.ModuleList([layer])
        self.audio_embeddings = torch.nn.Embedding(5, 4)
        self.audio_heads = torch.nn.Linear(4, 5, bias=False)


def make_model() -> TinyOmniVoice:
    torch.manual_seed(0)
    return TinyOmniVoice().half()


def adapter_tensors() -> dict:
    g = torch.Generator().manual_seed(1)
    q = "base_model.model.llm.layers.0.self_attn.q_proj"
    return {
        f"{q}.lora_A.weight": torch.randn(RANK, 4, generator=g),
        f"{q}.lora_B.weight": torch.randn(6, RANK, generator=g),
        "base_model.model.audio_heads.weight": torch.randn(5, 4, generator=g),
    }


def write_adapter(directory, tensors=None, **config_overrides) -> ModelTarget:
    config = {"peft_type": "LORA", "r": RANK, "lora_alpha": 4, "use_rslora": False,
              "use_dora": False, "bias": "none", "lora_bias": False, "rank_pattern": {},
              "alpha_pattern": {}, "target_parameters": None, **config_overrides}
    (directory / "adapter_config.json").write_text(json.dumps(config))
    save_file(adapter_tensors() if tensors is None else tensors, str(directory / "adapter_model.safetensors"))
    return ModelTarget.local(str(directory))


def state_of(model) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


@pytest.mark.parametrize("use_rslora,expected_scale", [(False, 4 / RANK), (True, 4 / math.sqrt(RANK))])
def test_merge_adds_scaled_delta_rounded_once_and_copies_replacements(tmp_path, use_rslora, expected_scale):
    # W' = fp16(fp32(W) + scale * B @ A); replacements are copied (rounded to
    # the model dtype); untouched weights stay bit-identical
    model = make_model()
    before = state_of(model)
    tensors = adapter_tensors()
    plan = load_lora(model, write_adapter(tmp_path, use_rslora=use_rslora))

    assert plan.scale == pytest.approx(expected_scale)
    a = tensors["base_model.model.llm.layers.0.self_attn.q_proj.lora_A.weight"]
    b = tensors["base_model.model.llm.layers.0.self_attn.q_proj.lora_B.weight"]
    name = "llm.layers.0.self_attn.q_proj.weight"
    expected = (before[name].float() + (b @ a) * expected_scale).half()
    after = state_of(model)
    assert torch.equal(after[name], expected)
    assert torch.equal(after["audio_heads.weight"], tensors["base_model.model.audio_heads.weight"].half())
    for untouched in ("llm.layers.0.self_attn.k_proj.weight", "audio_embeddings.weight"):
        assert torch.equal(after[untouched], before[untouched])


def _q(suffix: str) -> str:
    return f"base_model.model.llm.layers.0.self_attn.q_proj.{suffix}"


def _without(key: str):
    return lambda tensors: {k: v for k, v in tensors.items() if k != key}


def _with(key: str, value):
    return lambda tensors: {**tensors, key: value}


@pytest.mark.parametrize("config,edit,message", [
    ({"use_dora": True}, None, "use_dora"),
    ({"rank_pattern": {"q_proj": 4}}, None, "rank_pattern"),
    ({"peft_type": "LOHA"}, None, "LOHA"),
    ({"r": 3}, None, "Shape mismatch"),
    ({}, _without(_q("lora_B.weight")), "Incomplete LoRA pair"),
    ({}, _with("llm.layers.0.self_attn.q_proj.lora_A.weight", torch.ones(RANK, 4)), "Unexpected tensor"),
    ({}, _with("base_model.model.llm.embed_tokens.lora_embedding_A", torch.ones(RANK, 4)),
     "Unsupported adapter tensor"),
    ({}, _with("base_model.model.not_a_module.weight", torch.ones(2, 2)), "doesn't have"),
    ({}, _with("base_model.model.audio_heads.weight", torch.ones(4, 4)), "Shape mismatch"),
    ({}, _with(_q("lora_B.weight"), torch.full((6, RANK), float("nan"))), "NaN or infinite"),
    # A finite fp32 replacement that overflows the fp16 model when copied is caught as well
    ({}, _with("base_model.model.audio_heads.weight", torch.full((5, 4), 1e6)), "NaN or infinite"),
])
def test_rejected_adapter_leaves_model_unchanged(tmp_path, config, edit, message):
    # Validation happens before any weight is written, so an adapter is
    # rejected as a whole even when its other tensors would merge fine
    model = make_model()
    before = state_of(model)
    tensors = edit(adapter_tensors()) if edit else None
    with pytest.raises(LoraLoadError, match=message):
        load_lora(model, write_adapter(tmp_path, tensors, **config))
    after = state_of(model)
    assert all(torch.equal(after[name], value) for name, value in before.items())


def test_missing_adapter_file_is_a_load_error(tmp_path):
    (tmp_path / "adapter_config.json").write_text("{}")
    with pytest.raises(LoraLoadError, match="adapter_model.safetensors"):
        omnivoice_lora.resolve_adapter_dir(ModelTarget.local(str(tmp_path)))


def test_strength_scales_the_lora_delta_and_blends_replacements_from_the_base(tmp_path):
    # Strength scales the adapter's whole change: the low-rank delta, and the
    # full-rank change a replacement implies (W + s * (R - W)), each in fp32 and
    # rounded once. Untouched weights stay bit-identical.
    strength = 0.5
    model = make_model()
    before = state_of(model)
    tensors = adapter_tensors()
    plan = load_lora(model, write_adapter(tmp_path), strength)

    assert plan.scale == pytest.approx(strength * 4 / RANK) and plan.replacement_strength == strength
    a, b = tensors[_q("lora_A.weight")], tensors[_q("lora_B.weight")]
    name = "llm.layers.0.self_attn.q_proj.weight"
    after = state_of(model)
    assert torch.equal(after[name], (before[name].float() + (b @ a) * plan.scale).half())
    base_head = before["audio_heads.weight"].float()
    replacement = tensors["base_model.model.audio_heads.weight"]
    assert torch.equal(after["audio_heads.weight"], (base_head + (replacement - base_head) * strength).half())
    for untouched in ("llm.layers.0.self_attn.k_proj.weight", "audio_embeddings.weight"):
        assert torch.equal(after[untouched], before[untouched])


@pytest.mark.parametrize("strength", [0, float("nan"), True])
def test_invalid_strength_is_rejected(tmp_path, strength):
    with pytest.raises(LoraLoadError, match="strength"):
        omnivoice_lora.prepare_lora(make_model(), write_adapter(tmp_path), strength)


@pytest.mark.parametrize("strength,expected_dtype", [(1.0, torch.float16), (0.5, torch.float32)])
def test_planned_replacements_are_rounded_only_when_copied(tmp_path, strength, expected_dtype):
    # At strength 1 a cached plan holds replacements already rounded to the model's
    # fp16 (half the RAM, same result); a blended one keeps fp32, so it rounds once.
    plan = omnivoice_lora.prepare_lora(make_model(), write_adapter(tmp_path), strength).plan
    assert plan.replacements["audio_heads.weight"].dtype == expected_dtype


def test_strength_is_part_of_the_identity(tmp_path):
    # The same adapter at another strength merges differently, so it must remerge
    model = make_model()
    target = write_adapter(tmp_path)
    assert (omnivoice_lora.prepare_lora(model, target, 0.5).identity
            != omnivoice_lora.prepare_lora(model, target).identity)


def test_prepare_lora_is_nonmutating_and_tracks_tensor_and_scaling_changes(tmp_path):
    # Identity covers the weights and metadata, including rsLoRA, without merging.
    model = make_model()
    before = state_of(model)
    target = write_adapter(tmp_path)
    initial = omnivoice_lora.prepare_lora(model, target)
    assert initial.identity == omnivoice_lora.prepare_lora(model, target).identity
    assert initial.plan.target_names == {"llm.layers.0.self_attn.q_proj.weight", "audio_heads.weight"}
    assert all(torch.equal(state_of(model)[name], value) for name, value in before.items())

    alpha_changed = omnivoice_lora.prepare_lora(model, write_adapter(tmp_path, lora_alpha=8))
    assert alpha_changed.identity != initial.identity
    rs_changed = omnivoice_lora.prepare_lora(model, write_adapter(tmp_path, use_rslora=True))
    assert rs_changed.identity != initial.identity
    tensors = adapter_tensors()
    tensors[_q("lora_B.weight")] *= 2
    rewritten = omnivoice_lora.prepare_lora(model, write_adapter(tmp_path, tensors))
    assert rewritten.identity != initial.identity


def test_switching_restores_replacements_and_casts_base_checkpoint_weights(tmp_path):
    # An adapter may replace entire embeddings/heads as well as LoRA layers. Every
    # switch restores all of them, using the model's fp16 cast of fp32 base files.
    from tts_audiobook_tool.app_types.voice import VoiceLora
    from tts_audiobook_tool.tts_models.lora_merge import apply_merge
    from tts_audiobook_tool.tts_models.lora_switch import BaseWeightCache, LoraSwitcher, SafetensorsBaseWeights

    base_dir, a_dir, b_dir = (tmp_path / name for name in ("base", "a", "b"))
    for directory in (base_dir, a_dir, b_dir):
        directory.mkdir()
    torch.manual_seed(0)
    checkpoint = TinyOmniVoice()
    save_file(state_of(checkpoint), str(base_dir / "model.safetensors"))
    model = make_model()
    base = state_of(model)
    a = VoiceLora(write_adapter(a_dir))
    b_tensors = adapter_tensors()
    b_tensors[_q("lora_B.weight")] *= 2
    b_tensors["base_model.model.audio_embeddings.weight"] = torch.full((5, 4), 2.5)
    b = VoiceLora(write_adapter(b_dir, b_tensors, use_rslora=True), 0.5)
    switcher = LoraSwitcher(
        model, BaseWeightCache(SafetensorsBaseWeights(str(base_dir))),
        lambda lora: omnivoice_lora.prepare_lora(model, lora.target, lora.strength),
        log=lambda _: None,
    )
    pointers = {name: param.data_ptr() for name, param in model.named_parameters()}
    for lora in (a, b, a, None, b, None):
        assert switcher.activate(lora) == ""
        expected = make_model()
        if lora is not None:
            plan = omnivoice_lora.prepare_lora(expected, lora.target, lora.strength).plan
            apply_merge(expected, plan, delta_device="target")
        assert all(torch.equal(state_of(model)[name], value) for name, value in state_of(expected).items())
    assert all(torch.equal(state_of(model)[name], value) for name, value in base.items())
    assert {name: param.data_ptr() for name, param in model.named_parameters()} == pointers


@pytest.mark.parametrize("use_repo,device_name,expected_dtype", [
    (True, "CUDA", torch.float16),
    (False, "CPU", torch.float32),
])
def test_model_loading_and_base_restores_share_one_resolved_snapshot(
        tmp_path, monkeypatch, use_repo, device_name, expected_dtype):
    # Resolve a hub target once before loading; local checkpoints need no download.
    pytest.importorskip("omnivoice")
    import huggingface_hub
    from tts_audiobook_tool.app_types import DeviceType
    from tts_audiobook_tool.tts_models import omnivoice_model

    downloaded, loaded, sources = [], [], []
    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **_: downloaded.append(repo_id) or str(tmp_path))
    model = make_model()
    monkeypatch.setattr(omnivoice_model.OmniVoice, "from_pretrained",
                        lambda path, **kwargs: loaded.append((path, kwargs)) or model)
    sentinel = object()
    monkeypatch.setattr(omnivoice_model.OmniVoiceModel, "_make_lora_switcher",
                        staticmethod(lambda value, path: sources.append((value, str(path))) or sentinel))
    target = ModelTarget.hf("owner/base") if use_repo else ModelTarget.local(str(tmp_path))
    wrapper = omnivoice_model.OmniVoiceModel(target, DeviceType[device_name])
    assert downloaded == (["owner/base"] if use_repo else [])
    assert loaded == [(str(tmp_path), {"device_map": "cuda:0" if use_repo else "cpu", "dtype": expected_dtype})]
    assert sources == [(model, str(tmp_path))]
    assert wrapper._loras is sentinel


def make_wrapper(base_dir):
    pytest.importorskip("omnivoice")
    from tts_audiobook_tool.app_types import DeviceType
    from tts_audiobook_tool.tts_models.omnivoice_model import OmniVoiceModel

    model = OmniVoiceModel.__new__(OmniVoiceModel)
    model._model = make_model()  # type: ignore[assignment]
    model._device_type = DeviceType.CPU
    base_dir.mkdir()
    save_file({name: value.float() for name, value in state_of(model._model).items()},
              str(base_dir / "model.safetensors"))
    model._loras = OmniVoiceModel._make_lora_switcher(model._model, base_dir)
    return model


def test_omnivoice_preflight_checks_used_voices_and_rereads_after_a_run(tmp_path):
    # Only used LoRA voices are preflighted, and a later run rereads repaired files
    # rather than reusing an earlier failure.
    from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
    from tts_audiobook_tool.project import Project

    model = make_wrapper(tmp_path / "base")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    target = write_adapter(adapter_dir, use_dora=True)
    lora = VoiceLora(target)
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        Voice(), Voice(loras={"omnivoice_local": lora}),
    ])
    assert model.prepare_voice_loras(project, [0]) == []
    problems = model.prepare_voice_loras(project, [1])
    assert len(problems) == 1 and problems[0].startswith("Voice 2 LoRA") and "use_dora" in problems[0]
    write_adapter(adapter_dir)
    model.end_generation_run()
    assert model.prepare_voice_loras(project, [1]) == []


def test_omnivoice_activates_each_selected_voices_lora_before_generation(tmp_path, monkeypatch):
    # One loaded model serves adapter, no-adapter and different-strength voices;
    # rejected adapters block inference without changing previously merged weights.
    from tts_audiobook_tool.app_types.voice import Voice, VoiceLora
    from tts_audiobook_tool.project import Project

    model = make_wrapper(tmp_path / "base")
    adapter_dir, bad_dir = tmp_path / "adapter", tmp_path / "bad"
    adapter_dir.mkdir()
    bad_dir.mkdir()
    target = write_adapter(adapter_dir)
    bad = write_adapter(bad_dir, use_dora=True)
    project = Project(tts_model_type="omnivoice_local", voice_references=[
        Voice(loras={"omnivoice_local": VoiceLora(target)}), Voice(),
        Voice(loras={"omnivoice_local": VoiceLora(target, 0.5)}),
        Voice(loras={"omnivoice_local": VoiceLora(bad)}),
    ])
    calls = []

    def generate(**kwargs):
        calls.append(state_of(model._model))
        return []

    monkeypatch.setattr(model, "_generate_auto_voice", generate)
    for index in (0, 1, 2):
        assert model.generate_using_project(project, ["Hello."], voice_selection_index=index) == []
        expected = make_model()
        lora = project.voice_references[index].lora_for("omnivoice_local")
        if lora is not None:
            omnivoice_lora.apply_merge(expected, omnivoice_lora.prepare_lora(expected, lora.target, lora.strength).plan,
                                      delta_device="target")
        assert all(torch.equal(calls[-1][name], value) for name, value in state_of(expected).items())
    error = model.generate_using_project(project, ["Hello."], voice_selection_index=3)
    assert isinstance(error, str) and "voice 4" in error and "use_dora" in error
    assert len(calls) == 3
    assert model._loras.active_key == (target, 0.5)
    before_end = state_of(model._model)
    model.end_generation_run()
    assert model._loras.base.mode == "undecided" and model._loras.base.pinned_bytes == 0
    assert all(torch.equal(state_of(model._model)[name], value) for name, value in before_end.items())


def test_adapter_whose_targets_cant_be_restored_is_rejected_up_front(tmp_path):
    # A weight missing from the base checkpoint (eg a submodule loaded from elsewhere)
    # or a base file rewritten since loading would break the model at its first
    # switch, so preparation rejects the adapter before anything is merged.
    from tts_audiobook_tool.app_types.voice import VoiceLora

    model = make_wrapper(tmp_path / "base")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    lora = VoiceLora(write_adapter(adapter_dir))
    before = state_of(model._model)
    partial = {name: value.float() for name, value in before.items() if name != "audio_heads.weight"}
    save_file(partial, str(tmp_path / "base" / "model.safetensors"))
    error = model._loras.prepare(lora)
    assert "has changed since the model was loaded" in error

    model._loras = type(model)._make_lora_switcher(model._model, tmp_path / "base")
    error = model._loras.prepare(lora)
    assert "Can't restore" in error and "audio_heads.weight" in error
    assert all(torch.equal(state_of(model._model)[name], value) for name, value in before.items())


def test_chat_release_keeps_the_plan_but_drops_the_base_copy(tmp_path):
    # After each chat synthesis only the base copy is released; the active adapter's
    # plan stays, so the next chat line doesn't re-resolve or re-hash the adapter.
    from tts_audiobook_tool.app_types.voice import VoiceLora

    model = make_wrapper(tmp_path / "base")
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    lora = VoiceLora(write_adapter(adapter_dir))
    assert model._loras.activate(lora) == ""
    planned, planner = [], model._loras._planner
    model._loras._planner = lambda value: planned.append(value) or planner(value)
    model.release_inference_memory()
    assert model._loras.activate(lora) == "" and planned == []
    assert model._loras.base.mode == "undecided"
