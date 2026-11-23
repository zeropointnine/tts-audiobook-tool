"""Per-voice LoRA switching on a merged model (`lora_switch`): exact restore, in-place writes,
no-op switches, pinning policy and failure handling.

Needs torch (eg venv-breeze). The real-weights check is `testx/breeze_lora_poc.py`.
"""
import pytest

torch = pytest.importorskip("torch")

from tts_audiobook_tool.app_types.voice import VoiceLora
from tts_audiobook_tool.tts_models.lora_merge import LoraLoadError, LoraMergePlan, apply_merge
from tts_audiobook_tool.tts_models.lora_switch import (
    GIB, BaseWeightCache, LoraSwitcher, PreparedLora, SafetensorsBaseWeights, should_pin_base_weights,
)

RANK = 2


class Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q = torch.nn.Linear(4, 6, bias=False)
        self.k = torch.nn.Linear(4, 6, bias=False)
        self.head = torch.nn.Linear(4, 3, bias=False)


def make_model() -> Tiny:
    torch.manual_seed(0)
    return Tiny().to(torch.bfloat16)


def state_of(model) -> dict:
    return {name: p.detach().clone() for name, p in model.named_parameters()}


class MemorySource:
    """Base weights from a snapshot, counting reads."""

    def __init__(self, base: dict):
        self.base = base
        self.reads: list[str] = []

    def read(self, params, names):
        for name in names:
            self.reads.append(name)
            yield name, self.base[name].clone()

    def check(self, params, names):
        missing = sorted(set(names) - set(self.base))
        return f"no base value for {missing[0]}" if missing else ""


def make_plan(seed: int, targets=("q", "k"), strength: float = 1.0, replace_head: bool = False) -> LoraMergePlan:
    g = torch.Generator().manual_seed(seed)
    plan = LoraMergePlan(rank=RANK, alpha=4.0, scale=strength * 4.0 / RANK)
    for target in targets:
        plan.pairs[f"{target}.weight"] = (torch.randn(RANK, 4, generator=g), torch.randn(6, RANK, generator=g))
    if replace_head:
        plan.replacements["head.weight"] = torch.randn(3, 4, generator=g)
    return plan


PLANS = {"a": dict(seed=1), "b": dict(seed=2, targets=("q",), replace_head=True), "c": dict(seed=3, targets=("k",))}


def make_switcher(model, *, pin=False, identities=None, planner_error=None, emptied=None):
    base = state_of(model)
    source = MemorySource(base)
    cache = BaseWeightCache(source, pin_policy=lambda *_: pin, ram_info=lambda: (64 * GIB, 64 * GIB),
                            pin=lambda tensor: tensor.clone(),
                            empty_pinned_cache=(lambda: emptied.append(True)) if emptied is not None else lambda: None)
    identities = identities if identities is not None else {}
    logs: list[str] = []

    def planner(lora: VoiceLora) -> PreparedLora:
        if planner_error:
            raise LoraLoadError(planner_error)
        return PreparedLora(make_plan(strength=lora.strength, **PLANS[lora.target]),
                            identities.get(lora.target, lora.target), lora.target)

    return LoraSwitcher(model, cache, planner, log=logs.append), base, source, logs


def fresh_merge(name: str, strength: float = 1.0) -> dict:
    model = make_model()
    apply_merge(model, make_plan(strength=strength, **PLANS[name]), delta_device="target")
    return state_of(model)


def assert_weights(model, expected: dict):
    after = state_of(model)
    assert all(torch.equal(after[name], value) for name, value in expected.items())


def test_switching_is_exact_in_any_order():
    # Every switch restores the previous adapter's targets from the base copy, so weights
    # are bit-identical to a fresh merge (or the base), whatever came before
    model = make_model()
    switcher, base, _, _ = make_switcher(model)
    for name in ["a", "b", "a", "c", None, "b", "b"]:
        assert switcher.activate(VoiceLora(name) if name else None) == ""
        assert_weights(model, fresh_merge(name) if name else base)


def test_strength_is_part_of_the_lora():
    model = make_model()
    switcher, _, _, _ = make_switcher(model)
    switcher.activate(VoiceLora("a", 1.0))
    switcher.activate(VoiceLora("a", 0.5))
    assert_weights(model, fresh_merge("a", 0.5))


def test_parameters_are_written_in_place():
    # Captured CUDA graphs read weights by address, so storage must never move
    model = make_model()
    pointers = {name: p.data_ptr() for name, p in model.named_parameters()}
    switcher, _, _, _ = make_switcher(model, pin=True)
    for name in ["a", "b", None]:
        switcher.activate(VoiceLora(name) if name else None)
    assert {name: p.data_ptr() for name, p in model.named_parameters()} == pointers


def test_reactivating_the_active_lora_reads_and_writes_nothing():
    model = make_model()
    switcher, _, source, logs = make_switcher(model)
    switcher.activate(VoiceLora("a"))
    # The first merge onto pristine weights needs no base copy
    assert source.reads == [] and len(logs) == 1
    switcher.activate(VoiceLora("a"))
    switcher.activate(None)
    switcher.activate(None)
    assert sorted(source.reads) == ["k.weight", "q.weight"] and len(logs) == 2


def test_rewritten_adapter_is_merged_again_after_a_run():
    # Plans are re-read after end_run; a changed adapter file (identity) forces a re-merge
    model = make_model()
    identities = {"a": "sha-1"}
    switcher, _, source, _ = make_switcher(model, identities=identities)
    switcher.activate(VoiceLora("a"))
    switcher.end_run()
    switcher.activate(VoiceLora("a"))
    assert source.reads == []
    switcher.end_run()
    identities["a"] = "sha-2"
    switcher.activate(VoiceLora("a"))
    assert sorted(source.reads) == ["k.weight", "q.weight"]
    assert_weights(model, fresh_merge("a"))


def test_rejected_adapter_leaves_weights_unchanged():
    model = make_model()
    switcher, base, _, _ = make_switcher(model, planner_error="Not a Breeze LoRA")
    assert switcher.prepare(VoiceLora("a")) == "Not a Breeze LoRA"
    assert switcher.activate(VoiceLora("a")) == "Not a Breeze LoRA"
    assert_weights(model, base)
    assert not switcher.broken


@pytest.mark.parametrize("error_type", [LoraLoadError, RuntimeError])
def test_end_run_discards_cached_planning_errors(error_type):
    # Menu probes end the preparation scope, so a repaired adapter can be reread;
    # ordinary per-call preparation still reuses errors within one run.
    model = make_model()
    error = error_type("checksum mismatch")
    calls = []

    def planner(lora):
        calls.append(lora)
        if error is not None:
            raise error
        return PreparedLora(make_plan(**PLANS[lora.target]), "repaired", lora.target)

    switcher = LoraSwitcher(model, BaseWeightCache(MemorySource(state_of(model))), planner, log=lambda _: None)
    lora = VoiceLora("a")
    assert "checksum mismatch" in switcher.prepare(lora)
    error = None
    assert "checksum mismatch" in switcher.prepare(lora)
    assert len(calls) == 1
    switcher.end_run()
    assert switcher.prepare(lora) == ""
    assert switcher.activate(lora) == ""
    assert len(calls) == 2
    assert_weights(model, fresh_merge("a"))


def test_failed_restore_breaks_the_switcher():
    # Weights may be half-written, so every later switch fails until the model is reloaded
    model = make_model()
    switcher, _, source, _ = make_switcher(model)
    switcher.activate(VoiceLora("a"))

    def fail(params, names):
        raise OSError("disk gone")
        yield  # pragma: no cover

    source.read = fail
    error = switcher.activate(VoiceLora("b"))
    assert "unknown state" in error and switcher.broken
    assert switcher.activate(None) == error


def test_pinned_cache_reads_files_once_and_release_resets_it():
    model = make_model()
    emptied: list[bool] = []
    switcher, _, source, _ = make_switcher(model, pin=True, emptied=emptied)
    switcher.prepare(VoiceLora("b"))
    switcher.activate(VoiceLora("a"))
    switcher.activate(VoiceLora("b"))
    # Pinned at the first restore: the active adapter's targets plus every prepared one's
    assert switcher.base.mode == "pinned"
    assert sorted(source.reads) == ["head.weight", "k.weight", "q.weight"]
    switcher.activate(VoiceLora("a"))
    switcher.activate(VoiceLora("c"))
    assert len(source.reads) == 3
    switcher.end_run()
    assert switcher.base.mode == "undecided" and switcher.base.pinned_bytes == 0
    # Freed pinned tensors stay reserved in PyTorch's host cache until it's emptied
    assert emptied == [True]
    switcher.end_run()
    assert emptied == [True]


def test_adapter_with_unrestorable_targets_is_rejected_before_merging():
    # A merge whose targets couldn't be put back would break the model at the next
    # switch, so preparation refuses it and the weights are never touched
    model = make_model()
    switcher, base, _, _ = make_switcher(model)
    del switcher.base._source.base["head.weight"]
    error = switcher.activate(VoiceLora("b"))
    assert "Can't restore" in error and "head.weight" in error
    assert not switcher.broken and switcher.active_key is None
    assert_weights(model, base)
    assert switcher.activate(VoiceLora("a")) == ""


def test_pin_policy_is_sized_by_the_allocators_power_of_two_blocks():
    # A 6x4 bf16 weight is 48 bytes but takes a 64-byte pinned block
    model = make_model()
    sizes = []
    cache = BaseWeightCache(MemorySource(state_of(model)), pin_policy=lambda n, *_: sizes.append(n) or False,
                            ram_info=lambda: None)
    switcher = LoraSwitcher(model, cache, lambda lora: PreparedLora(make_plan(**PLANS[lora.target]), "x", "x"),
                            log=lambda _: None)
    switcher.activate(VoiceLora("a"))
    switcher.activate(None)
    assert sizes == [2 * 64]


def test_unpinned_cache_reads_on_every_restore():
    model = make_model()
    switcher, _, source, _ = make_switcher(model, pin=False)
    for name in ["a", "c", "a"]:
        switcher.activate(VoiceLora(name))
    assert switcher.base.mode == "files"
    assert source.reads == ["k.weight", "q.weight", "k.weight"]


def test_failed_pin_falls_back_to_files():
    model = make_model()
    base = state_of(model)
    cache = BaseWeightCache(MemorySource(base), pin_policy=lambda *_: True, ram_info=lambda: None,
                            pin=lambda tensor: (_ for _ in ()).throw(RuntimeError("cudaHostAlloc failed")))
    switcher = LoraSwitcher(model, cache, lambda lora: PreparedLora(make_plan(**PLANS[lora.target]), "x", "x"),
                            log=lambda _: None)
    switcher.activate(VoiceLora("a"))
    assert switcher.activate(None) == ""
    assert cache.mode == "files" and "pinning failed" in cache.mode_reason
    assert_weights(model, base)


@pytest.mark.parametrize("nbytes,device,ram,expected", [
    (4 * GIB, "cuda", (13 * GIB, 30 * GIB), True),     # 13 >= 4 + max(4, 4.5)
    (4 * GIB, "cuda", (8 * GIB, 30 * GIB), False),
    (1 * GIB, "cuda", (5 * GIB, 16 * GIB), True),      # margin floor is 4 GiB
    (1 * GIB, "cuda", (4.9 * GIB, 16 * GIB), False),
    (1 * GIB, "cpu", (60 * GIB, 64 * GIB), False),     # pinning only helps host-to-device copies
    (1 * GIB, "cuda", None, False),                    # unknown RAM: don't pin
])
def test_pin_policy(nbytes, device, ram, expected):
    assert should_pin_base_weights(int(nbytes), device, ram) is expected


@pytest.mark.parametrize("sharded", [True, False])
def test_safetensors_base_weights(tmp_path, sharded):
    # Reads sharded or single-file checkpoints, casting to the parameter's dtype
    import json
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    model = make_model()
    tensors = {name: p.detach().float().clone() for name, p in model.named_parameters()}
    if sharded:
        save_file({"q.weight": tensors["q.weight"]}, str(tmp_path / "a.safetensors"))
        save_file({"k.weight": tensors["k.weight"], "head.weight": tensors["head.weight"]},
                  str(tmp_path / "b.safetensors"))
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
            "q.weight": "a.safetensors", "k.weight": "b.safetensors", "head.weight": "b.safetensors"}}))
    else:
        save_file(tensors, str(tmp_path / "model.safetensors"))
    params = dict(model.named_parameters())
    read = dict(SafetensorsBaseWeights(str(tmp_path)).read(params, ["q.weight", "head.weight"]))
    assert set(read) == {"q.weight", "head.weight"}
    assert read["q.weight"].dtype == torch.bfloat16 and torch.equal(read["q.weight"], params["q.weight"].detach())

    params["q.weight"] = torch.nn.Parameter(torch.zeros(2, 2))
    with pytest.raises(ValueError, match="shape"):
        list(SafetensorsBaseWeights(str(tmp_path)).read(params, ["q.weight"]))
    if sharded:
        with pytest.raises(KeyError, match="index"):
            list(SafetensorsBaseWeights(str(tmp_path)).read({**params, "nope": params["k.weight"]}, ["nope"]))


def test_safetensors_base_weights_check_finds_unrestorable_weights(tmp_path):
    # check() reports, before any merge, what read() would fail on later: a key the
    # checkpoint lacks, a shape mismatch, a missing file, or a file rewritten after
    # the model was loaded (whose values the model was never loaded with)
    import os
    pytest.importorskip("safetensors")
    from safetensors.torch import save_file

    model = make_model()
    params = dict(model.named_parameters())
    tensors = {name: p.detach().float().clone() for name, p in params.items()}
    save_file({k: v for k, v in tensors.items() if k != "head.weight"}, str(tmp_path / "model.safetensors"))
    source = SafetensorsBaseWeights(str(tmp_path))
    assert source.check(params, ["q.weight", "k.weight"]) == ""
    assert "head.weight" in source.check(params, ["q.weight", "head.weight"])
    assert "shape" in source.check({**params, "q.weight": torch.nn.Parameter(torch.zeros(2, 2))}, ["q.weight"])

    stat = os.stat(tmp_path / "model.safetensors")
    os.utime(tmp_path / "model.safetensors", ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))
    assert "changed since the model was loaded" in source.check(params, ["q.weight"])
    with pytest.raises(ValueError, match="changed"):
        list(source.read(params, ["q.weight"]))

    assert "no model.safetensors" in SafetensorsBaseWeights(str(tmp_path / "missing")).check(params, ["q.weight"])
