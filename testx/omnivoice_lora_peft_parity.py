"""The app's OmniVoice LoRA merge matches peft's merge to within fp16 rounding.

Upstream applies an adapter with `PeftModel.from_pretrained(model, dir)
.merge_and_unload()` (`omnivoice.utils.lora.load_lora_adapter`). We merge by
hand instead (`tts_audiobook_tool/tts_models/omnivoice_lora.py`) to avoid a
peft dependency. This one-off check (not part of the
test suite) merges each adapter both ways into the production configuration
(fp16 on CUDA) and compares every parameter of the two models:

- untouched parameters must be bit-identical, and so must the full
  replacements (`audio_embeddings`, `audio_heads`), which both sides simply
  round from the adapter's F32 values to fp16
- LoRA-merged weights may differ by at most 1 fp16 ulp. The manual merge
  computes `W + scale * B @ A` in fp32 and rounds once; peft's matmul runs on
  the GPU and may round differently, so an off-by-one-ulp element is expected,
  anything larger is a real discrepancy. In practice (peft 0.22.0, which
  upcasts the adapter to fp32 and also rounds once) they are bit-identical
- the merge must actually change the base weights (guards against comparing
  two no-ops)

Each adapter is also re-run with `use_rslora` forced on, to cover the
alpha/sqrt(r) scaling path, which no public OmniVoice adapter uses.

peft is not a venv-omni dependency. If it is not importable, it is installed
with `pip --no-deps --target` into a throwaway directory (default
/tmp/omnivoice-peft-parity) and added to sys.path; the venv is untouched.

Run: ./venv-omni/bin/python -m testx.omnivoice_lora_peft_parity [adapter ...]
Adapters are local dirs or `owner/repo[/sub/dir]`; the defaults cover r=32
and r=16 (the latter in a repo subfolder).
"""

import argparse
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile

import torch

from tts_audiobook_tool.tts_models.omnivoice_lora import load_lora, resolve_adapter_dir

DEFAULT_ADAPTERS = ["My-Weird-Prompts/Herman-LoRA-V3", "wjcwqc/PaigeVoice/lora"]
BASE_MODEL = "k2-fsa/OmniVoice"
MAX_LORA_ULP = 1

State = dict[str, torch.Tensor]


def ensure_peft(target_dir: str) -> None:
    try:
        import peft  # noqa: F401  # type: ignore
        return
    except ImportError:
        pass
    if not os.path.isdir(os.path.join(target_dir, "peft")):
        print(f"installing peft (no deps) into {target_dir} ...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-deps", "--target", target_dir, "peft"],
            check=True,
        )
    sys.path.insert(0, target_dir)
    import peft  # type: ignore
    print(f"peft {peft.__version__} from {os.path.dirname(peft.__file__ or '')}")


def load_model():
    from omnivoice import OmniVoice  # type: ignore
    return OmniVoice.from_pretrained(BASE_MODEL, device_map="cuda:0", dtype=torch.float16)


def snapshot(model: torch.nn.Module) -> State:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


def release(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def merged_manually(adapter_dir: str) -> State:
    model = load_model()
    print(f"manual merge: {load_lora(model, adapter_dir).describe()}")
    state = snapshot(model)
    release(model)
    return state


def merged_with_peft(adapter_dir: str) -> State:
    from peft import PeftModel  # type: ignore
    model = load_model()
    model = PeftModel.from_pretrained(model, adapter_dir).merge_and_unload()
    state = snapshot(model)
    release(model)
    return state


def ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Elementwise distance in representable fp16 steps (sign-magnitude bits
    mapped onto a monotonic integer line, so +0 and -0 coincide)."""
    def monotonic(x: torch.Tensor) -> torch.Tensor:
        bits = x.contiguous().view(torch.int16).to(torch.int32)
        return torch.where(bits < 0, -(bits & 0x7FFF), bits)
    return (monotonic(a) - monotonic(b)).abs()


def adapter_targets(adapter_dir: str) -> tuple[set[str], set[str]]:
    """Model parameter names the adapter merges into, and those it replaces."""
    from safetensors import safe_open
    lora, replaced = set(), set()
    with safe_open(os.path.join(adapter_dir, "adapter_model.safetensors"), "pt") as f:
        for key in f.keys():
            name = key.removeprefix("base_model.model.").replace(".modules_to_save.default", "")
            for suffix in (".lora_A.weight", ".lora_B.weight"):
                if name.endswith(suffix):
                    lora.add(name.removesuffix(suffix) + ".weight")
                    break
            else:
                replaced.add(name)
    return lora, replaced


def compare(label: str, adapter_dir: str, base: State, manual: State, peft_state: State) -> bool:
    ok = True

    def fail(message: str) -> None:
        nonlocal ok
        ok = False
        print(f"  FAIL {message}")

    if manual.keys() != peft_state.keys() or manual.keys() != base.keys():
        fail(f"parameter names differ: manual-only {sorted(manual.keys() - peft_state.keys())[:5]}, "
             f"peft-only {sorted(peft_state.keys() - manual.keys())[:5]}")
        return False
    lora, replaced = adapter_targets(adapter_dir)
    if not lora <= manual.keys() or not replaced <= manual.keys():
        fail("adapter targets missing from the model")
        return False

    lora_elements = lora_identical = 0
    max_ulp = 0
    max_abs = max_delta = 0.0
    for name, m in manual.items():
        p = peft_state[name]
        if m.shape != p.shape or m.dtype != p.dtype:
            fail(f"{name}: {tuple(m.shape)} {m.dtype} vs {tuple(p.shape)} {p.dtype}")
            continue
        if name in lora:
            ulp = ulp_distance(m, p)
            lora_elements += m.numel()
            lora_identical += int((ulp == 0).sum())
            max_ulp = max(max_ulp, int(ulp.max()))
            max_abs = max(max_abs, float((m.float() - p.float()).abs().max()))
            max_delta = max(max_delta, float((m.float() - base[name].float()).abs().max()))
        elif not torch.equal(m, p):
            kind = "replacement" if name in replaced else "untouched parameter"
            fail(f"{kind} {name} differs (max abs {float((m.float() - p.float()).abs().max()):.3g})")
        elif name not in replaced and not torch.equal(m, base[name]):
            fail(f"untouched parameter {name} changed from base")
    for name in replaced:
        if torch.equal(manual[name], base[name]):
            fail(f"replacement {name} equals the base weight")

    print(f"  {len(lora)} LoRA-merged tensors, {lora_elements:,} elements: "
          f"{100 * lora_identical / max(lora_elements, 1):.4f}% bit-identical, "
          f"max {max_ulp} ulp, max abs diff {max_abs:.3g} (largest merge delta {max_delta:.3g})")
    print(f"  {len(replaced)} replacements bit-identical, {len(manual) - len(lora) - len(replaced)} untouched")
    if max_ulp > MAX_LORA_ULP:
        fail(f"LoRA-merged weights differ by {max_ulp} ulp (limit {MAX_LORA_ULP})")
    if max_delta == 0.0:
        fail("merge did not change any base weight")
    print(f"  {'PASS' if ok else 'FAIL'}: {label}")
    return ok


def rslora_variant(adapter_dir: str, scratch: str) -> str:
    variant = os.path.join(scratch, "rslora")
    os.makedirs(variant, exist_ok=True)
    os.symlink(os.path.realpath(os.path.join(adapter_dir, "adapter_model.safetensors")),
               os.path.join(variant, "adapter_model.safetensors"))
    config = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    config["use_rslora"] = True
    json.dump(config, open(os.path.join(variant, "adapter_config.json"), "w"))
    return variant


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("adapters", nargs="*", default=DEFAULT_ADAPTERS)
    parser.add_argument("--peft-dir", default="/tmp/omnivoice-peft-parity")
    parser.add_argument("--no-rslora", action="store_true", help="skip the forced-rsLoRA variants")
    args = parser.parse_args()

    ensure_peft(args.peft_dir)
    base_model = load_model()
    base = snapshot(base_model)
    release(base_model)

    results = []
    for target in args.adapters:
        adapter_dir = resolve_adapter_dir(target)
        config = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
        cases = [(f"{target} (r={config['r']}, alpha={config['lora_alpha']})", adapter_dir)]
        scratch = tempfile.mkdtemp(prefix="omnivoice-lora-parity-")
        if not args.no_rslora and not config.get("use_rslora"):
            cases.append((f"{target} with use_rslora forced on", rslora_variant(adapter_dir, scratch)))
        try:
            for label, case_dir in cases:
                print(f"\n{label}")
                results.append(compare(label, case_dir, base, merged_manually(case_dir), merged_with_peft(case_dir)))
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    print(f"\n{sum(results)}/{len(results)} cases passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
