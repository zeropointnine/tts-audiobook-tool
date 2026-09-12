# Chatterbox Multilingual V3 integration

Last updated: 2026-09-11

## Purpose

This document records the non-obvious compatibility, migration, startup, model-selection, and memory-management decisions involved in adding Chatterbox Multilingual V3. These details are coupled across the dependency file, project serialization, model lifecycle, menus, startup checks, and tests; changing only the model loader is not sufficient.

## Upstream dependency and model selection

### Why Chatterbox is installed from Git

The PyPI release `chatterbox-tts==0.1.7` predates Multilingual V3 support. V3 support was added upstream after that release, but upstream did not increment the package version: both the old PyPI wheel and the newer Git source report version `0.1.7`.

`requirements-chatterbox.txt` therefore pins the Chatterbox source repository rather than the PyPI release:

```text
chatterbox-tts @ git+https://github.com/resemble-ai/chatterbox.git@65b18437192794391a0308a8f705b1e33e633948
```

The pin is the official V3 release-era commit. V3 support originally landed in upstream commit `3f35dfc8fbe63e5b29793289dc68f1875bb317a5`; `65b18437192794391a0308a8f705b1e33e633948` includes the subsequent V3 release documentation and model-pack changes.

Do not replace this with `chatterbox-tts==0.1.7`: that silently restores the V2-only loader while appearing to preserve the same version.

Because both builds report version `0.1.7`, `pip install --upgrade -r requirements-chatterbox.txt` may consider an already-installed PyPI wheel current and leave it in place. When updating an existing environment from that wheel, uninstall Chatterbox before reinstalling the requirements:

```shell
pip uninstall chatterbox-tts
pip install -r requirements-chatterbox.txt
```

### Python and PyTorch requirements remain unchanged

The V3-capable Git pin does not change the supported environment for this app. Keep the Chatterbox virtual environment on Python 3.11 with:

```text
torch==2.6.0
torchaudio==2.6.0
```

The pinned upstream package declares `requires-python = ">=3.10"`, so its dependency set can resolve on Python 3.12 with Torch 2.6. However, upstream documents Python 3.11 as its developed-and-tested environment, and the app intentionally retains its Python 3.11 startup guard rather than directing users to an unneeded environment migration.

Torch 2.8 is not compatible with the package metadata at this pin: upstream explicitly requires `torch==2.6.0` and `torchaudio==2.6.0` for Python versions below 3.14. Adding the app's usual Torch 2.8 pins therefore makes pip resolution fail. Using Python 3.12 and Torch 2.8 would require overriding or patching upstream dependency metadata and independently validating installation, imports, model loading, and real generation; it is not the supported Chatterbox setup.

### V3 is a T3 checkpoint, not a new general model repository

The general V2 and V3 multilingual models both use the Hugging Face repository:

```text
ResembleAI/chatterbox
```

The V3 T3 checkpoint is `t3_mtl23ls_v3.safetensors`. The upstream API selects it through a loader argument:

```python
ChatterboxMultilingualTTS.from_pretrained(
    device=device_value,
    t3_model="v3",
)
```

Omitting `t3_model` still selects V2. The app must therefore always pass the selector explicitly for either multilingual variant. `ChatterboxType.multilingual_t3_model` is the central mapping:

- Multilingual V2 -> `"v2"`
- Multilingual V3 -> `"v3"`
- Turbo -> `None`

Turbo continues to use `ChatterboxTurboTTS.from_pretrained(device=...)` and must not receive `t3_model`.

### Weight revisions are not pinned

The Git pin fixes the installed Python implementation, but upstream `from_pretrained()` downloads `ResembleAI/chatterbox` with Hugging Face `revision="main"`. Model assets can therefore change independently of the application and Git dependency pin.

For bitwise-reproducible weights, the app would need to call `huggingface_hub.snapshot_download()` with an explicit revision and then use:

```python
ChatterboxMultilingualTTS.from_local(
    snapshot_dir,
    device=device_value,
    t3_model="v3",
)
```

## Chatterbox model variants and project compatibility

`ChatterboxType` now contains three user-visible variants:

| Enum member | Serialized ID | Role |
|---|---|---|
| `MULTILINGUAL_V3` | `multilingual-v3` | Recommended multilingual checkpoint and default for new projects |
| `MULTILINGUAL_V2` | `multilingual` | Legacy multilingual checkpoint |
| `TURBO` | `turbo` | English-only distilled model |

`MULTILINGUAL_V3` is the first enum member because new projects initialize `chatterbox_type` from `list(ChatterboxType)[0]`.

The legacy serialized ID `multilingual` intentionally maps to V2. Existing project files therefore continue using the checkpoint that created their previous output instead of silently changing model behavior. A source-compatibility alias, `ChatterboxType.MULTILINGUAL`, also points to `MULTILINGUAL_V2` for older callers.

Code that applies to both multilingual checkpoints must use:

```python
chatterbox_type.is_multilingual
```

It must not compare only against one multilingual enum member. This applies notably to:

- exposing CFG and exaggeration in `voice_chatterbox_menu.py`;
- choosing multilingual generation arguments in `chatterbox_model.py`;
- validating/selecting project language codes in `project_menu.py`.

The model lifecycle already treats `chatterbox_type` as an instantiation parameter in `Tts.get_model_params_using_project()` and `Tts.set_model_params()`. Switching among V2, V3, and Turbo invalidates the current Chatterbox instance and loads the selected model on the next use.

## Hyperparameter defaults

Upstream changed the multilingual repetition-penalty default when V3 support landed:

- Multilingual V2: `2.0`
- Multilingual V3: `1.2`
- Turbo: `1.2`

The app represents an unset project value with `-1`. `ChatterboxBaseModel.default_repetition_penalty(model_type)` resolves that sentinel according to the selected variant. Explicit user values are stored in per-variant project fields — `chatterbox_ml_v2_repetition_penalty`, `chatterbox_ml_v3_repetition_penalty`, and `chatterbox_turbo_repetition_penalty` — because a value tuned against one variant's operating point (V2 default `2.0`, V3 default `1.2`) does not transfer meaningfully to another.

The old constant name `DEFAULT_REPETITION_PENALTY_ML` remains as a V2-compatible alias. New code should use the explicit V2/V3 constants or `default_repetition_penalty()`.

Other relevant defaults were not changed by this integration. In particular, the app still uses its own `DEFAULT_TOP_P = 0.95`, even though the upstream multilingual default is `1.0`.

## Startup compatibility guard

### Why package version alone cannot work

Both the incompatible PyPI wheel and the compatible Git-pinned source package report version `0.1.7`. Installation provenance is also not the relevant contract: a compatible wheel or fork should work, while an older Git checkout should not pass merely because it has PEP 610 metadata.

`ChatterboxApiDetect` therefore performs a source-level feature probe without importing Chatterbox or Torch. It locates the installed `chatterbox/mtl_tts.py` through distribution metadata, parses it with Python's `ast` module, and checks for the V3-era feature cluster used by the app:

- `ChatterboxMultilingualTTS.from_pretrained()` explicitly accepts `t3_model`;
- `_resolve_multilingual_t3_model()` exists; and
- the module contains the V3 checkpoint name `t3_mtl23ls_v3.safetensors`.

This intentionally establishes API compatibility rather than repository provenance. A fork or backport with the required features passes; an old Git install without them fails.

### Blocking behavior

After TTS environment detection and the Python-version check, `Start.exit_on_incompatible_chatterbox_package()` runs before the generic new-package check. It only applies when `Tts.get_type() == TtsModelType.CHATTERBOX`.

When the required feature cluster is absent or cannot be inspected, startup prints `HINT_CHATTERBOX_PACKAGE_UPDATE` and exits with status 1. The remediation is to uninstall Chatterbox, then reinstall the requirements so pip cannot mistake the PyPI wheel for the Git build with the identical version:

```shell
pip uninstall chatterbox-tts
pip install -r requirements-chatterbox.txt
```

This is a blocking compatibility message, not a persisted one-time hint: continuing with a V3-default project would eventually call a loader that does not provide the required V3-era API.

## V3 availability hint for V2 projects

`HINT_CHATTERBOX_MULTILINGUAL_V3` is a non-blocking, persisted contextual hint. On the first main-menu display in a process, it is eligible when:

- the active TTS type is Chatterbox; and
- the current project selects `ChatterboxType.MULTILINGUAL_V2`.

It tells the user that V3 is available and directs them to the Voice settings. It is not shown for V3, Turbo, or another TTS model.

The hint uses the normal preference-backed hint mechanism. Consequently, **Options > Reset one-time hints** can make it appear again while a project still uses V2. Deleting preferences also removes the acknowledgement. Once the project is changed to V3, the condition itself is false, so resetting hints does not cause it to appear for that project.

## Memory-management interaction with the library update

Commit `879bd3b9f0445f54e91ce78e83d918fa7210fa94` previously addressed two independent Chatterbox leaks.

### Alignment-analyzer hook leak: superseded upstream

The old multilingual library created a new `AlignmentStreamAnalyzer` during generation. Each analyzer registered transformer forward hooks that accumulated between segments, retained CPU tensors, increased RSS, and progressively slowed inference. The app removed those hooks after every generation.

The V3-capable upstream source removes `AlignmentStreamAnalyzer` and its hook machinery entirely and removes the now-obsolete:

- `ChatterboxModel._strip_alignment_analyzer_hooks()`;
- `_ALIGNED_ATTN_LAYER_INDICES`;
- post-generation hook cleanup;
- `tests/test_chatterbox_alignment_hooks.py`;
- `testx/chatterbox_hook_leak.py`.

This upstream change applies to V2 as well as V3 when using the new library. Updated V2 is not behaviorally identical to the former V2 library because upstream removed the analyzer's alignment-based stopping/repetition heuristics rather than merely disposing its hooks.

### Why a substantial inference-speed increase is plausible

The analyzer removal is also a real inference-path optimization, not only leak cleanup. The old multilingual path did all of the following during every autoregressive T3 decode step:

- requested attention outputs from the full Llama backbone with `output_attentions=True`;
- changed the configured attention implementation from `sdpa` to `eager`;
- ran Python forward hooks on three attention layers; and
- copied their selected attention maps from the accelerator to CPU for alignment analysis.

The V3-era code deletes those hooks and calls both the initial T3 pass and every cached decode step with `output_attentions=False`. The model configuration still explicitly selects `attn_implementation="sdpa"`. This removes the per-token attention-materialization and device-to-host-copy work and allows PyTorch's optimized scaled-dot-product-attention kernels to run. Because T3 is autoregressive and its transformer pass is a major part of total synthesis time, a noticeable—and on suitable CUDA workloads potentially substantial—wall-clock improvement is technically credible.

There are two distinct improvements to keep separate:

1. **Lower cost per generated speech token:** no analyzer, eager-attention fallback, or attention-map CPU copies.
2. **Fewer generated tokens in analyzer failure cases:** the upstream PR reports old V2 Chinese samples expanding to 21–28 seconds of audio for approximately 9 seconds of text, versus 4–11 seconds afterward. Avoiding such unwanted continuation can reduce total request time independently of faster token throughput.

The project's former `_strip_alignment_analyzer_hooks()` workaround prevented hooks from accumulating *between* segments, avoiding the progressive long-run slowdown. It could not avoid the analyzer cost during the current segment, and it did not restore the transformer from eager attention to SDPA. The new upstream implementation therefore can still be materially faster in this app even though the hook-leak workaround was already active.

Evidence and limitations:

- Upstream [PR #516](https://github.com/resemble-ai/chatterbox/pull/516), merged as [`3f35dfc`](https://github.com/resemble-ai/chatterbox/commit/3f35dfc8fbe63e5b29793289dc68f1875bb317a5), explicitly says that disabling attention outputs "unlocks SDPA optimized attention." Its verification covers output duration, tail noise, and post-change long-form generation, but does not publish a controlled before/after tokens-per-second benchmark.
- Upstream [issue #504](https://github.com/resemble-ai/chatterbox/issues/504) independently documents the three-hooks-per-call leak and the analyzer's mutation from `sdpa` to `eager`, reproduced on CPU/MPS, Linux CUDA, and Windows CUDA.
- Upstream [issue #335](https://github.com/resemble-ai/chatterbox/issues/335) identifies the Llama/T3 backbone as the largest measured inference cost and the analyzer's required attention weights as incompatible with SDPA-based acceleration.
- [Issue #525](https://github.com/resemble-ai/chatterbox/issues/525) independently identifies `output_attentions=True` as blocking SDPA/flash attention. Its reported ROCm test did not improve iteration rate, so the magnitude is hardware, sequence-length, and workload dependent; its claimed larger CUDA impact was not accompanied by CUDA benchmark numbers.

Conclusion: the reported speedup is consistent with the actual code change and has a strong technical mechanism behind it. It should be treated as credible but locally observed rather than as a specific upstream-guaranteed multiplier until a controlled A/B benchmark uses the same checkpoint, prompt, voice conditionals, seed, generation parameters, warm-up state, and generated-token count.

### PerTh CPU leak: workaround retained

The separate PerTh watermarking workaround remains. Chatterbox still constructs its watermarker on CPU, and the app's measured CPU path retained native memory per generated segment. For CUDA-backed Chatterbox models, `_use_gpu_watermarker()` replaces it with:

```python
perth.PerthImplicitWatermarker(device="cuda")
```

The current PerTh API still supports this. No confirmed upstream fix for the measured CPU leak was identified, so the workaround remains active for Multilingual V2, Multilingual V3, and Turbo on CUDA.

Chatterbox's Git dependency currently references PerTh from its own `master`. A future PerTh update should not prompt removal of this workaround without rerunning a long-duration RSS test on the affected platform, especially Windows/CUDA.

## Voice cloning and generation API compatibility

The V3 integration continues relying on several Chatterbox internals used by the existing multi-voice cache:

- `prepare_conditionals()`;
- mutable `model.conds`;
- multilingual/Turbo `Conditionals` objects;
- `T3Cond` fields;
- the model's `device` attribute.

These remained compatible at the pinned upstream commit. V2 and V3 share the tokenizer, voice encoder, S3Gen assets, built-in conditions, and conditioning layout; the selected T3 checkpoint changes.

Cached conditionals remain CPU clones. Before generation, the app makes a fresh on-device copy and assigns it to `model.conds`, preventing library-side mutation from changing the cached value.

The upstream V3-era library additionally trims the final speech token's approximately 40 ms of degraded audio. The app should not add a second tail-trimming step.

## Menu and labeling behavior

The Voice menu's model selector lists V3, V2, and Turbo using their enum labels. V3 appears first. CFG and exaggeration remain visible for either multilingual version, while Turbo retains its separate `top_k` setting.

The repetition-penalty menu label includes the selected multilingual version and displays that version's resolved default. V2 and V3 use independent serialized custom-value fields (`chatterbox_ml_v2_repetition_penalty` and `chatterbox_ml_v3_repetition_penalty`), so switching versions never carries an explicit override across operating points. Loading an older project file remaps the former shared key `chatterbox_ml_repetition_penalty` into the V2 field only (see `ProjectLoadUtil.remap_legacy_keys()`); V3 then resolves to its default unless set explicitly.

The main menu and model status derive their Chatterbox label from `project.chatterbox_type.label`, making the active V2/V3 distinction visible.

## Tests and maintenance checklist


- `tests/test_chatterbox_variants.py`
  - new-project V3 default;
  - legacy V2 ID compatibility;
  - V3 serialization round trip;
  - checkpoint selectors and per-version defaults;
  - model lifecycle parameters.
- `tests/test_chatterbox_voice_clone_cache.py`
  - V2 and V3 loader arguments;
  - Turbo exclusion from `t3_model`;
  - existing conditional-cache behavior against the updated library.
- `tests/test_main_menu_heading.py`
  - V2 availability hint triggering;
  - V3 hint suppression.
- `tests/test_start_chatterbox_package.py`
  - old PyPI provenance detection;
  - Git installation acceptance;
  - alternate-version and missing-package behavior;
  - blocking startup behavior and model scoping.

When updating Chatterbox again:

1. Verify the pinned Git commit and whether upstream finally increments its package version.
2. Recheck the `from_pretrained(..., t3_model=...)` and `from_local()` signatures.
3. Confirm whether omitting `t3_model` still defaults to V2.
4. Verify V2/V3 checkpoint names and supported language IDs.
5. Check whether `Conditionals`, `T3Cond`, `prepare_conditionals()`, or `model.conds` changed.
6. Review upstream repetition-penalty and other generation defaults.
7. Reassess the PEP 610 startup guard; remove it once supported PyPI releases are unambiguous.
8. Reassess, but do not casually remove, the CUDA PerTh workaround.
9. Check whether the Hugging Face model revision should be pinned.
10. Run focused tests in both `venv-base` and `venv-cb`, followed by an actual V3 generation smoke test when model weights and suitable hardware are available.
