# TTS Model Architecture

Last updated: 2026-08-20

## Overview

Each supported local TTS model is isolated in its own Python virtual environment. Metadata, class references, and UI routing use a set of layered abstractions. For process mode, capability discovery, and choosing `Project.tts_model_type`, refer to [TTS model selection rules](<tts-model-selection.md>). Adding a new model means integrating the relevant layers; availability discovery does not automatically implement a new model.

Models come in two flavors:

- **Local models** — all interactive inference runs in the application's long-lived spawned model worker; each model still has a dedicated venv (e.g. `venv-cb`, `venv-g`). The interactive main process is forbidden from constructing models (see `model_runtime.require_model_owner()`); the REST server and `testx/` scripts keep their own in-process paths.
- **Server variants** — inference is delegated to an external SGL-Omni or audio.cpp server over HTTP. Several model variants share the server-mode venv (`venv-client`). Refer to [TTS model selection rules](<tts-model-selection.md>) for selection and exact-server-entry binding.

The process/UI boundary, worker ownership rules, cancellation behavior, and process-safety invariants are documented in [model-worker-architecture.md](model-worker-architecture.md).

---

## Virtual Environments and Requirements Files

Every local model has a dedicated `requirements-<model>.txt` at the project root:

```
requirements-base.txt       # app-only deps; the canonical app dependencies block
requirements-chatterbox.txt
requirements-fish-s1.txt
requirements-fish-s2.txt
requirements-glm.txt
requirements-higgs-v2.txt
requirements-indextts2.txt
requirements-mira.txt
requirements-moss.txt
requirements-omnivoice.txt
requirements-pocket.txt
requirements-qwen3tts.txt
requirements-vibevoice.txt
requirements-remote.txt    # remote-client mode; no model-specific deps (see below)
```

Each local model file follows the same two-section layout: model-specific dependencies appear at the top, followed by a `# App dependencies` comment and then the app's own dependencies. For example:

```
# Model dependencies
chatterbox-tts==0.1.7

# App dependencies
faster-whisper==1.2.1
torch==2.6.0
torchaudio==2.6.0
audiotsm==0.1.2
...
```

The app dependencies section is essentially identical across all requirements files (keep it in sync with `requirements-base.txt`). The one exception is `torch`/`torchaudio`, whose versions may differ between models depending on compatibility requirements of the model library (we favor 2.8.0 as much as possible; current deviations are Chatterbox at 2.6.0 and MOSS at 2.9.1).

[`requirements-remote.txt`](<../requirements-remote.txt>) is the exception to the two-section layout: it contains only the app dependencies, plus an artificial [launcher-marker package](<../launcher_markers/remote_client/pyproject.toml>) (`./launcher_markers/remote_client`) that identifies the venv as a remote TTS client for either SGL-Omni or audio.cpp. The historical installed module `tts_audiobook_tool_sgl_omni_marker` remains recognized for existing non-editable installations; no reinstall is required, even though its source package has been removed. Both packages may coexist without adding another model capability. Server-client functionality itself requires no extra libraries beyond the base app deps.

The corresponding virtual environments (e.g. `venv-chatterbox`, `venv-fish-s1`, `venv-glm`) live at the project root and are selected externally when launching the app — the app itself has no venv-switching logic. One venv → one local model. `venv-client` is the shared remote-client venv for SGL-Omni and audio.cpp, and `venv-base` is a dev-only venv with just the app dependencies.

When implementing a new local model, create `requirements-<newmodel>.txt` first and validate it in isolation before wiring anything into the app. Copy the app dependencies block from `requirements-base.txt` (or an existing model file) and adjust `torch`/`torchaudio` versions only if the model library requires it.


---

## `TtsModelSpec` and `TtsModelType`

**File:** [tts_audiobook_tool/tts_models/tts_model_type.py](tts_audiobook_tool/tts_models/tts_model_type.py)

`TtsModelSpec` is a `NamedTuple` holding application metadata for a model, loaded from [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>). `TtsModelType` is a canonical handle with a stable `.id` and a `.value` property resolving the current spec, not an enum with model-named members. Every shipped entry is installed in exact TOML order, including data-only models and the `"none"` placeholder; there is no separate Python declaration inventory.

- Use `TtsModelType.require_by_id("glm_local")` for hardcoded model IDs, base-class metadata, and handle-keyed registries. It returns the registered canonical handle and raises `ValueError` naming an unknown ID; it never creates an unregistered handle.
- Use `TtsModelType.get_by_id(saved_id)` for persisted selections or external IDs whose absence is expected. Unknown IDs return the registered `"none"` placeholder; retain the original saved string rather than normalizing it to the fallback.
- For a known ID both lookups return the same object, without loading a model or performing discovery. Metadata overlays replace the spec but retain handle identity, equality, and hashing; catalog reset restores the imported handles and specs.
- For simple classification use `model.id == "glm_local"`. For structural dispatch match `model.id` with literal string cases; function calls are not value patterns.

Key fields of `TtsModelSpec` most relevant to integration:

| Field | Purpose |
|---|---|
| `id` | Stable string identifier used for serialization |
| `backend_kind` | Catalog classification of how a variant executes; refer to [TTS model selection rules](<tts-model-selection.md>) for its relationship to process mode and selection |
| `local_module_test` | Probe used to detect whether the model's library is installed in the active venv: a plain importable module name, or `dist:<package>[==<version>]` tested via `importlib.metadata` (the `dist:` form is how Fish S1 vs S2 disambiguate, since both ship as `fish-speech` at different versions) |
| `local_module_exclude` | Optional list of probes (same forms as `local_module_test`) that veto a positive match when any is present; for a library that another model's venv also contains (eg `qwen3tts_local` excludes `breeze_tts`, since Breeze depends on `qwen_tts`). Applied in both `Tts.init_local_model_type()` and the `launch.py` venv probe |
| `local_torch_devices` | Supported torch device types for local inference (empty for server variants and models that don't take a device) |
| `file_tag` | Short identifier used in generated filenames (e.g. `"glm"`, `"chatterbox"`) |
| `default_output_sample_rate` | Native/default output sample rate; used directly by static-rate models and as the fallback for dynamically configured models |
| `requires_voice` | Whether generation is blocked without a voice clone |
| `can_stream` | Whether the model supports streaming chunk callbacks |
| `requires_ffmpeg_libs` | Whether the model requires FFmpeg shared libraries, not just the executable (usually because of TorchCodec) |
| `un_all_caps` | Force lowercase on all-caps prompts; set for models that perform poorly on them |
| `requirements_file_name` | The `requirements-<model>.txt` filename for this model |
| `ui` | Dict of UI strings/values: `proper_name`, `short_name`, `voice_path_console`, `voice_path_requestor`, `project_links`; optional positive `voice_sample_max_duration_s` is a warning-only recommendation, matching the upper duration in voice-path guidance |
| `output_filters` | Case-sensitive worker console substrings filtered out of output history (generation, realtime playback, and chat initialization). Blank/whitespace-only lines directly following a filtered line are dropped with it, so multi-line warnings leave no residue |
| `substitutions` | List of `(before, after)` string pairs applied to prompts before inference |
| `can_instructions` | Whether the model accepts per-voice instructions (`Voice.instructions`, keyed by model id). Defaults to `False`; local `[models.spec].can_instructions` sets the static default (currently `breeze_tts_2_local` and `omnivoice_local`). Ask `Tts.get_model_support(project).can_instructions(project, instance=None)`, not the static spec: local Qwen3 answers from its checkpoint type (CustomVoice/VoiceDesign only), and audio.cpp and SGL-Omni use their dedicated `voice_instructions` declaration (`[models.audio_cpp]` / `[models.sgl_omni.behavior]`) plus runtime behavior (CosyVoice3 Instruct mode only, on either backend). Whether they still apply to a voice that has a sample is the separate `uses_instructions_with_sample(project)` (False only for SGL-Omni AuK/AuK-Flash, which use them as the voice description of sample-less voices). Instructions are not ordinary model settings; OmniVoice's accepted text is tag-based, not arbitrary prose |
| `can_voice_clone` | Whether the model can clone a voice from a voice sample. Defaults to `True` (every shipped real model); the `none` placeholder must declare `false`. Overridable by the optional `can_voice_clone` key in a local model's `[models.spec]`. Not yet declarable for server backends. This is only the catalog default: ask `Tts.get_model_support(project).can_voice_clone(project, instance=None)`, which models may override from runtime state |

Primary voice/transcript pairs are stored once in top-level `Project.voice_references`, shared by every model in that project. Catalog/registry voice and transcript bindings (`REGISTRY.voice_binding()`, `.transcript_binding()`) remain compatibility/capability metadata, not separate clone-list owners; model-specific support and transcript requirements still govern consumption. Parameters and batch size retain catalog-declared per-model storage, and `TtsModelType.can_batch()` derives from the orchestration binding. See [Project Spec v5](<project-spec-v5.md>) for storage and v1/v2/v3 migration.

The effective output sample rate is exposed via `TtsBaseModel.get_output_sample_rate(project, instance)`. Its default implementation returns `INFO.default_output_sample_rate`; models whose rate depends on project configuration can override it (currently GLM, for its selectable samplerate). It is used for playback/export paths; it deliberately does not affect voice clone audio — every model resamples reference audio internally, so imported voice clones are simply resampled to the app-native 48 kHz (`SoundPipeline.apply_voice_clone_post_processing()`), peak-normalized, and saved under the project's `voice/` subdir as `<stem>.flac` (no model tag; `ProjectVoiceUtil.resolve_voice_file_path()` prefers the subdir and falls back to the legacy project-root location for older projects).

### Model selection

Refer to [TTS model selection rules](<tts-model-selection.md>) for the canonical rules covering process mode, available types, the saved project selection, automatic reconciliation, runtime binding, and status display. The former detection and override guidance has been replaced by that reference.

---

## Local Model Class Hierarchy and Configured Servers

**Directory:** [tts_audiobook_tool/tts_models/](tts_audiobook_tool/tts_models/)

Local inference models use the two-level `TtsBaseModel` subclass convention described below. SGL-Omni variants instead use validated JSON definitions, `ConfiguredModelSupport` and one `SglOmniBackendAdapter` for generation; they have no per-variant Python generation subclasses. The shared HTTP transport remains `SglOmniUtil`.

### Level 1 — `TtsBaseModel` (abstract)

**File:** [tts_audiobook_tool/tts_models/tts_base_model.py](tts_audiobook_tool/tts_models/tts_base_model.py)

Defines the interface all models must satisfy:

- `INFO: TtsModelSpec` — class-level attribute; `__init_subclass__` raises `TypeError` if missing
- `kill() -> None` — abstract; nulls out internal model references to aid garbage collection
- `generate_using_project(project, prompts, force_random_seed, on_stream_chunk, on_stream_end, voice_selection_index) -> list[Sound] | str` — abstract; the main generation entry point (stream callbacks only used when `INFO.can_stream`)
- `get_output_sample_rate(project, instance) -> int` — concrete classmethod returning `INFO.default_output_sample_rate`; GLM overrides it to return the project-configurable `glm_sr`
- `get_max_words_range_reco(project, instance) -> tuple[int, int, str]` — concrete classmethod returning the app-recommended max-words-per-segment range for the model plus an optional rationale string (empty by default); the range default comes from the global constants (`MAX_WORDS_PER_SEGMENT_RECO_RANGE`), and models with model-specific recommendations override it with their own class constants (currently GLM, Higgs V2, IndexTTS2, Chatterbox, VibeVoice, and the `"none"` placeholder)
- `massage_for_inference(text) -> str` — concrete; applies `INFO.substitutions`; subclasses may override-and-super
- `prepare_text_for_inference(project, text) -> str` — concrete; the full pre-inference pipeline: project word substitutions → generic prompt normalization (incl. `un_all_caps`) → `massage_for_inference`
- `clear_stream_state()` / `clear_continuation()` — concrete hooks for streaming and rolling-continuation state
- `RETAINS_MULTIPLE_VOICE_CLONES = False` — class attribute; opt-in to retaining prepared voice clones for multiple source files simultaneously (see below)
- `_get_or_create_voice_clone(source_path, transcript, factory)` — concrete; the shared get-or-create for prepared voice clones ([Voice Clone Cache](#voice-clone-cache))
- `clear_voice_clone_cache()` — concrete, idempotent; drops all prepared clone values (models call it from `kill()`)

Classmethods and helpers with default implementations (override when the defaults don't apply):

- `get_menu_text(project, instance) -> str`
- `get_blocking_issues(project, instance) -> list[ReadinessIssue]` (default: no voice-clone checks; voice state is validated by job-scoped `VoiceMenuShared.validate_voices` / `get_voice_problems` and at generation time; it checks the distinct clamped voice assignments of the job's lines, or voice 1 for chat/literal previews)
- `get_warning_issues(project) -> list[str]` — instance method (default: max-words warning). The "voices may vary" note is not a model warning: run sites (generation, realtime playback, chat inspection, server startup) use `voice_variation_warning.get_run_warnings(project, instance, phrase_indices)`, which prepends it when any voice in play for the run (`ProjectVoiceUtil.get_job_voice_indices`, as in the voice pre-flight) has no sample or preset (`resolve_voice_source` kind "instructions" or "builtin") and the active model settings either allow cloning (`can_voice_clone`) or expect voice-defining instructions (`expects_voice_instructions`), without a fixed default voice (`has_fixed_default_voice`, eg a VibeVoice LoRA). Missing required samples/presets resolve as "missing" and belong to validation instead. Qwen3-TTS VoiceDesign (local and audio.cpp) warns even with retained inactive samples/presets, while Base and CustomVoice do not. This is one rule for every model, so local and SGL-Omni MOSS now get it too. In-play voices with instructions get the model's optional `[models.spec.ui].instructions_voice_variation_note` instead (AuK, AuK-Flash); a run mixing described and blank voices gets both notes. VoiceDesign uses the shared `GENERIC_VOICE_VARIATION_NOTE` constant with or without instructions, grouping affected voices into one warning. Each note ends with the voice numbers it covers, eg " (Voices 1, 3)". Max-words warning text is shared via `model_support.make_max_words_exceed_warning`
- `get_voice_tag(project) -> str`
- `get_primary_voice_value(project) -> str`
- `should_trim_trailing_token_noise(project, instance) -> bool`
- `can_hallucinate_music(project, instance) -> bool`

### Level 2 — `AbcBaseModel(TtsBaseModel)`

Example: [tts_audiobook_tool/tts_models/glm_base_model.py](tts_audiobook_tool/tts_models/glm_base_model.py)

- Must **not** import any model library at module level
- Assigns metadata through strict lookup, e.g. `INFO = TtsModelType.require_by_id("glm_local").value`
- Inherits `get_output_sample_rate(project, instance)` for static-rate models; models with configurable output rates, such as GLM, override it
- Implements classmethods and any model-specific constants or static helpers
- User-tunable parameter defaults, bounds and choices live in the catalog (`[models.local.parameters]`, see [Local catalog parameters and menus](#local-catalog-parameters-and-menus)), exposed as `LOCAL = require_local_definition("glm_local")`; fixed implementation constants (token limits, repo pins, frame rates) stay in code
- This is the class registered in `Tts.get_class_for_type()` and used for all non-instance operations (readiness checks, voice tags, etc.)

```python
class GlmBaseModel(TtsBaseModel):
    INFO = TtsModelType.require_by_id("glm_local").value
    LOCAL = require_local_definition("glm_local")
    SAMPLE_RATES: list[int] = list(LOCAL.choice("sr").values)
```

### Level 3 — `AbcModel(AbcBaseModel)`

Example: [tts_audiobook_tool/tts_models/glm_model.py](tts_audiobook_tool/tts_models/glm_model.py)

- Model library imports live here and **only** here
- Implements `__init__` (loads weights, sets up state)
- Implements `generate_using_project()` — reads voice file path, transcript, seed, etc. from `project`, then delegates to a more parameter-explicit internal method. Catalog parameters are read with `LOCAL.resolve(project, name)` (or `resolve_float` / `resolve_int`), which maps the -1 sentinel and any invalid stored value to the catalog default
- Implements `kill()`

The split exists so that `AbcBaseModel` can be imported and its classmethods called without loading the heavy model library — which matters both for startup speed and for running the app outside the model's venv.

---

## Voice Clone Cache

**Rationale.** A single audiobook run calls `generate_using_project()` many times — hundreds or thousands — with the same voice reference sample. As part of per-call voice setup, each model derives a reference-sample-specific intermediate representation by running that sample through one or more forward passes: codec/audio tokens, speaker embeddings, mel features, prompt conditionals, or (for Pocket) the full voice state. That representation depends only on the reference sample, not on the text being synthesized, so re-deriving it on every segment is pure waste. The cache memoizes it, keyed on the sample's identity, so on a hit the derivation is skipped entirely and the call does only the text-dependent generation (LLM decoding + acoustic head). Generation itself is never amortized — the cache saves only the repeated re-derivation of the voice setup. The size of that saving is model-dependent (a single encoder pass over a short clip vs. several extractor forward passes vs. full voice-state construction) and is largest in multi-voice runs, where without the cache every voice switch would re-derive that voice's intermediate representation.

The machinery is shared; the *contents* are not.

**The key.** `TtsBaseModel._make_voice_clone_cache_key(source_path, transcript)` returns `VoiceCloneCacheKey = (normalized_path, transcript, st_mtime_ns, st_size)`, where the path is resolved through `os.path.realpath` and normalized with `os.path.normcase`. The mtime/size components make an in-place modification of the voice file (same path, new contents) a cache miss, not just a path change. Models that do not use a transcript pass `""`.

**The get-or-create.** `TtsBaseModel._get_or_create_voice_clone(source_path, transcript, factory)`:

- The per-instance dict `_voice_clone_cache` is created lazily on the first call (the attribute does not exist until then; code must not assume it does).
- On a key hit, the cached value is returned as-is — no factory call.
- On a miss, `factory()` prepares the value. If the factory raises, nothing is cached and the model's caller converts the exception into the standard error string `Couldn't create voice clone for <path> - <ExcType>: <msg>`; the model's voice bookkeeping (e.g. `_voice_info`) is only updated after a successful preparation.
- On success, multi-voice models (`RETAINS_MULTIPLE_VOICE_CLONES = True`) evict any cached entries for the same normalized source path (an older transcript or file revision of that file is superseded) and other models clear the whole dict, so single-voice models retain only the most recently selected value.

`clear_voice_clone_cache()` drops the entire dict (idempotent, safe when the attribute was never created). Concrete `kill()` implementations call it before nulling out the model reference.

**The values are opaque.** Each concrete model decides what a prepared clone is and where its tensors live; the base class only keys, evicts, and clears. A consistent convention across the current implementations is that cached values are CPU copies of anything that would otherwise sit on the inference device, so the cache itself stays VRAM-neutral. Two documented exceptions: Pocket keeps its (single) voice state on-device, matching its pre-cache memory profile, and Chatterbox copies the cached CPU `Conditionals` on-device per call because the library mutates that state during generation.

**Continuation interplay.** Models with rolling-continuation history (Qwen3 base, Fish S2, MOSS) clear that history when the voice *identity* — the `(path, transcript)` pair — changes, including when the voice is removed. A mere file revision (same identity, new mtime/size) re-prepares the clone but preserves the history.

| Model | Multi-voice | Cached value | Key inputs | Notes |
|---|---|---|---|---|
| Qwen3 base | yes | `VoiceClonePromptItem` (ref codes + speaker embedding) | path + transcript | The reference pattern the other models follow |
| Chatterbox | yes | prepared `Conditionals` (T3 prompt token/feature) | path + transcript | |
| Fish S1 | yes | encoded voice tokens (`VoiceClone`) | path + transcript | tokens are encoded eagerly at preparation time; failure surfaces as the standard error string |
| Fish S2 | yes | encoded voice tokens (`VoiceClone`) | path + transcript | plus: rolling-continuation history cleared on voice-identity change |
| GLM | yes | extracted speech tokens/features + speaker embedding | path + transcript | |
| Higgs V2 | yes | encoded audio token ids | path + transcript string | the transcript may itself be a file path; the raw string is the key element and the file is re-read per call, so editing its contents does not invalidate the token cache |
| Mira | yes | encoded voice prompt | path | no transcript support |
| MOSS | yes | CPU audio codes | path (transcript always `""`) | rolling-continuation history cleared on any voice change; a no-voice call resets the active voice and history but leaves the prepared clones cached, like Higgs and Pocket |
| OmniVoice | yes | `VoiceClonePrompt` (float audio tokens) | path + reference text | only voice-clone generation uses the cache; voice-design and auto-voice modes never touch it |
| Pocket | no | model state derived from the audio prompt | resolved path | single slot; a bare predefined-voice name is resolved to its on-disk file for keying, but the original reference is what gets passed to the library |

Server variants (SGL-Omni) do no local inference and keep no clone cache. Three local models are not part of the mechanism, each for a different reason:

- **IndexTTS2** — the library's `infer()` already caches the derived voice intermediates (speaker embedding, style, prompt condition, reference mel) internally, keyed by path string only. The adapter tracks the standard file-revision keys (empty transcript) for the speaker and effective emotion references and invalidates the affected library path marker on an mtime/size or resolved-path change, including in-place crop edits. Unchanged files retain the library cache, so a separate tool-side conditioning cache remains redundant for single-voice runs. Multi-voice coexistence is unreachable from the tool: the library cache is a single slot that fully evicts on every voice switch, and `infer()` only accepts a path string — adopting it would require forking the `indextts` package, for a gain limited to multi-voice runs (the model supports no batching, so alternating lines re-derive the voice every line).
- **Breeze TTS 2** — caches the reference clip's codec codes (small CPU int16 tensors, `RETAINS_MULTIPLE_VOICE_CLONES`), keyed with an empty transcript since the codes don't depend on it. Upstream's `prepare_inputs()` encodes the clip once per request (and again for the CFG negative prompt) through the module-level `breeze_tts.infer.templates._encode_prompt_audio`; `BreezeModel._cached_reference_codes()` swaps that hook for a cache-backed one only for the duration of the `prepare_inputs()` call.
- **VibeVoice** — the library API accepts the raw voice file path and re-reads and re-tokenizes the reference clip inside `processor()` on every call; it exposes no prepared voice-clone artifact for the tool to precompute and reuse (voice tokens are fused with the text inside the processor, and passing pre-loaded audio data previously caused inference issues). The per-call overhead is negligible relative to segment generation, and VibeVoice retains no voice state between calls, so a cache would buy nothing.

Fake-library tests for the mechanism live in `tests/models/test_<model>_voice_clone_cache.py` (one per local model with a cache), most `importorskip`-gated on their model's library so the full suite stays runnable in `venv-base`. IndexTTS2's invalidation tests load the adapter with a path-caching library stub and run in `venv-base` without model weights.

---

## Voice and Model Menus

The status block's shared sample summary is labelled **Voices**: it shows the first voice with a sample (using the standard sample label/link), then `+N more` for all remaining voice slots, including sample-less ones. With no samples, optional-sample models show `1 item` / `N items`; sample-required models retain the error-coloured `required` warning. Pocket uses this same line: a voice's predefined voice counts like a sample and is shown by name (it takes precedence over that voice's sample). Model-specific non-sample summaries (e.g. voice 0's Qwen3 CustomVoice speaker) remain unchanged. This presentation is owned by `menus/menu_status.py`, not the model's generation/primary-voice selection.

The main menu separates **[V] Voices** from **[M] Model settings**. Both require a project and a selected TTS model; neither requires an existing voice sample. Exception: in server mode, the catalog TTS model picker is the first item in **Model settings**, so that menu opens even with no model selected.

- [`menus/voice/`](<../tts_audiobook_tool/menus/voice/>) contains [`VoiceMenuShared`](<../tts_audiobook_tool/menus/voice/voice_menu_shared.py>) and `voice_*_menu.py` modules only for local models that add voice behavior of their own (IndexTTS2, Mira, Pocket, Qwen3-TTS, VibeVoice); every other local model gets the shared default (`VoiceMenuShared.make_local_items`) from the `VoiceMenuShared.menu()` fallback. These menus list each voice as a menu item (so voice N has hotkey N; a project always has at least one, possibly without a sample), followed by **Add** (hidden at nine voices), then any model-specific items the menu adds (eg VibeVoice's LoRA, IndexTTS2's emotion controls, audio.cpp voice controls), and finally **Edit voice/line selections** and **Auto-assign character voices**. `VoiceMenuShared.make_voice_sample_items` builds only the voice list and **Add**; `VoiceMenuShared.menu_wrapper` always appends the selection controls (`make_voice_selection_items`, via `with_voice_selection_items`) last, so a model's own items land before them without per-model ordering code. An empty item list (eg an unresolved remote model) stays empty. A voice row shows its sample (or `no sample set`, in the normal menu color with an error-colored `required` annotation when the model requires a voice clone sample; for other models a voice without a sample uses the model's built-in voice, even alongside voices that have samples), and `missing transcript` when it has a sample but no effective transcript. **Add** appends a default voice and opens its submenu; it does not ask for a file. Selecting a voice opens its own submenu (`VoiceMenuShared.voice_sample_submenu`, headed `Voice N`): **Voice sample** (`ask_and_set_voice_sample`: validates the file, then saves it with the trim reset and the transcript cleared, then fills the transcript from the sidecar text or STT; a failed STT leaves the transcript missing), then, only when the voice has a sample, **Clear voice sample** (no confirmation; clears only that voice's sample, transcript and trim, preserving its slot/index, effects, presets and instructions; keeps the original file, cleans up an unshared managed trim after saving, and stays in the submenu), **Trim or play**, and **Transcription** (with a preview of the effective transcript), then, while the model uses per-voice presets (`can_voice_presets`), the model's preset item (catalog `spec.ui.preset_label`: Pocket **Predefined voice**, Qwen3-TTS CustomVoice **Speaker**; choices come from the model menu's `preset_choices` callback, which may inspect the model) and its clear action when set (a preset takes precedence over the voice's sample: the voice list shows just the preset as the voice's value, and **Voice sample** notes the precedence), **Move voice sample position** (2+ voices; the submenu follows the voice), **Remove** (confirmed; exits the submenu; labelled **Remove (reset to default)** for the only voice, which is reset to a default voice, without confirmation when already at defaults), and, under **Effects**, the voice's **Stereo position** and **Room reverb**. Sample-specific callbacks/hints remain here and run around **Voice sample**, including Pocket's clone-access validation (skipped after its first success in an app run); Mira's unload-on-clear callback runs when a removal or successful sample clear leaves no voice with a sample. Local voice controls also live here: IndexTTS2's emotion reference/vector/strength under **Emotion**, OmniVoice's voice-design instructions, VibeVoice's LoRA selection, and Qwen3's checkpoint-specific instruction controls. Conditional clear actions follow their corresponding controls.
- [`menus/model/`](<../tts_audiobook_tool/menus/model/>) contains [`ModelMenuShared`](<../tts_audiobook_tool/menus/model/model_menu_shared.py>) and `model_*_menu.py` modules for local models whose Model settings menu is still hand-written (checkpoint/target selection or variant-dependent items: Chatterbox, dots.tts, MOSS, OmniVoice, Pocket, Qwen3-TTS, VibeVoice). Local models whose catalog entry declares `[[models.local.menu]]` (Breeze, Fish S1/S2, GLM, Higgs v2, IndexTTS2, Mira) have no module; see [Local catalog parameters and menus](#local-catalog-parameters-and-menus). Server models explicitly choose each settings control's menu in the catalog, as described below. MOSS's architecture-specific numeric helpers live in `model_moss_shared.py`.

The per-voice submenu queries `Tts.get_model_support(project).can_instructions(project)` on every redraw. When true, **Instructions/{model short name}** follows **Transcription**, or **Voice sample** for a sample-less voice; **Clear instructions** follows it only when the selected voice has a value for this exact model ID. The quoted preview is flattened and ellipsized to 40 characters (`none` when empty); editing uses the full prefill through `ask.ask_input(lower=False, prefill=...)` / `AskAdvanced.ask`. The optional string `[models.spec.ui].instructions_label` replaces the leading "Instructions" in that menu label (both OmniVoice entries use "Voice design"). The optional string `[models.spec.ui].instructions_message` replaces the editor's default prompt (`DEFAULT_INSTRUCTIONS_MESSAGE`); audio.cpp OmniVoice uses it to warn that invalid instruction tags will fail generation; both CosyVoice3 entries use it for an example instruction. The voice menus likewise follow the runtime `can_voice_clone(project)`: when false (eg Qwen3-TTS local CustomVoice/VoiceDesign checkpoints), the voice list omits sample details and the voice submenu hides Voice sample, Clear voice sample, Crop and Transcription, keeping any stored sample and transcript for a later Base checkpoint; voice preflight skips sample checks for the same reason. A list item with nothing else to show reads `no instructions set` when the model can't clone (under its settings) but supports instructions, else `no sample set` (`ProjectVoiceUtil.make_no_voice_value_text`; the status block's Voices line uses the same text, not error-colored, when no voice has a value and none is required). When instructions are supported but empty and `ModelSupport.expects_voice_instructions(project)` is true (Qwen3-TTS local VoiceDesign, where an empty description yields an arbitrary voice; CosyVoice3 on either backend in Instruct mode), the voice list item shows a non-blocking (so not error-colored) `no instructions set` hint; generation is never blocked. The voice list's instructions preview uses the same runtime `can_instructions(project)` gate, so mode-dependent models (eg CosyVoice3 outside Instruct) hide stored instructions there too. When `ModelSupport.uses_instructions_with_sample(project)` is false (SGL-Omni AuK/AuK-Flash), a voice with a sample still shows the Instructions item (so a description can be prepared before clearing the sample), with the sublabel "Not used while this voice has a voice sample" only while it has a value, and the voice list omits its preview. Blank/cancel and unchanged input do not save. Non-empty input is stripped and limited to 4,000 characters without truncation, then checked by `ModelSupport.validate_instructions(value)` (returns an error message or `""`; default accepts anything). Local OmniVoice (`OmniVoiceBaseModel`) overrides it to run OmniVoice's own `_resolve_instruct` via `tts_models/omnivoice_instruct_util.py`, importing OmniVoice lazily in the main process (no model is constructed); a rejection shows the resolver's message and saves nothing, and an unimportable OmniVoice skips the check. Validation never rewrites the value: the text is saved as entered and OmniVoice re-normalizes it at generation time. Set/Clear preserve other models' instructions and voice properties, follow the voice after a move, and roll back on save failure. The adapter's existing selected-voice routing handles generation; there is no global fallback.

Both shared classes route local models through literal catalog IDs and remote models through the selected definition. Their wrappers have separate headings/breadcrumbs and stop sample playback on exit; only the model wrapper adds the selected model's `settings_note`. Model controls retain their existing validation, reset/default semantics, storage ownership, and worker reload/rollback behavior.

### Definition-driven server menus

SGL-Omni and audio.cpp share one voice renderer ([`VoiceCatalogMenu`](<../tts_audiobook_tool/menus/voice/voice_catalog_menu.py>)) and one parallel model renderer ([`ModelCatalogMenu`](<../tts_audiobook_tool/menus/model/model_catalog_menu.py>)). Every settings control (numeric parameter, seed, or choice) requires `target_menu = "model"` for Model settings or `target_menu = "voice"` for Voice. A `choice` control edits a string parameter that declares a fixed `choices` list (`value`, `label`, optional `description`, parsed into shared `ParameterChoice` items) through a one-shot option submenu built by `ModelMenuShared.make_choice_item()` for both backends. CosyVoice3's "Mode" (`template_name`) on each backend and audio.cpp Qwen3-TTS's "Model type" use this control. Audio.cpp `choice` menu controls accept an optional string `subheading`, displayed above the submenu options (empty when omitted). An audio.cpp choice is sent in `options`; an SGL-Omni choice (`ChoiceParameter`, held in `SglOmniModelDefinition.choices` apart from the numeric request `parameters`) has no `request_key` and is never sent. Choices are plain data; behavior that depends on the selected choice lives in a backend model behavior subclass (see below). All shipped settings controls currently target `"model"`; instructions are per-voice data, not model settings or settings-menu controls. The special `voice_samples` control has no `target_menu` and always expands in Voice; validators still require it exactly once in the combined declaration.

Both renderers preserve the catalog order of controls assigned to their destination and share each backend's settings-control expansion, so editors, hints, validation, and storage do not change with menu location. Menu `group` declarations and the Advanced heading are no longer supported; unrelated catalog storage groups remain supported. Both shared `make_remote_items()` factories resolve the currently selected model/backend on every redraw, after menu/status reconciliation, and return no stale controls for None/unknown selections.

The renderers are backend-agnostic: each definition is wrapped by a `CatalogMenuSource` adapter ([`catalog_menu_source.py`](<../tts_audiobook_tool/menus/model/catalog_menu_source.py>), `SglOmniMenuSource` / `AudioCppMenuSource`, chosen by `catalog_menu_source(definition)`), which supplies the controls, parameter lookups, stored-value get/set and the per-backend presentation choices: numeric edit style (`"standard"`: the stock `ask.ask_number_and_save` editor with the suffix on its own line; `"sentinel_reset"`: SGL-Omni's editor where the sentinel resets to default and `max_request_value` caps input), the MOSS temperature hint, the seed's batch warning, and voice-sample preset choices (audio.cpp Qwen3-TTS speakers). `LocalMenuSource` adapts local catalog declarations (below). Another backend plugs in by adding an adapter and a factory branch; the renderers do not change.

### Local catalog parameters and menus

A local catalog entry may describe settings it already declares in `settings` with `[models.local.parameters.<name>]` ([`local_definition.py`](<../tts_audiobook_tool/tts_models/local_definition.py>), parsed by `parse_local_declaration` in `model_catalog.py`):

- **Numeric** (`float`/`int`): `default`, `min`, `max`, optional `default_sentinel = -1` (storage keeps -1 for "use default", so the setting's storage default must be -1) and `input_prompt_suffix`. Without a sentinel, the storage default must equal the parameter default.
- **Choice**: a `bool` (an implicit True/False choice) or any type with typed `choices = [{ value, label, description? }]`. Menus key options by `str(value)`; `LocalChoiceParameter` converts back to the typed value.

Numeric local parameters also supply load-time bounds for saved projects (as server parameters do), taking precedence over `BORROWED_PARAMETER_BOUNDS`. An optional `[[models.local.menu]]` makes the whole Model settings menu catalog-driven: `ModelMenuShared.menu()` routes such a model to `make_local_items()` before the per-model `match`. Controls render only in Model settings (no `target_menu`): numeric parameter items, `kind = "choice"`, `kind = "rolling_continuation"` (an int parameter whose `max` is the maximum length; its optional `subheading` is the prerequisite line) and at most one `kind = "seed"` (optional `batch_warning`). Parameters may be left out of a menu (eg IndexTTS2's `emo_alpha`, edited in its Voices menu). Local numeric items use the stock editor with -1 resetting to the default (`"standard_reset"`); a saved choice re-syncs `Tts.set_model_params_using_project`, so construction parameters (compile, fp16, sample rate) invalidate a loaded instance. Hand-written local menus build numeric items from the catalog with `ModelCatalogMenu.make_local_parameter_item(state, model_id, name, label)` and keep only their conditional logic. MOSS (per-variant `MossConfigs`) and Qwen3-TTS (defaults read from the loaded checkpoint) are not catalog-described yet.

### audio.cpp model behavior layer

The catalog declares plain data for audio.cpp entries: parameters, bounds, request placement, menu order, and simple flags shared by several models. Behavior that depends on other settings does not get a new catalog key; it goes in an optional `AudioCppModelBehavior` subclass ([`audio_cpp_behavior.py`](<../tts_audiobook_tool/tts_models/audio_cpp_behavior.py>)), registered by catalog model ID (not audio.cpp's server-side `family`) in `get_audio_cpp_behavior()`. Entries without a subclass use the base class, which reproduces purely catalog-driven behavior.

Hooks are narrow, not whole-method overrides: `uses_reference_transcript(values)` decides whether a transcript is required and sent; `uses_voice_instructions(values)` decides whether the selected voice's instructions are sent; `expects_voice_instructions(values)` decides whether empty instructions get the non-blocking voice-list hint; `get_ignored_parameters(values)` omits model parameters the server does not read; and `adjust_payload(payload, values)` permits a final request edit without changing the adapter-owned `model`, `input`, `response_format`, `seed`, `voice_ref` or `reference_text`. Readiness blockers (`get_blocking_issues(project, values)`) and warnings (`get_warning_issues(project, values)`) run only after all model parameter values are valid. The shared adapter owns request building, voice/transcript validation and seed handling. `AudioCppModelSupport` constructs the behavior, so subclasses run in the main process and must stay lightweight. A subclass declares the catalog parameters it reads and their kind in `REQUIRED_PARAMETERS`; construction rejects missing or retyped parameters.

CosyVoice3 ([`audio_cpp_behavior_cosyvoice3.py`](<../tts_audiobook_tool/tts_models/audio_cpp_behavior_cosyvoice3.py>)) keeps its model-wide Mode choice: Zero-shot uses the reference transcript; Cross-lingual and Instruct omit it. Only Instruct sends the selected voice's instructions. Every mode still requires a reference sample. The shared per-voice instruction editor appears only in Instruct mode, where a voice without instructions also gets the non-blocking `no instructions set` hint. Per-voice instruction validation is deferred; there is no model-global instruction editor or empty-global-instruction blocker.

Qwen3-TTS ([`audio_cpp_behavior_qwen3.py`](<../tts_audiobook_tool/tts_models/audio_cpp_behavior_qwen3.py>)) adds project-aware `can_voice_clone`, `can_voice_presets` and `requires_voice_preset` behavior hooks. Only Base consumes a sample and its required transcript, and only CustomVoice consumes the selected voice's preset (`options.voice_id`). CustomVoice/VoiceDesign consume instructions. The adapter must not resolve or decode inactive samples, even if their stored paths are missing. `get_ignored_parameters` removes the app-only model-type selector. `adjust_payload` mirrors `temperature`/`top_k`/`top_p` to `options.subtalker_*`, matching the local backend, so they govern every codebook rather than only the main talker's codebook 0. `resolve_language` maps normalized project ISO codes to full Qwen language names, also accepted by older servers that reject ISO codes; warnings use the same resolved hint. The default implementations preserve other families' behavior.

The single `qwen3tts_audiocpp` family type matches offline `tts`/`clon`/`vdes` entries and uses the persisted exact-server-entry selection. Its **Model type** remains manually overridable and is not authoritative discovery or checkpoint loading. On explicit picker selection only, a unique matching entry whose path basename exactly matches a known audio.cpp GGUF download filename sets and saves the type, then shows an accented confirmation. Unknown filenames retain the manual value; opening the picker, project loading and rebinding never apply the heuristic. Model settings are private to this catalog ID (as are those of local and SGL-Omni Qwen3) and remain shared across its exact server entries. CustomVoice/VoiceDesign instructions and CustomVoice speakers are keyed by this catalog ID on each `Voice`; speakers offer official defaults plus manual custom names, without claiming checkpoint inventory discovery. Switching types retains inactive data. Base always uses ICL cloning, matching the local backend: both a sample and its matching transcript are required, and every Base request explicitly sends `options.x_vector_only_mode = false`. There is no clone-mode setting; retired stored values are ignored by catalog reconciliation. All requests pin `text_chunk_size = 100000`; no batching, streaming or rolling continuation is enabled. Like the local backend, there is no max-tokens setting: the server uses the checkpoint's `generation_config.json` frame limit (8192 on official checkpoints, 2048 when absent), and retired stored values are ignored. See [user setup](<../docs/tips-audio-cpp.md#qwen3-tts>) and [live smoke script](<../testx/qwen3_audio_cpp.py>).

### SGL-Omni model behavior layer

SGL-Omni has the same optional layer: `SglOmniModelBehavior` ([`sgl_omni_behavior.py`](<../tts_audiobook_tool/tts_models/sgl_omni_behavior.py>)), registered by catalog model ID in `get_sgl_omni_behavior()` and constructed by `ConfiguredModelSupport`. It has only the hooks a shipped entry needs so far: `uses_reference_transcript(values)` (default: transcript policy is not `omitted`), `uses_voice_instructions(values)` (default: `[models.sgl_omni.behavior].voice_instructions`), `expects_voice_instructions(values)` (default False) and `uses_instructions_with_sample()` (default True; False drops the instructions of a voice that has a sample). `values` are the resolved numeric parameters plus choices (`ConfiguredModelSupport.get_values()`); `REQUIRED_PARAMETERS` checks a subclass's `"number"`/`"choice"` parameters at construction. When used, instructions are sent stripped as top-level `instructions`, which is a reserved request key no catalog parameter or default may claim.

The SGL-Omni CosyVoice3 server infers its prompt mode from the request fields (`instructions` -> instruct, reference `text` -> zero-shot, neither -> cross-lingual) and answers HTTP 400 when a transcript and instructions are combined. `CosyVoice3SglOmniBehavior` ([`sgl_omni_behavior_cosyvoice3.py`](<../tts_audiobook_tool/tts_models/sgl_omni_behavior_cosyvoice3.py>)) makes that choice explicit with the same app-side Mode as audio.cpp: Zero-shot sends (and the pre-flight fills in) the transcript, Cross-lingual sends neither, Instruct sends only the selected voice's instructions. Blank instructions in Instruct send neither, which the server treats as cross-lingual. Instructions are keyed by exact model ID, so the two CosyVoice3 backends do not share them. [`test_cosyvoice3_sgl_omni_entry_points.py`](<../tests/models/test_cosyvoice3_sgl_omni_entry_points.py>) mirrors the audio.cpp entry-point test.

AuK and AuK-Flash have no Mode: the SGL-Omni server picks the route per request from whether reference audio is present, so the app picks it per voice (`AukSglOmniBehavior`, [`sgl_omni_behavior_auk.py`](<../tts_audiobook_tool/tts_models/sgl_omni_behavior_auk.py>)), and one project can mix cloned and described voices. A voice with a sample is cloned; the server ignores `instructions` then (verified byte-identical), so `uses_instructions_with_sample()` is False and they are not sent. A sample-less voice (`voice_required = false`) sends its instructions as the voice description; blank sends none and the server uses its default ("A clear, natural voice."). Without reference audio the server requires `stage_params.auk_engine.gen_seconds`, and neither it nor upstream's core inference estimates one, so the `auk_seconds` prompt policy always sends upstream AuK's Prompt Enhancer "ordinary speaking" baseline ([`auk_duration.py`](<../tts_audiobook_tool/tts_models/auk_duration.py>), a faithful port of `_estimate_f5_instruct_duration()` without its LLM adjust step: seconds per UTF-8 byte by script, 0.0656 English / 0.0803 Chinese, a short-text factor under 10 bytes, F5 frame truncation), divided by Speech speed. The project language code is only upstream's fallback for text with neither CJK characters nor English words. With a sample, duration is unchanged: the server derives it from the sample and transcript at the default speed, and the app sends the same ratio divided by speed otherwise. Described voices are not stable across segments (the seed and the per-segment text both change the speaker; a fixed seed does not hold it), so in-play described voices get the AuK-specific run note (`instructions_voice_variation_note`) in place of the generic one (blank voices still get the generic note). Instructions are keyed by exact model ID, and AuK and AuK-Flash share no model settings either (speed, seed and concurrency are private to each ID). [`test_auk_sgl_omni_voice_routes.py`](<../tests/models/test_auk_sgl_omni_voice_routes.py>) covers both routes.

A static capability that a hook can override must be read at runtime through a project-aware query, never straight from the registry or definition. For the transcript, that query is `Tts.requires_reference_transcript(project)`: False when the model has no transcript storage, otherwise the model support's `uses_reference_transcript(project)` (part of `ModelSupport`). Local models default to their storage binding in `TtsBaseModel`; `MossBaseModel` answers False (it clones from audio alone), `Qwen3BaseModel` True only for the Base checkpoint, and `ConfiguredModelSupport` (SGL-Omni, given transcript storage) and `AudioCppModelSupport` ask their behavior (falling back to the static storage/catalog flag while a stored value is invalid). The voice pre-flight (`validate_voices` / `get_voice_problems`), standalone server startup (`is_missing_required_transcript`) and the adapter all use it. The catalog `reference_transcript` flag and `REGISTRY.transcript_binding()` still decide transcript storage, import and editing, so switching CosyVoice3 back to Zero-shot finds the transcript intact. [`test_cosyvoice3_audio_cpp_entry_points.py`](<../tests/models/test_cosyvoice3_audio_cpp_entry_points.py>) and [`test_cosyvoice3_sgl_omni_entry_points.py`](<../tests/models/test_cosyvoice3_sgl_omni_entry_points.py>) check menu visibility, readiness, pre-flight, server startup and the request together for each Mode.

An audio.cpp `choice` control whose stored value is invalid shows that value (in the error color) rather than the default, and opens its submenu with nothing selected, so choosing any option, including the default, saves and repairs storage.

Rule of thumb: a catalog key is fine for a plain value more than one model could use that does not depend on other settings; "if X then Y" belongs in a subclass. Older single-model keys (`language_policy = "moss"`, `language_hint_note`) predate this layer and are candidates to move into subclasses.

### Qwen3 checkpoint types

The voice menu always offers shared sample/selection management, including CustomVoice and VoiceDesign checkpoints; opening it does not inspect/load a model. The stored checkpoint type controls whether each voice's CustomVoice **Speaker** and **Instructions**, or VoiceDesign's **Instructions**, appear in its submenu. Selecting **Speaker** inspects the current checkpoint on demand for supported speakers (a single-speaker checkpoint's only speaker is saved directly); inspection errors do not block sample management. Speakers are per voice (`Voice.presets`); changing the checkpoint keeps them, and readiness with a loaded checkpoint flags any it lacks. Instructions and their clear action do not need a loaded model. The model menu still obtains worker inspection metadata for checkpoint type and generation defaults. Checkpoint changes refresh its captured metadata; clearing a custom target restores Base metadata before redraw. Managing samples does not change which inputs a checkpoint consumes.

### Storage remains independent of menu location

Primary voice settings remain one ordered list of `{file_name, transcript}` pairs in `Project.voice_references`, not separate per-model filename/transcript lists. Append, replace, reorder and clear preserve pair alignment, and transcripts stay retained for models that do not consume them; transcript editing is offered only for models that use them. A voice's Pocket predefined voice (`Voice.presets`) takes precedence over that voice's sample; setting either leaves the other in place. IndexTTS2's secondary emotion clip remains model-specific. Non-voice controls continue using `SettingRef` and the settings registry; this UI split introduces no storage migration.

---

## Integration Points — Where New Models Must Be Wired In

Implementing the class hierarchy and voice/model menus is necessary but not sufficient. The following locations contain explicit per-model dispatching that does not auto-discover new additions. Each must be updated when adding a new model (Consider devising abstraction patterns for some of these).

### Model catalog

Add a new entry to [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>) with a stable backend-suffix ID and fully populated spec/settings. No model-named attribute or declaration is added to `TtsModelType`; the canonical handle is installed from the catalog and retrieved by ID.

### `tts_audiobook_tool/tts.py`

The handle-keyed `Tts._MODEL_REGISTRY` needs one entry, mapping strict lookup of the new literal catalog ID to `(AbcBaseModel, Tts.get_abc, "_abc")`. For example, the GLM key is `TtsModelType.require_by_id("glm_local")`. The shared registry backs class lookup, lazy instance creation, existing-instance lookup, and instance clearing; do not convert its keys to strings or create alias constants.

The factory function and the `Tts._abc` cached instance variable also need to be added as class members.

If the model has any constructor parameters sourced from `Project` (device flags, sample rate, variant type, etc.), also update:

- **`Tts.get_model_params_using_project()`** ([tts.py](tts_audiobook_tool/tts.py)) — extract the relevant project fields into `model_params`
- **`Tts.set_model_params()`** ([tts.py](tts_audiobook_tool/tts.py)) — add a dirty-check comparison so that changing the param invalidates the cached instance

### Shared menu dispatchers

A local model needs no Voices menu code unless it adds voice behavior (extra items, sample callbacks or preset choices): the `VoiceMenuShared.menu()` fallback shows the shared voice-sample menu. Only then add a literal ID case (for example `case "pocket_local":` under `match model_type.id`) ([voice_menu_shared.py](<../tts_audiobook_tool/menus/voice/voice_menu_shared.py>)) that imports and calls the new `VoiceAbcMenu.menu(state)`. For the Model settings menu, prefer declaring `[models.local.parameters]` and `[[models.local.menu]]` in the catalog, which needs no dispatch. Only a hand-written menu needs a matching case in [`ModelMenuShared.menu()`](<../tts_audiobook_tool/menus/model/model_menu_shared.py>) for `ModelAbcMenu.menu(state)`.

### `tts_audiobook_tool/menus/voice/__init__.py`

Export any hand-written voice menu class, and any hand-written model menu class from `menus/model/__init__.py`.

### Project storage (`tts_audiobook_tool/tts_models/model_catalog.toml`)

Declare the model's non-voice persisted storage and voice/transcript capability in its catalog entry. `catalog_settings.parse_model_settings()` derives declarations from the entry's backend/parameter tables plus the explicit `settings` array, and `ModelSettingsRegistry.register_model_settings()` installs the bindings. `REGISTRY.voice_binding()` / `.transcript_binding()` remain compatibility/capability metadata; primary clone access uses the same project-wide ordered pairs for every model. V4 keeps non-voice overrides under `Project.model_settings`, with primary clones in top-level `Project.voice_references`; do not add another authoritative model-scoped clone list.

`ProjectVoiceUtil.set_voice_and_save()` / `set_voice_sample_at_index_and_save()` / `clear_voice_sample_at_index_and_save()` / `add_voice_and_save()` / `clear_voice_and_save()` ([project_voice_util.py](<../tts_audiobook_tool/project_support/project_voice_util.py>)) edit the shared project list for primary clones, including through compatibility model-setting accessors. Only true special cases remain model-specific (e.g. IndexTTS2's secondary emotion clip). Per-voice presets (`set_voice_preset_at_index_and_save()`) never clear a voice's sample.

### SGL-Omni variants (server mode only)

For a new SGL-Omni variant instead of a new local model: add a catalog entry with `backend_kind = "sgl_omni"`, matching rules and request/menu configuration. The canonical handle is available through ID lookup without Python declarations; definition-driven generation and menus use the shared backend adapter, not a new per-variant local factory. No new model venv or requirements file is needed — the existing server-mode venv (identified by the launcher-marker package) hosts server clients.
