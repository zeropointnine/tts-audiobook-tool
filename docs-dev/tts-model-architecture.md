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
| `local_torch_devices` | Supported torch device types for local inference (empty for server variants and models that don't take a device) |
| `file_tag` | Short identifier used in generated filenames (e.g. `"glm"`, `"chatterbox"`) |
| `default_output_sample_rate` | Native/default output sample rate; used directly by static-rate models and as the fallback for dynamically configured models |
| `requires_voice` | Whether generation is blocked without a voice clone |
| `can_stream` | Whether the model supports streaming chunk callbacks |
| `requires_ffmpeg_libs` | Whether the model requires FFmpeg shared libraries, not just the executable (usually because of TorchCodec) |
| `un_all_caps` | Force lowercase on all-caps prompts; set for models that perform poorly on them |
| `requirements_file_name` | The `requirements-<model>.txt` filename for this model |
| `ui` | Dict of UI strings: `proper_name`, `short_name`, `voice_path_console`, `voice_path_requestor`, `project_links` |
| `output_filters` | Case-sensitive worker console substrings filtered out of output history |
| `substitutions` | List of `(before, after)` string pairs applied to prompts before inference |

Voice/transcript storage, parameters and batch size are declared in the catalog and exposed through the settings registry (`REGISTRY.voice_binding()`, `.transcript_binding()`, `.orchestration_binding()`); `TtsModelType.can_batch()` derives from the orchestration binding.

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
- `get_blocking_issues(project, instance) -> list[ReadinessIssue]` (default: standard voice-clone blocker)
- `get_warning_issues(project) -> list[str]` — instance method (default: random-voice warning)
- `get_voice_tag(project) -> str`
- `get_voice_display_info(project, instance) -> VoiceDisplayInfo | None`
- `get_primary_voice_value(project) -> str`
- `get_missing_voice_file_issue(project) -> ReadinessIssue | None`
- `should_trim_trailing_token_noise(project, instance) -> bool`
- `can_hallucinate_music(project, instance) -> bool`

### Level 2 — `AbcBaseModel(TtsBaseModel)`

Example: [tts_audiobook_tool/tts_models/glm_base_model.py](tts_audiobook_tool/tts_models/glm_base_model.py)

- Must **not** import any model library at module level
- Assigns metadata through strict lookup, e.g. `INFO = TtsModelType.require_by_id("glm_local").value`
- Inherits `get_output_sample_rate(project, instance)` for static-rate models; models with configurable output rates, such as GLM, override it
- Implements classmethods and any model-specific constants or static helpers
- This is the class registered in `Tts.get_class_for_type()` and used for all non-instance operations (readiness checks, voice display info, etc.)

```python
class GlmBaseModel(TtsBaseModel):
    INFO = TtsModelType.require_by_id("glm_local").value
    SAMPLE_RATES = [24000, 32000]
```

### Level 3 — `AbcModel(AbcBaseModel)`

Example: [tts_audiobook_tool/tts_models/glm_model.py](tts_audiobook_tool/tts_models/glm_model.py)

- Model library imports live here and **only** here
- Implements `__init__` (loads weights, sets up state)
- Implements `generate_using_project()` — reads voice file path, transcript, seed, etc. from `project`, then delegates to a more parameter-explicit internal method
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

- **IndexTTS2** — the library's `infer()` already caches the derived voice intermediates (speaker embedding, style, prompt condition, reference mel) internally, keyed by path string only (no mtime/size check), so a tool-side cache would be redundant for single-voice runs. Multi-voice coexistence is unreachable from the tool: the library cache is a single slot that fully evicts on every voice switch, and `infer()` only accepts a path string — adopting it would require forking the `indextts` package, for a gain limited to multi-voice runs (the model supports no batching, so alternating lines re-derive the voice every line).
- **VibeVoice** — the library API accepts the raw voice file path and re-reads and re-tokenizes the reference clip inside `processor()` on every call; it exposes no prepared voice-clone artifact for the tool to precompute and reuse (voice tokens are fused with the text inside the processor, and passing pre-loaded audio data previously caused inference issues). The per-call overhead is negligible relative to segment generation, and VibeVoice retains no voice state between calls, so a cache would buy nothing.

Fake-library tests for the mechanism live in `tests/test_<model>_voice_clone_cache.py` (one per local model with a cache), each `importorskip`-gated on its model's library so the full suite stays runnable in `venv-base`.

---

## Voice Menus

**Directory:** [tts_audiobook_tool/menus/voice/](tts_audiobook_tool/menus/voice/)

Local models have dedicated voice menu modules. All SGL-Omni variants use one definition-driven module:

```
menus/voice/
  voice_menu_shared.py
  voice_configured_sgl_omni_menu.py
  voice_chatterbox_menu.py
  voice_fish_s1_menu.py
  voice_fish_s2_menu.py
  voice_glm_menu.py
  voice_higgs_v2_menu.py
  voice_indextts2_menu.py
  voice_mira_menu.py
  voice_moss_menu.py
  voice_moss_shared.py       # local MOSS architecture settings controls
  voice_omnivoice_menu.py
  voice_pocket_menu.py
  voice_qwen3_menu.py
  voice_vibevoice_menu.py
```

### `VoiceMenuShared`

**File:** [tts_audiobook_tool/menus/voice/voice_menu_shared.py](tts_audiobook_tool/menus/voice/voice_menu_shared.py)

Contains shared operations used by most model menus:

- `menu(state)` — selects the configured menu when a SGL-Omni definition is active; otherwise dispatches to a local per-model menu via `match state.project.get_tts_model_type().id` with literal string cases
- `menu_wrapper(state, items, subheading)` — standardized menu heading and exit callback
- `make_resolved_voice_label(state)` — "Add voice sample …" status label
- `ask_and_set_voice_file(state, tts_type, is_secondary, message_override, append)` — prompts for a voice audio file, optionally gets its transcript, resamples it, and calls `ProjectVoiceUtil.set_voice_and_save()` (`append` adds to a multi-voice list rather than replacing)
- `ask_voice_file(default_dir_path, tts_type, message_override)` — prompts for the file path; uses `tts_type.value.ui` for display strings
- `make_clear_voice_item(state, info_item, callback)` — builds a menu item to clear the voice setting
- `make_seed_item(state, attr, prompt_override, add_batch_warning)` — builds a seed control menu item

### Per-model menu pattern

Each local `VoiceAbcMenu.menu(state)` builds a list of `MenuItem`s and passes them to `VoiceMenuShared.menu_wrapper()`. Model-specific options (e.g. sample rate for GLM, emotion clip for IndexTTS2, model target/variant for MOSS) are added inline alongside the shared voice clone item. The SGL-Omni menu builds its items from the selected definition instead of dispatching to server-specific menus. Shared operations like `ask_and_set_voice_file` and `make_clear_voice_item` accept a `TtsModelType` argument rather than being baked into the menu class.

Note that voice settings are multi-valued for most models: voice clone filenames (and transcripts) are stored as lists on `Project`, and the app auto-advances through them across generation calls via `voice_selection_index` (see `ProjectVoiceUtil` and `Tts.get_next_voice_selection_index()`).

---

## Integration Points — Where New Models Must Be Wired In

Implementing the class hierarchy and voice menu is necessary but not sufficient. The following locations contain explicit per-model dispatching that does not auto-discover new additions. Each must be updated when adding a new model (Consider devising abstraction patterns for some of these).

### Model catalog

Add a new entry to [`model_catalog.toml`](<../tts_audiobook_tool/tts_models/model_catalog.toml>) with a stable backend-suffix ID and fully populated spec/settings. No model-named attribute or declaration is added to `TtsModelType`; the canonical handle is installed from the catalog and retrieved by ID.

### `tts_audiobook_tool/tts.py`

The handle-keyed `Tts._MODEL_REGISTRY` needs one entry, mapping strict lookup of the new literal catalog ID to `(AbcBaseModel, Tts.get_abc, "_abc")`. For example, the GLM key is `TtsModelType.require_by_id("glm_local")`. The shared registry backs class lookup, lazy instance creation, existing-instance lookup, and instance clearing; do not convert its keys to strings or create alias constants.

The factory function and the `Tts._abc` cached instance variable also need to be added as class members.

If the model has any constructor parameters sourced from `Project` (device flags, sample rate, variant type, etc.), also update:

- **`Tts.get_model_params_using_project()`** ([tts.py](tts_audiobook_tool/tts.py)) — extract the relevant project fields into `model_params`
- **`Tts.set_model_params()`** ([tts.py](tts_audiobook_tool/tts.py)) — add a dirty-check comparison so that changing the param invalidates the cached instance

### `tts_audiobook_tool/menus/voice/voice_menu_shared.py`

Add a literal ID case to `VoiceMenuShared.menu()` (for example `case "glm_local":` under `match state.project.get_tts_model_type().id`) ([voice_menu_shared.py](tts_audiobook_tool/menus/voice/voice_menu_shared.py)) that imports and calls the new `VoiceAbcMenu.menu(state)`.

### `tts_audiobook_tool/menus/voice/__init__.py`

Export the new menu class.

### Project storage (`tts_audiobook_tool/tts_models/model_catalog.toml`)

Declare the model's persisted storage in its catalog entry. `catalog_settings.parse_model_settings()` derives voice-reference, transcript, seed and orchestration storage from the entry's backend/parameter tables plus the explicit `settings` array, and `ModelSettingsRegistry.register_model_settings()` installs the bindings. Voice references are then detected through `REGISTRY.voice_binding()` / `.transcript_binding()`. Version 3 stores these under `Project.model_settings`; there are no new top-level `Project` fields and no spec attribute names to match.

Voice set/clear is not a per-model `match` block in `Project`: `ProjectVoiceUtil.set_voice_and_save()` / `clear_voice_and_save()` ([project_voice_util.py](tts_audiobook_tool/project_support/project_voice_util.py)) apply changes generically through `Project.set_model_setting()` using those bindings. Only true special cases need explicit branches there (e.g. IndexTTS2's secondary emotion clip, and Pocket's predefined-voice reset).

### SGL-Omni variants (server mode only)

For a new SGL-Omni variant instead of a new local model: add a catalog entry with `backend_kind = "sgl_omni"`, matching rules and request/menu configuration. The canonical handle is available through ID lookup without Python declarations; definition-driven generation and menus use the shared backend adapter, not a new per-variant local factory. No new model venv or requirements file is needed — the existing server-mode venv (identified by the launcher-marker package) hosts server clients.
