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
| `ui` | Dict of UI strings/values: `proper_name`, `short_name`, `voice_path_console`, `voice_path_requestor`, `project_links`; optional positive `voice_sample_max_duration_s` is a warning-only recommendation, matching the upper duration in voice-path guidance |
| `output_filters` | Case-sensitive worker console substrings filtered out of output history |
| `substitutions` | List of `(before, after)` string pairs applied to prompts before inference |

Primary voice/transcript pairs are stored once in top-level `Project.voice_references`, shared by every model in that project. Catalog/registry voice and transcript bindings (`REGISTRY.voice_binding()`, `.transcript_binding()`) remain compatibility/capability metadata, not separate clone-list owners; model-specific support and transcript requirements still govern consumption. Parameters and batch size retain catalog-declared model/group storage, and `TtsModelType.can_batch()` derives from the orchestration binding. See [Project Spec v4](<project-spec-v4.md>) for storage and v1/v2/v3 migration.

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
- `get_blocking_issues(project, instance) -> list[ReadinessIssue]` (default: no voice-clone checks; voice state is validated by the interactive pre-flight `VoiceMenuShared.validate_voices` and at generation time)
- `get_warning_issues(project) -> list[str]` — instance method (default: random-voice warning)
- `get_voice_tag(project) -> str`
- `get_voice_display_info(project, instance) -> VoiceDisplayInfo | None`
- `get_primary_voice_value(project) -> str`
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

- **IndexTTS2** — the library's `infer()` already caches the derived voice intermediates (speaker embedding, style, prompt condition, reference mel) internally, keyed by path string only. The adapter tracks the standard file-revision keys (empty transcript) for the speaker and effective emotion references and invalidates the affected library path marker on an mtime/size or resolved-path change, including in-place crop edits. Unchanged files retain the library cache, so a separate tool-side conditioning cache remains redundant for single-voice runs. Multi-voice coexistence is unreachable from the tool: the library cache is a single slot that fully evicts on every voice switch, and `infer()` only accepts a path string — adopting it would require forking the `indextts` package, for a gain limited to multi-voice runs (the model supports no batching, so alternating lines re-derive the voice every line).
- **VibeVoice** — the library API accepts the raw voice file path and re-reads and re-tokenizes the reference clip inside `processor()` on every call; it exposes no prepared voice-clone artifact for the tool to precompute and reuse (voice tokens are fused with the text inside the processor, and passing pre-loaded audio data previously caused inference issues). The per-call overhead is negligible relative to segment generation, and VibeVoice retains no voice state between calls, so a cache would buy nothing.

Fake-library tests for the mechanism live in `tests/test_<model>_voice_clone_cache.py` (one per local model with a cache), most `importorskip`-gated on their model's library so the full suite stays runnable in `venv-base`. IndexTTS2's invalidation tests load the adapter with a path-caching library stub and run in `venv-base` without model weights.

---

## Voice and Model Menus

The main menu separates **[V] Voice clone** from **[M] Model settings**. Both require a project and a selected TTS model; neither requires an existing voice sample. Exception: in server mode, the catalog TTS model picker is the first item in **Model settings**, so that menu opens even with no model selected.

- [`menus/voice/`](<../tts_audiobook_tool/menus/voice/>) contains the per-model `voice_*_menu.py` modules and [`VoiceMenuShared`](<../tts_audiobook_tool/menus/voice/voice_menu_shared.py>). These menus display the enumerated sample list directly above the shared sample-management group: **Add voice sample**, **Remove voice sample**, **Edit voice sample transcript** when samples exist and the active model uses reference transcripts, conditional **Voice selection mode** for 2+ samples, and **Edit voice selections**. There is no intermediate Add/remove submenu; Add is hidden at nine samples. Sample-specific callbacks/hints remain here, including Pocket's clone-access validation (skipped after its first success in an app run) and Mira's unload-on-clear callback. Local voice controls also live here: IndexTTS2's emotion reference/vector/strength under **Emotion**, OmniVoice's voice-design instructions, Pocket's predefined voice selection, VibeVoice's LoRA selection, and Qwen3's checkpoint-specific speaker/instruction controls. Conditional clear actions follow their corresponding controls.
- [`menus/model/`](<../tts_audiobook_tool/menus/model/>) contains parallel `model_*_menu.py` modules and [`ModelMenuShared`](<../tts_audiobook_tool/menus/model/model_menu_shared.py>). Local models keep inference controls here: checkpoint/variant selection, sampling, seed, continuation, and compile/precision/sample-rate settings. Server models explicitly choose each settings control's menu in the catalog, as described below. MOSS's architecture-specific numeric helpers live in `model_moss_shared.py`.

Both shared classes route local models through literal catalog IDs and remote models through the selected definition. Their wrappers have separate headings/breadcrumbs and stop sample playback on exit; only the model wrapper adds the selected model's `settings_note`. Model controls retain their existing validation, reset/default semantics, storage ownership, and worker reload/rollback behavior.

### Definition-driven server menus

SGL-Omni and audio.cpp each have one voice renderer and one parallel model renderer. Every settings control (numeric parameter, seed, audio.cpp voice instructions, or audio.cpp choice) requires `target_menu = "model"` for Model settings or `target_menu = "voice"` for Voice clone. An audio.cpp `choice` control edits a string parameter that declares a fixed `choices` list (`value`, `label`, optional `description`) through a one-shot option submenu; CosyVoice3's "Mode" (`template_name`) is the only one so far. Choices are plain data; behavior that depends on the selected choice lives in an audio.cpp model behavior subclass (see below). Breeze TTS 2's audio.cpp Instructions and OmniVoice's audio.cpp Voice design instructions controls target `"voice"`; other shipped settings currently target `"model"`. The special `voice_samples` control has no `target_menu` and always expands in Voice clone; validators still require it exactly once in the combined declaration.

Both renderers preserve the catalog order of controls assigned to their destination and share each backend's settings-control expansion, so editors, hints, validation, and storage do not change with menu location. An instruction control's conditional Clear row follows its Edit row into the same menu. Menu `group` declarations and the Advanced heading are no longer supported; unrelated catalog storage groups remain supported. Both shared `make_remote_items()` factories resolve the currently selected model/backend on every redraw, after menu/status reconciliation, and return no stale controls for None/unknown selections.

### audio.cpp model behavior layer

The catalog declares plain data for audio.cpp entries: parameters, bounds, request placement, menu order, and simple flags shared by several models. Behavior that depends on other settings does not get a new catalog key; it goes in an optional `AudioCppModelBehavior` subclass ([`audio_cpp_behavior.py`](<../tts_audiobook_tool/tts_models/audio_cpp_behavior.py>)), registered by catalog model ID (not audio.cpp's server-side `family`) in `get_audio_cpp_behavior()`. Entries without a subclass use the base class, which reproduces purely catalog-driven behavior.

Hooks are narrow, not whole-method overrides: `uses_reference_transcript(values)` (whether the transcript is required and sent), `get_ignored_parameters(values)` (catalog parameters left out of the request because the server does not read them under these values; stored values are kept), `adjust_payload(payload, values)` (an escape hatch: a final in-place edit of the built request, which must not change the adapter-owned `model`, `input`, `response_format`, `seed`, `voice_ref` or `reference_text`; the adapter raises if it does), `get_blocking_issues(project, values)` (readiness blockers; evaluated on menu redraws, so no I/O, and rechecked by the adapter before sending), and `get_warning_issues(project, values)`. Blocker and warning hooks only run once every parameter value is valid. Two menu-facing hooks answer *what* to show without importing menu code: `is_control_visible(control, values)` hides a settings control (its stored value is kept) and `get_required_label(name, values)` returns status text (eg "required for Instruct") that replaces an empty instruction control's "(optional)", or None. Both audio.cpp menus evaluate them through `AudioCppMenuContext` ([`model_audio_cpp_menu.py`](<../tts_audiobook_tool/menus/model/model_audio_cpp_menu.py>)), which resolves values once per redraw; when any stored value is invalid it skips the hooks, so every control shows and a broken value can never hide the item that fixes it. The shared `AudioCppBackendAdapter` still owns request building, voice/transcript validation and seed handling. `AudioCppModelSupport` constructs the behavior, so subclasses run in the main process and must stay lightweight. A subclass declares the catalog parameters it reads, and their kind (`number`, `text` or `choice`), in `REQUIRED_PARAMETERS`; construction rejects a missing or retyped parameter, so a catalog rename fails at startup rather than mid-generation. CosyVoice3 ([`audio_cpp_behavior_cosyvoice3.py`](<../tts_audiobook_tool/tts_models/audio_cpp_behavior_cosyvoice3.py>)) is the first: only its `zero_shot` Mode sends the transcript, only `instruct` sends the instruction, Instruct without instructions is a blocker, and the Instructions control shows only in Instruct mode, labeled "required for Instruct" while empty. FireRedTTS3 ([`audio_cpp_behavior_fireredtts3.py`](<../tts_audiobook_tool/tts_models/audio_cpp_behavior_fireredtts3.py>)) maps the normalized project language code to the exact language tag its session requires (`"English"`, not `"en"`; omitted means Chinese), swapping it into `options.language` in `adjust_payload` and blocking languages with no tag. It is clone-only: the entry matches `tts`/`clon`, never `vdes`, and sends no `template_name`. It is designed for the Base package; a served Instruct package is accepted but runs a different path (its prompt has no language tag, so the tag only picks the text normalizer, and it has no speaker embedding), and the server does not reveal which variant is loaded, so nothing distinguishes them.

A static capability that a hook can override must be read at runtime through a project-aware query, never straight from the registry or definition. For the transcript, that query is `Tts.requires_reference_transcript(project)`: False when the model has no transcript storage, otherwise the model support's `uses_reference_transcript(project)` (part of `ModelSupport`). Local models default to their storage binding in `TtsBaseModel`; `MossBaseModel` answers False (it clones from audio alone), `Qwen3BaseModel` True only for the Base checkpoint, SGL-Omni entries follow their storage binding, and `AudioCppModelSupport` asks the behavior (falling back to the catalog flag while a stored value is invalid). The voice pre-flight (`validate_voices` / `get_voice_problems`), standalone server startup (`is_missing_required_transcript`) and the adapter all use it. The catalog `reference_transcript` flag and `REGISTRY.transcript_binding()` still decide transcript storage, import and editing, so switching CosyVoice3 back to Zero-shot finds the transcript intact. [`test_cosyvoice3_audio_cpp_entry_points.py`](<../tests/test_cosyvoice3_audio_cpp_entry_points.py>) checks menu visibility, readiness, pre-flight, server startup and the request together for each Mode.

An audio.cpp `choice` control whose stored value is invalid shows that value (in the error color) rather than the default, and opens its submenu with nothing selected, so choosing any option, including the default, saves and repairs storage.

Rule of thumb: a catalog key is fine for a plain value more than one model could use that does not depend on other settings; "if X then Y" belongs in a subclass. Older single-model keys (`language_policy = "moss"`, `language_hint_note`) predate this layer and are candidates to move into subclasses.

### Qwen3 checkpoint types

The voice menu always offers shared sample/selection management, including CustomVoice and VoiceDesign checkpoints; opening it does not inspect/load a model. The stored checkpoint type controls whether CustomVoice's **Set speaker** and **Instructions**, or VoiceDesign's **Instructions**, appear after the sample controls. Selecting **Set speaker** inspects the current checkpoint on demand for supported speakers; inspection errors do not block sample management. Once inspected, speaker labels retain single-speaker resolution and invalid-ID warnings. Instructions and their clear action do not need a loaded model. The model menu still obtains worker inspection metadata for checkpoint type and generation defaults. Checkpoint changes refresh its captured metadata; clearing a custom target restores Base metadata before redraw. Managing samples does not change which inputs a checkpoint consumes.

### Storage remains independent of menu location

Primary voice settings remain one ordered list of `{file_name, transcript}` pairs in `Project.voice_references`, not separate per-model filename/transcript lists. Append, replace, reorder and clear preserve pair alignment, and transcripts stay retained for models that do not consume them; transcript editing is offered only for models that use them. Pocket's selected preset takes precedence without clearing the list; adding a Pocket clone clears its preset. IndexTTS2's secondary emotion clip remains model-specific. Non-voice controls continue using `SettingRef` and the settings registry; this UI split introduces no storage migration.

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

Add a literal ID case to `VoiceMenuShared.menu()` (for example `case "glm_local":` under `match state.project.get_tts_model_type().id`) ([voice_menu_shared.py](tts_audiobook_tool/menus/voice/voice_menu_shared.py)) that imports and calls the new `VoiceAbcMenu.menu(state)`. Add the matching settings dispatch in [`ModelMenuShared.menu()`](<../tts_audiobook_tool/menus/model/model_menu_shared.py>) for `ModelAbcMenu.menu(state)`.

### `tts_audiobook_tool/menus/voice/__init__.py`

Export the new voice menu class, and export its parallel model class from `menus/model/__init__.py`.

### Project storage (`tts_audiobook_tool/tts_models/model_catalog.toml`)

Declare the model's non-voice persisted storage and voice/transcript capability in its catalog entry. `catalog_settings.parse_model_settings()` derives declarations from the entry's backend/parameter tables plus the explicit `settings` array, and `ModelSettingsRegistry.register_model_settings()` installs the bindings. `REGISTRY.voice_binding()` / `.transcript_binding()` remain compatibility/capability metadata; primary clone access uses the same project-wide ordered pairs for every model. V4 keeps non-voice overrides under `Project.model_settings`, with primary clones in top-level `Project.voice_references`; do not add another authoritative model-scoped clone list.

`ProjectVoiceUtil.set_voice_and_save()` / `clear_voice_and_save()` ([project_voice_util.py](<../tts_audiobook_tool/project_support/project_voice_util.py>)) edit the shared project list for primary clones, including through compatibility model-setting accessors. Only true special cases remain model-specific (e.g. IndexTTS2's secondary emotion clip and Pocket's predefined-voice state). Selecting a Pocket preset must not clear shared references.

### SGL-Omni variants (server mode only)

For a new SGL-Omni variant instead of a new local model: add a catalog entry with `backend_kind = "sgl_omni"`, matching rules and request/menu configuration. The canonical handle is available through ID lookup without Python declarations; definition-driven generation and menus use the shared backend adapter, not a new per-variant local factory. No new model venv or requirements file is needed — the existing server-mode venv (identified by the launcher-marker package) hosts server clients.
