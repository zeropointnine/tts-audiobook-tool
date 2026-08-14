# TTS Model Architecture

Last updated: 2026-08-14

## Overview

Each supported TTS model is isolated in its own Python virtual environment. The app discovers which local model is active at startup by probing for the presence of model-specific packages, then loads metadata, class references, and UI routing through a set of layered abstractions. Adding a new model means touching each of these layers; none of them auto-discover new additions.

Models come in two flavors:

- **Local models** — inference runs in-process; each has a dedicated venv (e.g. `venv-cb`, `venv-g`).
- **SGL-Omni server variants** — inference is delegated to an external SGL-Omni server over HTTP. Several model variants share one server-mode venv (`venv-client`) and one requirements file (`requirements-sgl-omni.txt`); the active variant is selected at runtime from the model name reported by the server's models endpoint (see `TtsModelSpec.is_sgl_omni` / `server_model_id_substring`).

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
requirements-oute.txt
requirements-pocket.txt
requirements-qwen3tts.txt
requirements-vibevoice.txt
requirements-sgl-omni.txt   # server mode; no model-specific deps (see below)
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

`requirements-sgl-omni.txt` is the exception to the two-section layout: it contains only the app dependencies, plus an artificial launcher-marker package (`./launcher_markers/sgl_omni`) that identifies the venv as the SGL-Omni server-mode venv. Server functionality itself requires no extra libraries beyond the base app deps.

The corresponding virtual environments (e.g. `venv-chatterbox`, `venv-fish-s1`, `venv-glm`) live at the project root and are selected externally when launching the app — the app itself has no venv-switching logic. One venv → one local model. `venv-client` is the shared SGL-Omni server-mode venv, and `venv-base` is a dev-only venv with just the app dependencies.

When implementing a new local model, create `requirements-<newmodel>.txt` first and validate it in isolation before wiring anything into the app. Copy the app dependencies block from `requirements-base.txt` (or an existing model file) and adjust `torch`/`torchaudio` versions only if the model library requires it.

When studying pre-existing model implementations, avoid Oute, which departs from the standard pattern in several ways (e.g. its "voice" is a JSON file, not an audio sample).

---

## `TtsModelSpec` and `TtsModelType`

**File:** [tts_audiobook_tool/tts_models/tts_model_type.py](tts_audiobook_tool/tts_models/tts_model_type.py)

`TtsModelSpec` is a `NamedTuple` holding all hardcoded, static properties for a model. `TtsModelType` is an `Enum` whose values are `TtsModelSpec` instances — it acts as the central registry of every supported model, including a `NONE` placeholder used when no model is detected.

Key fields of `TtsModelSpec` most relevant to integration:

| Field | Purpose |
|---|---|
| `id` | Stable string identifier used for serialization |
| `is_sgl_omni` | Whether the model is backed by an external SGL-Omni backend rather than local inference |
| `server_model_id_substring` | Substring matched against the SGL-Omni model name to select this variant (empty = not applicable) |
| `local_module_test` | Probe used to detect whether the model's library is installed in the active venv: a plain importable module name, or `dist:<package>[==<version>]` tested via `importlib.metadata` (the `dist:` form is how Fish S1 vs S2 disambiguate, since both ship as `fish-speech` at different versions) |
| `local_torch_devices` | Supported torch device types for local inference (empty for server variants and models that don't take a device) |
| `file_tag` | Short identifier used in generated filenames (e.g. `"glm"`, `"chatterbox"`) |
| `sample_rate` | Native output sample rate; the app resamples voice clone audio to this before saving |
| `max_words_default` / `max_words_reco_range` | App-recommended segment length settings for the model |
| `voice_target_attr` | Name of the `Project` attribute that stores the voice clone filename (empty if not applicable) |
| `requires_voice` | Whether generation is blocked without a voice clone |
| `voice_transcript_attr` | Name of the `Project` attribute for the voice clone transcript (empty = not needed) |
| `extra_file_attrs` | Additional `Project` attributes for model-specific saved files (e.g. IndexTTS2 emotion clip) |
| `batch_size_attr` | Name of the `Project` field for batch size (or concurrent requests for SGL-Omni variants); empty string means no batch support |
| `can_stream` | Whether the model supports streaming chunk callbacks |
| `semantic_trim_last` | Whether to run semantic trim at the end of the last word |
| `requires_ffmpeg_libs` | Whether the model requires FFmpeg shared libraries, not just the executable (usually because of TorchCodec) |
| `un_all_caps` | Force lowercase on all-caps prompts; set for models that perform poorly on them |
| `requirements_file_name` | The `requirements-<model>.txt` filename for this model |
| `ui` | Dict of UI strings: `proper_name`, `short_name`, `voice_path_console`, `voice_path_requestor`, `project_links` |
| `substitutions` | List of `(before, after)` string pairs applied to prompts before inference |

There is also a derived `can_batch` property.

### Model type detection

`Tts.init_local_model_type()` ([tts.py](tts_audiobook_tool/tts.py)) runs at startup and probes every non-SGL-Omni enum member's `local_module_test` in the current Python environment. SGL-Omni variants are skipped by local probing; instead `Tts.update_tts_type()` selects one by matching the model id reported by the SGL-Omni server's models endpoint against each variant's `server_model_id_substring` (see `TtsModelType.find_tts_type_using_sgl_omni_model_id()`). `Tts._sgl_omni_type` allows the user to override auto-detection.

> See [tts-type-refactor-todo.md](tts-type-refactor-todo.md) for a planned refactor to make the local/server backend distinction a formal concept rather than an `is_sgl_omni` flag.

---

## Two-Level Class Hierarchy

**Directory:** [tts_audiobook_tool/tts_models/](tts_audiobook_tool/tts_models/)

Every model is implemented with exactly two subclass levels. This is enforced by convention and documented in `TtsBaseModel`'s docstring. SGL-Omni server variants follow the same two levels; their "concrete" classes simply contain no heavy library imports at all, since inference goes through `SglOmniUtil` over HTTP (e.g. `fish_s2_server_base_model.py` / `fish_s2_server_model.py`).

### Level 1 — `TtsBaseModel` (abstract)

**File:** [tts_audiobook_tool/tts_models/tts_base_model.py](tts_audiobook_tool/tts_models/tts_base_model.py)

Defines the interface all models must satisfy:

- `INFO: TtsModelSpec` — class-level attribute; `__init_subclass__` raises `TypeError` if missing
- `kill() -> None` — abstract; nulls out internal model references to aid garbage collection
- `generate_using_project(project, prompts, force_random_seed, on_stream_chunk, on_stream_end, voice_selection_index) -> list[Sound] | str` — abstract; the main generation entry point (stream callbacks only used when `INFO.can_stream`)
- `massage_for_inference(text) -> str` — concrete; applies `INFO.substitutions`; subclasses may override-and-super
- `prepare_text_for_inference(project, text) -> str` — concrete; the full pre-inference pipeline: project word substitutions → generic prompt normalization (incl. `un_all_caps`) → `massage_for_inference`
- `clear_stream_state()` / `clear_continuation()` — concrete hooks for streaming and rolling-continuation state

Classmethods and helpers with default implementations (override when the defaults don't apply):

- `get_menu_text(project, instance) -> str`
- `get_blocking_issues(project, instance) -> list[ReadinessIssue]` (default: standard voice-clone blocker)
- `get_warning_issues(project) -> list[str]` — instance method (default: random-voice warning)
- `get_voice_tag(project) -> str`
- `get_voice_display_info(project, instance) -> VoiceDisplayInfo | None`
- `get_primary_voice_value(project) -> str`
- `get_missing_voice_file_issue(project, voice_file_name_attr) -> ReadinessIssue | None`
- `should_trim_trailing_token_noise(project, instance) -> bool`
- `can_hallucinate_music(project, instance) -> bool`
- `get_strictness_warning(strictness, project, instance) -> str`

### Level 2 — `AbcBaseModel(TtsBaseModel)`

Example: [tts_audiobook_tool/tts_models/glm_base_model.py](tts_audiobook_tool/tts_models/glm_base_model.py)

- Must **not** import any model library at module level
- Assigns `INFO = TtsModelType.###.value`
- Implements classmethods and any model-specific constants or static helpers
- This is the class registered in `Tts.get_class()` and used for all non-instance operations (readiness checks, voice display info, etc.)

```python
class GlmBaseModel(TtsBaseModel):
    INFO = TtsModelType.GLM.value
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

## Voice Menus

**Directory:** [tts_audiobook_tool/menus/voice/](tts_audiobook_tool/menus/voice/)

Each model has a dedicated voice menu module:

```
menus/voice/
  voice_menu_shared.py
  voice_chatterbox_menu.py
  voice_fish_s1_menu.py
  voice_fish_s2_menu.py
  voice_fish_s2_server_menu.py
  voice_glm_menu.py
  voice_higgs_v2_menu.py
  voice_higgs_v3_menu.py
  voice_indextts2_menu.py
  voice_mira_menu.py
  voice_moss_menu.py
  voice_moss_server_menu.py
  voice_moss_shared.py       # options shared between local MOSS and MOSS server
  voice_omnivoice_menu.py
  voice_oute_menu.py
  voice_pocket_menu.py
  voice_qwen3_menu.py
  voice_qwen3_server_menu.py
  voice_vibevoice_menu.py
  voice_zonos2_server_menu.py
```

### `VoiceMenuShared`

**File:** [tts_audiobook_tool/menus/voice/voice_menu_shared.py](tts_audiobook_tool/menus/voice/voice_menu_shared.py)

Contains shared operations used by most model menus:

- `menu(state)` — dispatches to the correct per-model menu via `match Tts.get_type()`
- `menu_wrapper(state, items, subheading)` — standardized menu heading and exit callback
- `make_resolved_voice_label(state)` — "Add voice sample …" status label
- `ask_and_set_voice_file(state, tts_type, is_secondary, message_override, append)` — prompts for a voice audio file, optionally gets its transcript, resamples it, and calls `ProjectVoiceUtil.set_voice_and_save()` (`append` adds to a multi-voice list rather than replacing)
- `ask_voice_file(default_dir_path, tts_type, message_override)` — prompts for the file path; uses `tts_type.value.ui` for display strings
- `make_clear_voice_item(state, info_item, callback)` — builds a menu item to clear the voice setting
- `make_seed_item(state, attr, prompt_override, add_batch_warning)` — builds a seed control menu item

### Per-model menu pattern

Each `VoiceAbcMenu.menu(state)` builds a list of `MenuItem`s and passes them to `VoiceMenuShared.menu_wrapper()`. Model-specific options (e.g. sample rate for GLM, emotion clip for IndexTTS2, model target/variant for MOSS) are added inline alongside the shared voice clone item. Shared operations like `ask_and_set_voice_file` and `make_clear_voice_item` accept a `TtsModelType` argument rather than being baked into the menu class.

Note that voice settings are multi-valued for most models: voice clone filenames (and transcripts) are stored as lists on `Project`, and the app auto-advances through them across generation calls via `voice_selection_index` (see `ProjectVoiceUtil` and `Tts.get_next_voice_selection_index()`).

---

## Integration Points — Where New Models Must Be Wired In

Implementing the class hierarchy and voice menu is necessary but not sufficient. The following locations contain explicit per-model dispatching that does not auto-discover new additions. Each must be updated when adding a new model (Consider devising abstraction patterns for some of these).

### `tts_audiobook_tool/tts_models/tts_model_type.py`

Add a new `TtsModelType` enum member with a fully populated `TtsModelSpec`.

### `tts_audiobook_tool/tts.py`

Several per-model maps/lists must each gain a new entry:

- **`Tts.get_class()` MAP** ([tts.py](tts_audiobook_tool/tts.py)) — maps `TtsModelType.ABC` → `AbcBaseModel`
- **`Tts.get_instance()` MAP** ([tts.py](tts_audiobook_tool/tts.py)) — maps `TtsModelType.ABC` → a factory function (e.g. `Tts.get_abc`) that lazily instantiates `AbcModel`
- **`Tts.get_instance_if_exists()` MAP** ([tts.py](tts_audiobook_tool/tts.py)) — maps `TtsModelType.ABC` → the cached instance variable `Tts._abc`
- **`Tts.instance_exists()`** — the list of cached instance variables
- **`Tts.clear_tts_model()`** — the list of instance variables nulled on invalidation

The factory function and the `Tts._abc` cached instance variable also need to be added as class members.

If the model has any constructor parameters sourced from `Project` (device flags, sample rate, variant type, etc.), also update:

- **`Tts.set_model_params_using_project()`** ([tts.py](tts_audiobook_tool/tts.py)) — extract the relevant project fields into `model_params`
- **`Tts.set_model_params()`** ([tts.py](tts_audiobook_tool/tts.py)) — add a dirty-check comparison so that changing the param invalidates the cached instance

### `tts_audiobook_tool/menus/voice/voice_menu_shared.py`

Add a `case TtsModelType.ABC:` branch to `VoiceMenuShared.menu()` ([voice_menu_shared.py](tts_audiobook_tool/menus/voice/voice_menu_shared.py)) that imports and calls the new `VoiceAbcMenu.menu(state)`.

### `tts_audiobook_tool/menus/voice/__init__.py`

Export the new menu class.

### `tts_audiobook_tool/project.py`

Add Pydantic field definitions for any new voice filename, transcript, seed, or other model-specific settings on the `Project` class. The voice filename and transcript attributes must match the names given in the model's `TtsModelSpec` (`voice_target_attr`, `voice_transcript_attr`).

Voice set/clear is no longer a per-model `match` block in `Project`: `ProjectVoiceUtil.set_voice_and_save()` / `clear_voice_and_save()` ([project_voice_util.py](tts_audiobook_tool/project_support/project_voice_util.py)) apply changes generically via `setattr` using the spec's attribute names. Only true special cases need explicit branches there (e.g. IndexTTS2's secondary emotion clip, Pocket's predefined-voice reset, Oute's JSON voice). Consequently, a mismatched or missing attribute name will surface at save time rather than failing loudly at dispatch.

### SGL-Omni variants (server mode only)

For a new SGL-Omni variant instead of a new local model: add the enum member with `is_sgl_omni=True`, a matching `server_model_id_substring`, `requirements_file_name="requirements-sgl-omni.txt"`, plus the `tts.py` entries above and a voice menu. No new venv or requirements file is needed — the existing `venv-client` (identified by the launcher-marker package) hosts all server variants.
