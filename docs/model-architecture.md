# TTS Model Architecture

## Overview

Each supported TTS model is isolated in its own Python virtual environment. The app discovers which model is active at startup by probing for the presence of a model-specific Python module, then loads metadata, class references, and UI routing through a set of layered abstractions. Adding a new model means touching each of these layers; none of them auto-discover new additions.

---

## Virtual Environments and Requirements Files

Every model has a dedicated `requirements-<model>.txt` at the project root:

```
requirements-chatterbox.txt
requirements-fish-s1.txt
requirements-fish-s2.txt
requirements-glm.txt
requirements-higgs.txt
requirements-indextts2.txt
requirements-mira.txt
requirements-oute.txt
requirements-qwen3tts.txt
requirements-vibevoice.txt
```

Each file follows the same two-section layout: model-specific dependencies appear at the top, followed by a `# App dependencies` comment and then the app's own dependencies. For example:

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

The app dependencies section is essentially identical across all requirements files. The one exception is `torch` and `torchaudio`, whose versions may differ between models depending on compatibility requirements of the model library (but we favor 2.8.0 as much as possible for consistency's sake)

The corresponding virtual environments (e.g. `venv-cb`, `venv-fish-s1`) live at the project root and are selected externally when launching the app — the app itself has no venv-switching logic. One venv → one model family.

When implementing a new model, create `requirements-<newmodel>.txt` first and validate it in isolation before wiring anything into the app. Copy the app dependencies block from an existing file and adjust `torch`/`torchaudio` versions only if the model library requires it.

When studying pre-existing model implementations, avoid Oute, which departs from the standard pattern in several ways.

---

## `TtsModelInfos` Enum

**File:** [tts_audiobook_tool/tts_model/tts_model_info.py](tts_audiobook_tool/tts_model/tts_model_info.py)

`TtsModelInfo` is a `NamedTuple` holding all hardcoded, static properties for a model. `TtsModelInfos` is an `Enum` whose values are `TtsModelInfo` instances — it acts as the central registry of every supported model.

Key fields of `TtsModelInfo` most relevant to integration:

| Field | Purpose |
|---|---|
| `module_test` | Python module name imported to detect whether the model's library is installed in the active venv |
| `file_tag` | Short identifier used in generated filenames (e.g. `"glm"`, `"chatterbox"`) |
| `sample_rate` | Native output sample rate; the app resamples voice clone audio to this before saving |
| `requirements_file_name` | The `requirements-<model>.txt` filename for this model |
| `voice_file_name_attr` | Name of the `Project` attribute that stores the voice clone filename (empty if not applicable) |
| `requires_voice` | Whether generation is blocked without a voice clone |
| `requires_voice_transcript` | Whether the model API needs the text transcript of the voice clone audio |
| `batch_size_project_field` | Name of the `Project` field for batch size; empty string means no batch support |
| `ui` | Dict of UI strings: `proper_name`, `short_name`, `voice_path_console`, `voice_path_requestor` |
| `substitutions` | List of `(before, after)` string pairs applied to prompts before inference |

`module_test` is how `Tts.detect_and_set_type()` auto-detects the active model on startup — it tries to import this module in the current Python environment. This means the field must be an importable module name present only when the model's library is installed (see the Fish S1 vs S2 distinction for a nuanced example).

---

## Two-Level Class Hierarchy

**Directory:** [tts_audiobook_tool/tts_model/](tts_audiobook_tool/tts_model/)

Every model is implemented with exactly two subclass levels. This is enforced by convention and documented in `TtsBaseModel`'s docstring.

### Level 1 — `TtsBaseModel` (abstract)

**File:** [tts_audiobook_tool/tts_model/tts_base_model.py](tts_audiobook_tool/tts_model/tts_base_model.py)

Defines the interface all models must satisfy:

- `INFO: TtsModelInfo` — class-level attribute; `__init_subclass__` raises `TypeError` if missing
- `kill() -> None` — abstract; nulls out internal model references to aid garbage collection
- `generate_using_project(project, prompts, force_random_seed) -> list[Sound] | str` — abstract; the main generation entry point
- `massage_for_inference(text) -> str` — concrete; applies `INFO.substitutions`; subclasses may override-and-super

Classmethods with default implementations (override when the defaults don't apply):

- `get_prereq_errors(project, instance, short_format) -> list[str]`
- `get_prereq_warnings(project) -> list[str]`
- `get_voice_tag(project) -> str`
- `get_voice_display_info(project, instance) -> tuple[str, str]`
- `get_strictness_warning(strictness, project, instance) -> str`

### Level 2 — `XxxBaseModel(TtsBaseModel)`

Example: [tts_audiobook_tool/tts_model/glm_base_model.py](tts_audiobook_tool/tts_model/glm_base_model.py)

- Must **not** import any model library at module level
- Assigns `INFO = TtsModelInfos.XXX.value`
- Implements classmethods and any model-specific constants or static helpers
- This is the class registered in `Tts.get_class()` and used for all non-instance operations (prereq checks, voice display info, etc.)

```python
class GlmBaseModel(TtsBaseModel):
    INFO = TtsModelInfos.GLM.value
    SAMPLE_RATES = [24000, 32000]
```

### Level 3 — `XxxModel(XxxBaseModel)`

Example: [tts_audiobook_tool/tts_model/glm_model.py](tts_audiobook_tool/tts_model/glm_model.py)

- Model library imports live here and **only** here
- Implements `__init__` (loads weights, sets up state)
- Implements `generate_using_project()` — reads voice file path, transcript, seed, etc. from `project`, then delegates to a more parameter-explicit internal method
- Implements `kill()`

The split exists so that `XxxBaseModel` can be imported and its classmethods called without loading the heavy model library — which matters both for startup speed and for running the app outside the model's venv.

---

## Voice Menus

**Directory:** [tts_audiobook_tool/voice_menu/](tts_audiobook_tool/voice_menu/)

Each model has a dedicated voice menu module:

```
voice_menu/
  voice_menu_shared.py       # shared utilities and dispatch
  voice_chatterbox_menu.py
  voice_fish_s1_menu.py
  voice_fish_s2_menu.py
  voice_glm_menu.py
  voice_higgs_menu.py
  voice_indextts2_menu.py
  voice_mira_menu.py
  voice_oute_menu.py
  voice_qwen3_menu.py
  voice_vibevoice_menu.py
```

### `VoiceMenuShared`

**File:** [tts_audiobook_tool/voice_menu/voice_menu_shared.py](tts_audiobook_tool/voice_menu/voice_menu_shared.py)

Contains shared operations used by most model menus:

- `menu(state)` — dispatches to the correct per-model menu via `match Tts.get_type()`
- `ask_and_set_voice_file(state, tts_type, is_secondary, message_override)` — prompts for a voice audio file, optionally gets its transcript, resamples it, and calls `project.set_voice_and_save()`
- `ask_voice_file(default_dir_path, tts_type, message_override)` — prompts for the file path; uses `tts_type.value.ui` for display strings
- `make_clear_voice_item(state, tts_type)` — builds a menu item to clear the voice setting
- `make_seed_item(state, field_name)` — builds a seed control menu item

### Per-model menu pattern

Each `VoiceXxxMenu.menu(state)` builds a list of `MenuItem`s and passes them to `VoiceMenuShared.menu_wrapper()`. Model-specific options (e.g. sample rate for GLM, emotion clip for IndexTTS2) are added inline alongside the shared voice clone item. Shared operations like `ask_and_set_voice_file` and `make_clear_voice_item` accept a `TtsModelInfos` argument rather than being baked into the menu class.

---

## Integration Points — Where New Models Must Be Wired In

Implementing the class hierarchy and voice menu is necessary but not sufficient. The following locations contain explicit per-model dispatching that does not auto-discover new additions. Each must be updated when adding a new model (Consider devising abstraction patterns for some of these).

### `tts_audiobook_tool/tts_model/tts_model_info.py`

Add a new `TtsModelInfos` enum member with a fully populated `TtsModelInfo`.

### `tts_audiobook_tool/tts.py`

Three `dict` maps must each gain a new entry:

- **`Tts.get_class()` MAP** ([tts.py](tts_audiobook_tool/tts.py)) — maps `TtsModelInfos.XXX` → `XxxBaseModel`
- **`Tts.get_instance()` MAP** ([tts.py](tts_audiobook_tool/tts.py)) — maps `TtsModelInfos.XXX` → a factory function (e.g. `Tts.get_xxx`) that lazily instantiates `XxxModel`
- **`Tts.get_instance_if_exists()` MAP** ([tts.py](tts_audiobook_tool/tts.py)) — maps `TtsModelInfos.XXX` → the cached instance variable `Tts._xxx`

If the model has any constructor parameters sourced from `Project` (device flags, sample rate, variant type, etc.), also update:

- **`Tts.set_model_params_using_project()`** ([tts.py](tts_audiobook_tool/tts.py)) — extract the relevant project fields into `model_params`
- **`Tts.set_model_params()`** ([tts.py](tts_audiobook_tool/tts.py)) — add a dirty-check comparison so that changing the param invalidates the cached instance

### `tts_audiobook_tool/voice_menu/voice_menu_shared.py`

Add a `case TtsModelInfos.XXX:` branch to `VoiceMenuShared.menu()` ([voice_menu_shared.py](tts_audiobook_tool/voice_menu/voice_menu_shared.py)) that imports and calls the new `VoiceXxxMenu.menu(state)`.

### `tts_audiobook_tool/voice_menu/__init__.py`

Export the new menu class.

### `tts_audiobook_tool/project.py`

Two `match tts_type` blocks must each gain a new case:

- **`Project.set_voice_and_save()`** ([project.py](tts_audiobook_tool/project.py)) — assign the saved filename (and transcript if applicable) to the correct `Project` fields
- **`Project.clear_voice_and_save()`** ([project.py](tts_audiobook_tool/project.py)) — clear those same fields

Both methods raise on `case _:`, so missing a new model will fail loudly at runtime.

> **Note:** `Project` also needs corresponding field definitions for any new voice filename, transcript, seed, or other model-specific settings. These are Pydantic fields on the `Project` class.
